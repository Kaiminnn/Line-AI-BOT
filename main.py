import os
import io # ← これを追加
import re
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
import threading # threading を追加
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, MessagingApiBlob, # MessagingApiBlob を追加
    ReplyMessageRequest, TextMessage, PushMessageRequest
)from linebot.v3.webhooks import MessageEvent, TextMessageContent
from sqlalchemy import create_engine, text, Column, Integer, String, Text as AlchemyText, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timezone
from PIL import Image
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent # ImageMessageContentを追加

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からキーなどを取得
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
DATABASE_URL = os.environ.get('DATABASE_URL')

# 保存するメッセージの上限数
MAX_DOCUMENTS = 20000
MAX_CHAT_HISTORY = 10 

# Flaskアプリの初期化
app = Flask(__name__)

# LINE Bot SDKの初期化
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Geminiの初期化
genai.configure(api_key=GEMINI_API_KEY)

# データベース設定
Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    content = Column(AlchemyText)
    embedding = Column(Vector(768))
    source = Column(String(2048), nullable=True)

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)
    content = Column(AlchemyText, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

Base.metadata.create_all(engine, checkfirst=True)


# --- ヘルパー関数群---
def describe_and_store_image(user_id, image_data):
    """
    画像データを受け取り、内容を説明する文章を生成してDBに保存する関数
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # Vision機能を持つモデルを指定

        # Geminiに渡すプロンプト（目的に応じて調整可能）
        prompt = [
            "この画像はユーザーから提供された記憶すべき情報です。画像の内容を、後から検索しやすいように客観的な事実に基づいて、詳細な日本語の文章で説明してください。",
            img,
        ]
        
        response = model.generate_content(prompt)
        image_description = response.text.strip()

        if not image_description:
            print("画像の分析に失敗しました。説明文が空です。")
            return False

        print(f"生成された画像の説明文: {image_description[:100]}...")

        # 生成した説明文を、通常のメッセージと同様にDBに保存
        session = Session()
        try:
            source_id = f"image_from_user:{user_id}"
            # 説明文にプレフィックスを付けて、画像由来の情報だと分かりやすくする
            content_to_store = f"ユーザーが送信した画像についての説明：\n{image_description}"
            embedding = embed_text(content_to_store)
            
            if embedding:
                document = Document(content=content_to_store, embedding=embedding, source=source_id)
                session.add(document)
                session.commit()
                print(f"ユーザー({user_id[:5]})の画像情報をDBに保存しました。")
                check_and_prune_db(session)
                return True
            else:
                return False

        except Exception as e:
            print(f"画像情報のDB保存中にエラーが発生しました: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    except Exception as e:
        print(f"画像の処理またはGemini API呼び出し中にエラーが発生しました: {e}")
        return False

def scrape_website(url):
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string if soup.title else 'No Title'
        main_content_tags = ["article", "main", ".main", ".content", "#main", "#content"]
        content_element = None
        for tag in main_content_tags:
            if soup.select_one(tag):
                content_element = soup.select_one(tag)
                break
        if not content_element: content_element = soup.body
        for script_or_style in content_element(["script", "style"]):
            script_or_style.decompose()
        text = content_element.get_text()
        return {"title": title, "raw_text": text}
    except Exception as e:
        print(f"URLのスクレイピング中にエラーが発生しました: {url} - {e}")
        return None

def clean_text(raw_text):
    lines = (line.strip() for line in raw_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  ") if len(phrase.strip()) > 20)
    cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
    return cleaned_text

def chunk_and_store_text(cleaned_text, title, source_url):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_text(cleaned_text)
    if not chunks: return False
    session = Session()
    try:
        for chunk in chunks:
            content_to_store = f"記事「{title}」より抜粋：\n{chunk}"
            embedding = embed_text(content_to_store)
            if embedding:
                document = Document(content=content_to_store, embedding=embedding, source=source_url)
                session.add(document)
        session.commit()
        print(f"URLの内容を {len(chunks)} 個のチャンクに分割してDBに保存しました。")
        check_and_prune_db(session)
        return True
    except Exception as e:
        print(f"チャンクのDB保存中にエラーが発生しました: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def embed_text(text_to_embed):
    try:
        response = genai.embed_content(model="models/text-embedding-004", content=text_to_embed, task_type="RETRIEVAL_DOCUMENT")
        return response['embedding']
    except Exception as e:
        print(f"ベクトル化中にエラーが発生しました: {e}")
        return None

def store_message(user_id, message_text):
    if not message_text or message_text.isspace(): return
    session = Session()
    try:
        source_id = f"user:{user_id}"
        embedding = embed_text(message_text)
        if embedding:
            document = Document(content=message_text, embedding=embedding, source=source_id)
            session.add(document)
            session.commit()
            print(f"ユーザー({user_id[:5]})のメッセージをDBに保存しました。")
            check_and_prune_db(session)
    except Exception as e:
        print(f"データベース処理中にエラーが発生しました: {e}")
        session.rollback()
    finally:
        session.close()

def check_and_prune_db(session):
    total_count = session.query(Document).count()
    if total_count > MAX_DOCUMENTS:
        items_to_delete_count = total_count - MAX_DOCUMENTS
        oldest_docs = session.query(Document).order_by(Document.id.asc()).limit(items_to_delete_count).all()
        for doc in oldest_docs:
            session.delete(doc)
        session.commit()
        print(f"上限を超えたため、古いメッセージを {items_to_delete_count} 件削除しました。")

def get_chat_history(session_id):
    session = Session()
    try:
        history = session.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.created_at.desc()).limit(MAX_CHAT_HISTORY).all()
        return history[::-1]
    finally:
        session.close()

def add_to_chat_history(session_id, role, content):
    session = Session()
    try:
        history_entry = ChatHistory(session_id=session_id, role=role, content=content)
        session.add(history_entry)
        session.commit()
    except Exception as e:
        print(f"チャット履歴の保存中にエラー: {e}")
        session.rollback()
    finally:
        session.close()

def rerank_documents(question, documents):
    if not documents: return []
    rerank_prompt = f"""以下の「ユーザーの質問」と、それに関連する可能性のある「資料リスト」があります。資料リストの中から、質問に答えるために本当に重要度の高い資料を、重要度順に最大5つ選び、その番号だけをカンマ区切りで出力してください。例： 3,1,5,2,4

# ユーザーの質問
{question}

# 資料リスト
"""
    for i, doc in enumerate(documents):
        rerank_prompt += f"【資料{i}】\n{doc.content}\n\n"
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(rerank_prompt)
        reranked_indices = [int(i.strip()) for i in response.text.split(',') if i.strip().isdigit()]
        reranked_docs = [documents[i] for i in reranked_indices if i < len(documents)]
        print(f"リランキング後のドキュメント順: {reranked_indices}")
        return reranked_docs
    except Exception as e:
        print(f"リランキング中にエラーが発生しました: {e}")
        return documents[:5]

def answer_question(question, user_id, session_id):
    history = get_chat_history(session_id)
    rephrased_question = question
    if history:
        history_text = "\n".join([f"{h.role}: {h.content}" for h in history])
        prompt = f"""以下は、ユーザーとの直近の会話履歴です。この文脈を踏まえて、最後の「新しい質問」を、データベース検索に最適な、具体的で自己完結した一つの質問に書き換えてください。もし新しい質問が既に具体的であれば、そのまま出力してください。

# 会話履歴
{history_text}

# 新しい質問
{question}

# 書き換えた検索用の質問：
"""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt)
            rephrased_question = response.text.strip()
            print(f"書き換えられた質問: {rephrased_question}")
        except Exception as e:
            print(f"質問の書き換え中にエラー: {e}")
            rephrased_question = question

    session = Session()
    try:
        question_embedding = embed_text(rephrased_question)
        if question_embedding is None: return "質問の解析に失敗しました。"
        candidate_docs = session.query(Document).order_by(Document.embedding.l2_distance(question_embedding)).limit(25).all()
        if not candidate_docs: return "まだ情報が十分に蓄積されていないようです。"
        final_results = rerank_documents(rephrased_question, candidate_docs)
        if not final_results: return "関連性の高い情報が見つかりませんでした。"

        context = "\n".join(f"- {doc.content}" for doc in final_results)
        final_prompt = f"""以下の非常に精度の高い参考情報だけを使って、ユーザーの質問に簡潔に答えてください。

# 参考情報
{context}

# 質問
{question}

# 回答
"""
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        final_response = model.generate_content(final_prompt)
        add_to_chat_history(session_id, 'model', final_response.text)
        return final_response.text
    except Exception as e:
        print(f"質問応答中にエラーが発生しました: {e}")
        return "申し訳ありません、応答の生成中にエラーが発生しました。"
    finally:
        session.close()

# LINEからのWebhookを受け取るエンドポイント
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 【ここを大幅修正】シンプルで堅牢な、新しい受付係
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    source = event.source
    session_id = source.group_id if source.type == 'group' else source.user_id
    user_id = source.user_id
    message_text = event.message.text
    reply_token = event.reply_token

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        # 仕事1：まずユーザーのメッセージをチャット履歴に保存する
        add_to_chat_history(session_id, 'user', message_text)

        # 仕事2：「質問」かどうかを最優先で判断し、そうなら素早く応答する
        if message_text.startswith(("質問：", "質問:")):
            question = message_text.replace("質問：", "", 1).replace("質問:", "", 1).strip()
            answer = answer_question(question, user_id, session_id)
            line_bot_api.reply_message(
                ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=answer)])
            )
            return

        # 仕事3：「DB確認」なら、すぐに応答する
        elif message_text == "DB確認":
            session = Session()
            doc_count = session.query(Document).count()
            history_count = session.query(ChatHistory).count()
            reply_text = f"長期記憶(Documents)の件数: {doc_count} 件\n短期記憶(ChatHistory)の件数: {history_count} 件"
            session.close()
            line_bot_api.reply_message(
                ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=reply_text)])
            )
            return

        # 仕事4：上記以外（通常の会話やURL共有）の場合
        else:
            # ユーザーの文章部分とURLを分離
            urls = re.findall(r'https?://\S+', message_text)
            commentary = re.sub(r'https?://\S+', '', message_text).strip()

            # 文章部分があれば、長期記憶に保存
            if commentary:
                store_message(user_id, commentary)

            # URLがあれば、処理を開始したことをまず返信する（時間切れ対策）
            if urls:
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=reply_token,
                        messages=[TextMessage(text=f"メッセージと{len(urls)}件のURL、承知しました。内容を読んで記憶しますね。")]
                    )
                )
                
                # 時間のかかるURL処理は、返信が終わった後でゆっくり行う
                for url in urls:
                    scraped_data = scrape_website(url)
                    if scraped_data and scraped_data['raw_text']:
                        cleaned_text = clean_text(scraped_data['raw_text'])
                        is_success = chunk_and_store_text(cleaned_text, scraped_data['title'], url)
                        
                        # 処理結果をプッシュメッセージで（任意で）通知
                        # is_success の結果に応じて通知内容を変えても良い
                    
            # URLも質問もない、純粋な会話の場合は、ここでは返信しない（静かな記録係に徹する）

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    """
    画像メッセージを受信したときの処理（タイムアウト対策版）
    """
    user_id = event.source.user_id
    reply_token = event.reply_token
    message_id = event.message.id

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★ ここからが重要な修正点です ★★★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        line_bot_blob_api = MessagingApiBlob(api_client) # コンテンツ取得用のAPIをインスタンス化

        try:
            # 1. まずユーザーに「処理中」であることを即座に返信する
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[TextMessage(text="画像を承知しました。内容を解析しますので、少しお待ちください。")]
                )
            )

            # 2. LINEサーバーから画像データを取得（正しいAPIオブジェクトから呼び出す）
            message_content = line_bot_blob_api.get_message_content(message_id=message_id)
            image_data = b''
            for chunk in message_content:
                print(f"DEBUG: chunkの中身: {chunk}, chunkの型: {type(chunk)}") # この調査用の行を追加
                image_data += chunk
            
            # 3. 時間のかかる処理を別のスレッド（バックグラウンド）で実行する
            thread = threading.Thread(
                target=process_image_and_push_result, 
                args=(user_id, image_data)
            )
            thread.start()

        except Exception as e:
            print(f"画像メッセージの受付処理中にエラーが発生しました: {e}")
            # ここでのエラーは即時返信が失敗した場合など
            # ユーザーへの通知は試みない（無限ループを避けるため）

def process_image_and_push_result(user_id, image_data):
    """
    【新設】バックグラウンドで画像処理と結果通知を行う関数
    """
    is_success = describe_and_store_image(user_id, image_data)

    if is_success:
        result_text = "先ほどの画像を記憶しました！この画像について質問があれば、いつでもどうぞ。"
    else:
        result_text = "申し訳ありません、先ほどの画像の解析・保存に失敗しました。"

    # ApiClientをこの関数内で再度作成してPush Messageを送る
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.push_message(
            PushMessageRequest(
                to=user_id,
                messages=[TextMessage(text=result_text)]
            )
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
