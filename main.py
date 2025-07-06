import os
import re
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage, PushMessageRequest
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from sqlalchemy import create_engine, text, Column, Integer, String, Text as AlchemyText
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からキーなどを取得
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
DATABASE_URL = os.environ.get('DATABASE_URL')

# 保存するメッセージの上限数
MAX_DOCUMENTS = 20000

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

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

# テーブルが存在しない場合に作成
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()
Base.metadata.create_all(engine)


# --- 既存のヘルパー関数群（変更なし） ---
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
        response = genai.embed_content(model="models/text-embedding-004",
                                       content=text_to_embed,
                                       task_type="RETRIEVAL_DOCUMENT")
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


# --- 【ここからが今回の改造の心臓部です】 ---

# 【新機能】Geminiに情報の「目利き（リランキング）」をさせる専門家
def rerank_documents(question, documents):
    if not documents:
        return []

    # Geminiに渡すためのプロンプトを作成
    rerank_prompt = f"""以下の「ユーザーの質問」と、それに関連する可能性のある「資料リスト」があります。
資料リストの中から、質問に答えるために**本当に重要度の高い資料**を、重要度順に最大5つ選び、その番号だけをカンマ区切りで出力してください。
例： 3,1,5,2,4

---
# ユーザーの質問
{question}

---
# 資料リスト
"""
    # 資料に番号を付けてプロンプトに追加
    for i, doc in enumerate(documents):
        rerank_prompt += f"【資料{i}】\n{doc.content}\n\n"
    
    try:
        # Geminiにリランキングを依頼
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(rerank_prompt)
        
        # Geminiの回答（例: "1,5,2"）から、番号のリストを抽出
        reranked_indices = [int(i.strip()) for i in response.text.split(',') if i.strip().isdigit()]
        
        # 抽出した番号順に、元のドキュメントを並べ替える
        reranked_docs = [documents[i] for i in reranked_indices if i < len(documents)]
        
        print(f"リランキング後のドキュメント順: {reranked_indices}")
        return reranked_docs

    except Exception as e:
        print(f"リランキング中にエラーが発生しました: {e}")
        # エラーが起きた場合は、元のリストの上位5件をそのまま返す
        return documents[:5]


# 【修正】質問応答関数を「リランキング方式」にアップグレード
def answer_question(question, user_id):
    session = Session()
    try:
        question_embedding = embed_text(question)
        if question_embedding is None:
            return "質問の解析に失敗しました。"

        # ステップ1：広く情報を集める（アシスタントの仕事）
        # 類似度の高い情報を多めに25件取得する
        candidate_docs = session.query(Document).order_by(Document.embedding.l2_distance(question_embedding)).limit(25).all()
        
        if not candidate_docs:
            return "まだ情報が十分に蓄積されていないようです。"

        # ステップ2：情報の「目利き」をさせる（司書の仕事）
        # 取得した25件の候補を、Geminiを使ってリランキングする
        final_results = rerank_documents(question, candidate_docs)

        if not final_results:
             # リランキングで何も選ばれなかった場合
            return "関連性の高い情報が見つかりませんでした。"

        # ステップ3：最終的な回答を生成する
        context = "\n".join(f"- {doc.content}" for doc in final_results)
        prompt = f"""以下の非常に精度の高い参考情報だけを使って、ユーザーの質問に簡潔に答えてください。

# 参考情報
{context}

# 質問
{question}

# 回答
"""
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
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

# メッセージを仕分ける、受付係（変更なし）
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    message_text = event.message.text
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        if message_text.startswith(("質問：", "質問:")):
            question = message_text.replace("質問：", "", 1).replace("質問:", "", 1).strip()
            answer = answer_question(question, user_id)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=answer)]
                )
            )
            return

        elif message_text == "DB確認":
            session = Session()
            total_count = session.query(Document).count()
            reply_text = f"現在のデータベース保存件数は {total_count} 件です。"
            session.close()
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )
            return

        urls = re.findall(r'https?://\S+', message_text)
        commentary = re.sub(r'https?://\S+', '', message_text).strip()

        if commentary:
            store_message(user_id, commentary)

        if urls:
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=f"メッセージと{len(urls)}件のURL、承知しました。内容を読んで記憶しますね。")]
                )
            )
            for url in urls:
                print(f"メッセージ内のURLを検出しました: {url}")
                scraped_data = scrape_website(url)
                if scraped_data and scraped_data['raw_text']:
                    cleaned_text = clean_text(scraped_data['raw_text'])
                    is_success = chunk_and_store_text(cleaned_text, scraped_data['title'], url)
                    if is_success:
                        line_bot_api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=f"【完了】URLの内容を記憶しました！\n{url}")]))
                    else:
                        line_bot_api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=f"【失敗】URLの読み込み・保存に失敗しました。\n{url}")]))
                else:
                    line_bot_api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=f"【失敗】URLへのアクセスに失敗しました。\n{url}")]))
        elif commentary:
             # URLがなく、純粋な会話だった場合は、ここで返信する（任意）
            pass
        else:
            # URLも文章もない（スタンプなど）の場合は何もしない
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
