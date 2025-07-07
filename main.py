import os
import re
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from flask import Flask, request, abort

from io import BytesIO
from PIL import Image

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
# 【修正】荷物部門（Blob API）の専門家もインポートする
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, MessagingApiBlobApi, ReplyMessageRequest, TextMessage, PushMessageRequest
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent

from sqlalchemy import create_engine, text, Column, Integer, String, Text as AlchemyText, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timezone

# .envファイルから環境変数を読み込む
load_dotenv()

# （...ここから先の環境変数やデータベース設定のコードは、前回から変更ありません...）
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
DATABASE_URL = os.environ.get('DATABASE_URL')
MAX_DOCUMENTS = 20000
MAX_CHAT_HISTORY = 10 
app = Flask(__name__)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai.configure(api_key=GEMINI_API_KEY)
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


# --- ヘルパー関数群（ここは変更なし） ---
def scrape_website(url):
    # (省略...内容は前回と同じ)
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
    # (省略...内容は前回と同じ)
    lines = (line.strip() for line in raw_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  ") if len(phrase.strip()) > 20)
    cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
    return cleaned_text

def chunk_and_store_text(cleaned_text, title, source_url):
    # (省略...内容は前回と同じ)
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
    # (省略...内容は前回と同じ)
    try:
        response = genai.embed_content(model="models/text-embedding-004", content=text_to_embed, task_type="RETRIEVAL_DOCUMENT")
        return response['embedding']
    except Exception as e:
        print(f"ベクトル化中にエラーが発生しました: {e}")
        return None

def store_message(user_id, message_text):
    # (省略...内容は前回と同じ)
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
    # (省略...内容は前回と同じ)
    total_count = session.query(Document).count()
    if total_count > MAX_DOCUMENTS:
        items_to_delete_count = total_count - MAX_DOCUMENTS
        oldest_docs = session.query(Document).order_by(Document.id.asc()).limit(items_to_delete_count).all()
        for doc in oldest_docs:
            session.delete(doc)
        session.commit()
        print(f"上限を超えたため、古いメッセージを {items_to_delete_count} 件削除しました。")

def get_chat_history(session_id):
    # (省略...内容は前回と同じ)
    session = Session()
    try:
        history = session.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.created_at.desc()).limit(MAX_CHAT_HISTORY).all()
        return history[::-1]
    finally:
        session.close()

def add_to_chat_history(session_id, role, content):
    # (省略...内容は前回と同じ)
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
    # (省略...内容は前回と同じ)
    if not documents: return []
    # ...
    return documents[:5]

def answer_question(question, user_id, session_id):
    # (省略...内容は前回と同じ)
    # ...
    # この関数のロジックは前回から変更ありません
    history = get_chat_history(session_id)
    rephrased_question = question
    if history:
        history_text = "\n".join([f"{h.role}: {h.content}" for h in history])
        prompt = f"""...""" # 省略
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt)
            rephrased_question = response.text.strip()
        except Exception:
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
        final_prompt = f"""...""" # 省略
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

# テキストメッセージを処理する受付係（変更なし）
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    # (この関数は変更なし)
    # ...
    pass # 省略...内容は前回と同じ

# 【ここを修正】画像メッセージを処理する、新しい専門の受付係
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    source = event.source
    session_id = source.group_id if source.type == 'group' else source.user_id
    user_id = source.user_id
    message_id = event.message.id
    reply_token = event.reply_token

    try:
        with ApiClient(configuration) as api_client:
            # 郵便部門の担当者を準備
            line_bot_api = MessagingApi(api_client)
            
            # まずユーザーに、画像の処理を開始したことを素早く知らせる
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[TextMessage(text="画像を認識中です...")]
                )
            )

            # 【ここからが修正点】荷物部門の専門家を呼び出す
            line_bot_blob_api = MessagingApiBlobApi(api_client)
            # 荷物部門の専門家に、画像データのダウンロードを依頼する
            message_content = line_bot_blob_api.get_message_content(message_id=message_id)
            
            # Geminiに画像を渡せる形式に変換
            img = Image.open(BytesIO(message_content.read()))

            # Geminiに画像を渡して、その説明を生成させる
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(["この画像を日本語で詳しく、見たままに説明してください。", img])
            image_description = response.text.strip()

            # ユーザーの発言として、チャット履歴にも保存
            history_text = f"（画像が投稿されました。画像の内容： {image_description}）"
            add_to_chat_history(session_id, 'user', history_text)
            
            # 長期記憶にも、説明文を一つの情報として保存
            image_source = f"image_from_user:{user_id}"
            store_message(user_id, history_text) # store_messageを再利用する方がシンプル

            # 処理完了をプッシュメッセージで通知
            push_text = f"画像を記憶しました！\n\n【AIによる画像の説明】\n{image_description}"
            line_bot_api.push_message(
                PushMessageRequest(
                    to=session_id,
                    messages=[TextMessage(text=push_text)]
                )
            )
            
    except Exception as e:
        print(f"画像処理中にエラーが発生しました: {e}")
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.push_message(
                PushMessageRequest(
                    to=session_id,
                    messages=[TextMessage(text=f"画像の処理中にエラーが発生しました。")]
                )
            )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
