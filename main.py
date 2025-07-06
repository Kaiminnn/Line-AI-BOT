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

# 【変更点1】データベースの設計図に source 列を追加
class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    content = Column(AlchemyText)
    embedding = Column(Vector(768))
    source = Column(String(2048), nullable=True) # 出典元（URLやユーザーID）を保存する列

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

# テーブルが存在しない場合に作成
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()
Base.metadata.create_all(engine)


# URLからテキストを抽出する関数
def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        print(f"URLの読み取り中にエラーが発生しました: {url} - {e}")
        return None

# テキストをベクトル化する関数
def embed_text(text_to_embed):
    try:
        response = genai.embed_content(model="models/text-embedding-004",
                                       content=text_to_embed,
                                       task_type="RETRIEVAL_DOCUMENT")
        return response['embedding']
    except Exception as e:
        print(f"ベクトル化中にエラーが発生しました: {e}")
        return None

# 【変更点2】記憶関数に source を記録する機能を追加
def chunk_and_store_text(full_text, source_url):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(full_text)
    
    if not chunks:
        print("チャンクの生成に失敗しました。")
        return

    session = Session()
    try:
        for chunk in chunks:
            embedding = embed_text(chunk) # チャンクのみをベクトル化
            if embedding:
                # contentには元のチャンク、sourceにはURLを保存
                document = Document(content=chunk, embedding=embedding, source=source_url)
                session.add(document)
        session.commit()
        print(f"URLの内容を {len(chunks)} 個のチャンクに分割してDBに保存しました。")
        check_and_prune_db(session) # 上限チェック
    except Exception as e:
        print(f"チャンクのDB保存中にエラーが発生しました: {e}")
        session.rollback()
    finally:
        session.close()

def store_message(user_id, message_text):
    session = Session()
    try:
        source_id = f"user:{user_id}"
        embedding = embed_text(message_text)
        if embedding:
            # contentには会話本文、sourceにはユーザーIDを保存
            document = Document(content=message_text, embedding=embedding, source=source_id)
            session.add(document)
            session.commit()
            check_and_prune_db(session) # 上限チェック
    except Exception as e:
        print(f"データベース処理中にエラーが発生しました: {e}")
        session.rollback()
    finally:
        session.close()

# 【追加】DBの上限チェックと削除を共通関数化
def check_and_prune_db(session):
    total_count = session.query(Document).count()
    if total_count > MAX_DOCUMENTS:
        items_to_delete_count = total_count - MAX_DOCUMENTS
        oldest_docs = session.query(Document).order_by(Document.id.asc()).limit(items_to_delete_count).all()
        for doc in oldest_docs:
            session.delete(doc)
        session.commit()
        print(f"上限を超えたため、古いメッセージを {items_to_delete_count} 件削除しました。")


# 【変更点3】質問応答関数を「二段階検索」にアップグレード
def answer_question(question, user_id):
    session = Session()
    try:
        question_embedding = embed_text(question)
        if question_embedding is None:
            return "質問の解析に失敗しました。"

        # --- 第一段階：大まかな検索で、関連する情報源を探す ---
        initial_results = session.query(Document).order_by(Document.embedding.l2_distance(question_embedding)).limit(5).all()
        
        if not initial_results:
            return "まだ情報が十分に蓄積されていないようです。"

        # 見つかった情報源（source）の候補をリストアップ
        source_candidates = [doc.source for doc in initial_results if doc.source]
        
        # 最も可能性の高い情報源を決定（一番最初の結果のsourceを採用する簡単な方法）
        likely_source = source_candidates[0] if source_candidates else None
        
        print(f"質問に関連する可能性の高い情報源: {likely_source}")

        # --- 第二段階：情報源を絞って、深掘り検索 ---
        if likely_source:
            # 特定の情報源（URLやユーザー）に絞って、再度検索
            final_results = session.query(Document).filter(Document.source == likely_source).order_by(Document.embedding.l2_distance(question_embedding)).limit(5).all()
        else:
            # 特定できなければ、全体から探す
            final_results = initial_results

        # AIへのプロンプトを作成
        context = "\n".join(f"- {doc.content}" for doc in final_results)
        prompt = f"""以下の情報を参考にして、質問に答えてください。情報源がURLの場合、その記事の内容について答えているように振る舞ってください。

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

# メッセージイベントを処理するハンドラ
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    message_text = event.message.text
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        url_match = re.search(r'https?://\S+', message_text)

        if url_match:
            # ... (URL処理のコードは、簡潔さのためここでは省略。動作は同じ)
            url = url_match.group(0)
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=f"URLを読み込んでいます...")]))
            article_text = get_text_from_url(url)
            if article_text:
                chunk_and_store_text(article_text, url)
                line_bot_api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text="URLの内容を記憶しました！")]))
            else:
                line_bot_api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text="URLの読み込みに失敗しました。")]))

        elif message_text.startswith(("質問：", "質問:")):
            # 応答モード
            question = message_text.replace("質問：", "", 1).replace("質問:", "", 1).strip()
            # 【修正】answer_question に user_id も渡す
            answer = answer_question(question, user_id)
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=answer)]))

        elif message_text == "DB確認":
            # DB確認モード
            # (省略...動作は同じ)
            session = Session()
            total_count = session.query(Document).count()
            reply_text = f"現在のデータベース保存件数は {total_count} 件です。"
            session.close()
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=reply_text)]))

        else:
            # 記憶モード
            store_message(user_id, message_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
