import os
import re # 【追加】URLを見つけるために正規表現ライブラリをインポート
import requests # 【追加】Webサイトにアクセスするためのライブラリ
from bs4 import BeautifulSoup # 【追加】HTMLを解析するためのライブラリ
import google.generativeai as genai
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from sqlalchemy import create_engine, text, Column, Integer, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

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
    content = Column(Text)
    embedding = Column(Vector(768))

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# テーブルが存在しない場合に作成
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()
Base.metadata.create_all(engine)


# 【追加】URLからテキストを抽出する専門の関数
def get_text_from_url(url):
    try:
        # タイムアウトを設定して、長時間待たないようにする
        response = requests.get(url, timeout=10)
        response.raise_for_status() # エラーがあれば例外を発生させる
        
        # BeautifulSoupでHTMLを解析
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 不要なタグ（スクリプト、スタイルなど）を削除
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # テキスト部分だけを抽出
        text = soup.get_text()
        
        # 改行や空白を整理
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"URLの読み取り中にエラーが発生しました: {url} - {e}")
        return None


# テキストをベクトル化（数値化）する関数
def embed_text(text_to_embed):
    try:
        response = genai.embed_content(model="models/text-embedding-004",
                                       content=text_to_embed,
                                       task_type="RETRIEVAL_DOCUMENT")
        return response['embedding']
    except Exception as e:
        print(f"ベクトル化中にエラーが発生しました: {e}")
        return None

# メッセージをDBに保存し、上限を超えたら古いものを削除する関数
def store_message(user_id, message_text):
    session = Session()
    try:
        content_to_store = f"ユーザー({user_id[:5]}): {message_text}" if user_id else message_text
        embedding = embed_text(content_to_store)

        if embedding is None:
            print("ベクトル化に失敗したため、DBへの保存をスキップします。")
            return

        document = Document(content=content_to_store, embedding=embedding)
        session.add(document)
        session.commit()

        total_count = session.query(Document).count()
        if total_count > MAX_DOCUMENTS:
            items_to_delete_count = total_count - MAX_DOCUMENTS
            oldest_docs = session.query(Document).order_by(Document.id.asc()).limit(items_to_delete_count).all()
            for doc in oldest_docs:
                session.delete(doc)
            session.commit()
            print(f"上限を超えたため、古いメッセージを {items_to_delete_count} 件削除しました。")

    except Exception as e:
        print(f"データベース処理中にエラーが発生しました: {e}")
        session.rollback()
    finally:
        session.close()

# 質問に答える関数 (RAG)
def answer_question(question):
    session = Session()
    try:
        question_embedding = embed_text(question)
        if question_embedding is None:
            return "質問の解析に失敗しました。"

        similar_docs = session.scalars(
            text("SELECT content FROM documents ORDER BY embedding <=> :embedding LIMIT 5")
            .bindparams(embedding=str(question_embedding))
        ).all()
        
        if not similar_docs:
            return "まだ情報が十分に蓄積されていないようです。"

        context = "\n".join(f"- {doc}" for doc in similar_docs)
        prompt = f"""以下の過去の会話ログやWebサイトの情報を参考にして、質問に答えてください。

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

        # 【修正】仕事の順番を変更し、最初にURLのチェックを入れる
        # メッセージからURLを正規表現で探す
        url_match = re.search(r'https?://\S+', message_text)

        if url_match:
            # URLが見つかった場合の処理
            url = url_match.group(0)
            print(f"URLが見つかりました: {url}")
            
            # Webサイトのテキストを抽出
            article_text = get_text_from_url(url)
            
            if article_text:
                # テキストを要約して、記憶させる
                summary_prompt = f"以下の記事を200文字程度で要約してください。\n\n{article_text[:3000]}" # 長すぎるテキストは最初の3000文字に制限
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                summary_response = model.generate_content(summary_prompt)
                summary_text = summary_response.text

                # 「（URLの内容要約：...）」という形でDBに保存
                store_message(None, f"（URLの内容要約：{url} ）\n{summary_text}")
                
                reply_text = "URLの記事を読み込んで、内容を記憶しました！"
            else:
                reply_text = "URLの読み込みに失敗しました。"

            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )

        elif message_text.startswith(("質問：", "質問:")):
            # 応答モード
            question = message_text.replace("質問：", "", 1).replace("質問:", "", 1).strip()
            answer = answer_question(question)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=answer)]
                )
            )

        elif message_text == "DB確認":
            # DB確認モード
            session = Session()
            try:
                total_count = session.query(Document).count()
                reply_text = f"現在のデータベース保存件数は {total_count} 件です。"
            except Exception as e:
                reply_text = f"件数の取得中にエラーが発生しました: {e}"
            finally:
                session.close()
            
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )

        else:
            # 記憶モード
            store_message(user_id, message_text)
            # 記憶したことをユーザーに知らせたい場合は返信する
            # line_bot_api.reply_message(...)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))