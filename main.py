import os
import google.generativeai as genai
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from sqlalchemy import create_engine, text, Column, Integer, Index, Text
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
    embedding = Column(Vector(768)) # text-embedding-004の次元数

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# テーブルが存在しない場合に作成
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()
Base.metadata.create_all(engine)


# テキストをベクトル化（数値化）する関数
def embed_text(text_to_embed):
    response = genai.embed_content(model="models/text-embedding-004",
                                   content=text_to_embed,
                                   task_type="RETRIEVAL_DOCUMENT")
    return response['embedding']

# メッセージをDBに保存する関数
def store_message(user_id, message_text):
    session = Session()
    content_to_store = f"ユーザー({user_id[:5]}): {message_text}"
    embedding = embed_text(content_to_store)
    document = Document(content=content_to_store, embedding=embedding)
    session.add(document)
    session.commit()
    session.close()

# 質問に答える関数 (RAG)
def answer_question(question):
    session = Session()
    question_embedding = embed_text(question)
    
    # データベースから類似度の高い過去の会話を5つ検索
    similar_docs = session.scalars(
        text("SELECT content FROM documents ORDER BY embedding <=> :embedding LIMIT 5")
        .bindparams(embedding=str(question_embedding))
    ).all()
    
    session.close()
    
    if not similar_docs:
        return "まだ情報が十分に蓄積されていないようです。"

    # AIへのプロンプトを作成
    context = "\n".join(f"- {doc}" for doc in similar_docs)
    prompt = f"""以下の過去の会話ログを参考にして、質問に答えてください。

# 参考情報（過去の会話ログ）
{context}

# 質問
{question}

# 回答
"""
    # Geminiに質問を投げる
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)
    return response.text


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

        # 全角・半角両方のコロンに対応
        if message_text.startswith(("質問：", "質問:")):
            # 「質問：」または「質問:」で始まる場合は、質問応答モード
            question = message_text.replace("質問：", "", 1).replace("質問:", "", 1).strip()
            answer = answer_question(question)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=answer)]
                )
            )
        else:
            # それ以外のメッセージは記憶モード
            store_message(user_id, message_text)
            # 記憶したことをユーザーに知らせたい場合は、以下のコメントを外す
            # line_bot_api.reply_message(
            #     ReplyMessageRequest(
            #         reply_token=event.reply_token,
            #         messages=[TextMessage(text="記憶しました！")]
            #     )
            # )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))