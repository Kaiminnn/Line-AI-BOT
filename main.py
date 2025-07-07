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
from sqlalchemy import create_engine, text, Column, Integer, String, Text as AlchemyText, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timezone #【追加】タイムスタンプのためにインポート

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からキーなどを取得
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
DATABASE_URL = os.environ.get('DATABASE_URL')

# 保存するメッセージの上限数
MAX_DOCUMENTS = 20000
MAX_CHAT_HISTORY = 10 # 記憶しておく会話の往復数

# Flaskアプリの初期化
app = Flask(__name__)

# LINE Bot SDKの初期化
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Geminiの初期化
genai.configure(api_key=GEMINI_API_KEY)

# データベース設定
Base = declarative_base()

# 【変更点】Documentテーブルのsource列を必須に変更
class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    content = Column(AlchemyText)
    embedding = Column(Vector(768))
    source = Column(String(2048), nullable=False) # Not Null

# 【新機能】チャット履歴を保存するための新しいテーブルの設計図
class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), nullable=False, index=True) # ユーザーID or グループID
    role = Column(String(50), nullable=False) # 'user' or 'model'
    content = Column(AlchemyText, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

# テーブルが存在しない場合に作成
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()
Base.metadata.create_all(engine)


# --- ここからが新しい関数・修正された関数です ---

# 【新機能】チャット履歴を取得する専門家
def get_chat_history(session_id):
    session = Session()
    try:
        # 新しい順に、指定した件数だけ履歴を取得
        history = session.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.created_at.desc()).limit(MAX_CHAT_HISTORY).all()
        # 古い順に並べ直して返す
        return history[::-1]
    finally:
        session.close()

# 【新機能】チャット履歴を追加する専門家
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


# 【修正】質問応答関数を、チャット履歴を考慮するようにアップグレード
def answer_question(question, user_id, session_id):
    # ステップ1：短期記憶（チャット履歴）を取得
    history = get_chat_history(session_id)
    
    # ステップ2：AIに会話の文脈を理解させ、検索用の質問を生成させる
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
            # エラーが起きても、元の質問で続行
            rephrased_question = question

    # ステップ3：書き換えられた質問で、長期記憶（データベース）を検索・リランキング
    session = Session()
    try:
        question_embedding = embed_text(rephrased_question)
        if question_embedding is None: return "質問の解析に失敗しました。"

        candidate_docs = session.query(Document).order_by(Document.embedding.l2_distance(question_embedding)).limit(25).all()
        if not candidate_docs: return "まだ情報が十分に蓄積されていないようです。"

        final_results = rerank_documents(rephrased_question, candidate_docs)
        if not final_results: return "関連性の高い情報が見つかりませんでした。"

        # ステップ4：最終的な回答を生成
        context = "\n".join(f"- {doc.content}" for doc in final_results)
        final_prompt = f"""以下の非常に精度の高い参考情報だけを使って、ユーザーの質問に簡潔に答えてください。

# 参考情報
{context}

# 質問
{question} # ここでは、ユーザーの元の質問を使う

# 回答
"""
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        final_response = model.generate_content(final_prompt)
        
        # ステップ5：AIの回答もチャット履歴に保存
        add_to_chat_history(session_id, 'model', final_response.text)
        
        return final_response.text

    except Exception as e:
        print(f"質問応答中にエラーが発生しました: {e}")
        return "申し訳ありません、応答の生成中にエラーが発生しました。"
    finally:
        session.close()

# リランキング関数（変更なし）
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


# メッセージを仕分ける、受付係
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    # 【修正】会話の場所を特定する (グループID or ユーザーID)
    source = event.source
    session_id = source.group_id if source.type == 'group' else source.user_id
    user_id = source.user_id
    message_text = event.message.text
    
    # 【修正】まずユーザーのメッセージをチャット履歴に保存
    add_to_chat_history(session_id, 'user', message_text)
    
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        if message_text.startswith(("質問：", "質問:")):
            question = message_text.replace("質問：", "", 1).replace("質問:", "", 1).strip()
            # 【修正】answer_question に session_id も渡す
            answer = answer_question(question, user_id, session_id)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=answer)]
                )
            )
            return

        elif message_text == "DB確認":
            session = Session()
            doc_count = session.query(Document).count()
            history_count = session.query(ChatHistory).count()
            reply_text = f"長期記憶(Documents)の件数: {doc_count} 件\n短期記憶(ChatHistory)の件数: {history_count} 件"
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
                    messages=[TextMessage(text=f"メッセージと{len(urls)}件のURL、承知しました。内容を読んで長期記憶に保存します。")]
                )
            )
            for url in urls:
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
