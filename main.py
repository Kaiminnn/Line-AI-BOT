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

# --- 【新機能】ここからが「記憶の質」を高めるための新しい関数群です ---

# 1. Webサイトから、より賢くテキストとタイトルを抽出する関数
def scrape_website(url):
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.title.string if soup.title else 'No Title'
        
        # 本文が含まれていそうな主要なタグを優先的に探す
        main_content_tags = ["article", "main", ".main", ".content", "#main", "#content"]
        content_element = None
        for tag in main_content_tags:
            if soup.select_one(tag):
                content_element = soup.select_one(tag)
                break
        
        if not content_element:
            content_element = soup.body

        for script_or_style in content_element(["script", "style"]):
            script_or_style.decompose()
            
        text = content_element.get_text()
        return {"title": title, "raw_text": text}
    except Exception as e:
        print(f"URLのスクレイピング中にエラーが発生しました: {url} - {e}")
        return None

# 2. 抽出したテキストを掃除（クリーニング）する関数
def clean_text(raw_text):
    lines = (line.strip() for line in raw_text.splitlines())
    # 短すぎる行や意味のない行を削除するルール
    chunks = (phrase.strip() for line in lines for phrase in line.split("  ") if len(phrase.strip()) > 30)
    cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
    return cleaned_text

# 3. テキストをチャンクに分割し、メタデータ付きでDBに保存する関数
def chunk_and_store_text(cleaned_text, title, source_url):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_text(cleaned_text)
    
    if not chunks:
        print("チャンクの生成に失敗しました。")
        return

    session = Session()
    try:
        for chunk in chunks:
            # 【変更点】タイトルという文脈（メタデータ）を付けて内容を整形
            content_to_store = f"記事「{title}」より抜粋：\n{chunk}"
            embedding = embed_text(content_to_store)
            if embedding:
                document = Document(content=content_to_store, embedding=embedding, source=source_url)
                session.add(document)
        session.commit()
        print(f"URLの内容を {len(chunks)} 個のチャンクに分割してDBに保存しました。")
        check_and_prune_db(session)
    except Exception as e:
        print(f"チャンクのDB保存中にエラーが発生しました: {e}")
        session.rollback()
    finally:
        session.close()

# --- ここまでが新機能・変更点です ---


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

# 通常のメッセージをDBに保存する関数
def store_message(user_id, message_text):
    session = Session()
    try:
        source_id = f"user:{user_id}"
        embedding = embed_text(message_text)
        if embedding:
            document = Document(content=message_text, embedding=embedding, source=source_id)
            session.add(document)
            session.commit()
            check_and_prune_db(session)
    except Exception as e:
        print(f"データベース処理中にエラーが発生しました: {e}")
        session.rollback()
    finally:
        session.close()

# DBの上限チェックと削除を共通関数化
def check_and_prune_db(session):
    total_count = session.query(Document).count()
    if total_count > MAX_DOCUMENTS:
        items_to_delete_count = total_count - MAX_DOCUMENTS
        oldest_docs = session.query(Document).order_by(Document.id.asc()).limit(items_to_delete_count).all()
        for doc in oldest_docs:
            session.delete(doc)
        session.commit()
        print(f"上限を超えたため、古いメッセージを {items_to_delete_count} 件削除しました。")


# 質問に答える関数 (RAG)
def answer_question(question, user_id):
    session = Session()
    try:
        question_embedding = embed_text(question)
        if question_embedding is None:
            return "質問の解析に失敗しました。"

        # --- 【変更点】二段階検索の実装 ---
        # 第一段階：まず広く検索して、関連する情報源（source）の候補を見つける
        initial_results = session.query(Document).order_by(Document.embedding.l2_distance(question_embedding)).limit(10).all()
        
        if not initial_results:
            return "まだ情報が十分に蓄積されていないようです。"

        # 候補となる情報源をリストアップ
        source_candidates = [doc.source for doc in initial_results if doc.source and doc.source.startswith('http')]
        
        # 最も可能性の高いURLを特定（一番多く出現したURLを選ぶ）
        if source_candidates:
            likely_source = max(set(source_candidates), key=source_candidates.count)
            print(f"質問に関連する可能性の高いURL: {likely_source}")
            # 第二段階：そのURLの情報源に絞って、再度検索を行う（深掘り検索）
            final_results = session.query(Document).filter(Document.source == likely_source).order_by(Document.embedding.l2_distance(question_embedding)).limit(5).all()
        else:
            # URLが見つからなければ、通常の会話から探す
            final_results = initial_results[:5]

        context = "\n".join(f"- {doc.content}" for doc in final_results)
        prompt = f"""以下の情報を参考にして、質問に簡潔に答えてください。

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
            # 【変更点】URL処理のフローを新しい関数群を使うように変更
            url = url_match.group(0)
            print(f"URLが見つかりました: {url}")
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=f"URLを読み込んでいます...")]))
            
            scraped_data = scrape_website(url)
            if scraped_data and scraped_data['raw_text']:
                cleaned_text = clean_text(scraped_data['raw_text'])
                chunk_and_store_text(cleaned_text, scraped_data['title'], url)
                line_bot_api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text="URLの内容を記憶しました！")]))
            else:
                line_bot_api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text="URLの読み込みに失敗しました。")]))

        elif message_text.startswith(("質問：", "質問:")):
            question = message_text.replace("質問：", "", 1).replace("質問:", "", 1).strip()
            answer = answer_question(question, user_id)
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=answer)]))

        elif message_text == "DB確認":
            session = Session()
            total_count = session.query(Document).count()
            reply_text = f"現在のデータベース保存件数は {total_count} 件です。"
            session.close()
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=reply_text)]))

        else:
            store_message(user_id, message_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
