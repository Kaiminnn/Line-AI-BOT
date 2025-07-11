import os
import re
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from flask import Flask, request, abort
import json
from io import BytesIO
from PIL import Image
import threading
import fitz

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage, PushMessageRequest
)
from linebot.v3.messaging.api.messaging_api_blob import MessagingApiBlob
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent
from sqlalchemy import create_engine, text, Column, Integer, String, Text as AlchemyText, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timezone
from flask import jsonify
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# loggingの基本設定
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からキーなどを取得
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
DATABASE_URL = os.environ.get('DATABASE_URL')
GOOGLE_DRIVE_FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')

# 保存するメッセージの上限数
MAX_DOCUMENTS = 20000

# Flaskアプリの初期化
app = Flask(__name__)
CORS(app)

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
Base.metadata.create_all(engine, checkfirst=True)


# --- ヘルパー関数群 ---

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
        logging.error(f"URLのスクレイピング中にエラーが発生しました: {url} - {e}")
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
        logging.info(f"URLの内容を {len(chunks)} 個のチャンクに分割してDBに保存しました。")
        check_and_prune_db(session)
        return True
    except Exception as e:
        logging.error(f"チャンクのDB保存中にエラーが発生しました: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def embed_text(text_to_embed):
    try:
        response = genai.embed_content(model="models/text-embedding-004", content=text_to_embed, task_type="RETRIEVAL_DOCUMENT")
        return response['embedding']
    except Exception as e:
        logging.error(f"ベクトル化中にエラーが発生しました: {e}")
        return None

def store_message(user_id, message_text, source=None):
    if not message_text or message_text.isspace(): return
    session = Session()
    try:
        source_id = source if source else f"user:{user_id}"
        embedding = embed_text(message_text)
        if embedding:
            document = Document(content=message_text, embedding=embedding, source=source_id)
            session.add(document)
            session.commit()
            logging.info(f"情報をDBに保存しました。source: {source_id}")
            check_and_prune_db(session)
    except Exception as e:
        logging.error(f"データベース処理中にエラーが発生しました: {e}")
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
        logging.info(f"上限を超えたため、古いメッセージを {items_to_delete_count} 件削除しました。")

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
        logging.info(f"リランキング後のドキュメント順: {reranked_indices}")
        return reranked_docs
    except Exception as e:
        logging.error(f"リランキング中にエラーが発生しました: {e}")
        return documents[:5]

def answer_question(question, user_id, session_id):
    session = Session()
    try:
        question_embedding = embed_text(question)
        if question_embedding is None: return "質問の意味あがわからないにAIは解析にSIPPAISITAYO。"
        candidate_docs = session.query(Document).order_by(Document.embedding.l2_distance(question_embedding)).limit(20).all()
        if not candidate_docs: return "情報が十分に蓄積されていないに。"
        final_results = rerank_documents(question, candidate_docs)
        if not final_results: return "関連性の高い情報が見つからないに。"
        context = "\n".join([f"- {doc.content}" for doc in final_results])
        final_prompt = f"""以下の非常に精度の高い参考情報を中心に、ユーザーの質問に答えてください。語尾に"ポチ"」とつけて下さい。

# 参考情報
{context}

# 質問
{question}

# 回答
"""
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        final_response = model.generate_content(final_prompt)
        return final_response.text
    except Exception as e:
        logging.error(f"質問応答中にエラーが発生しました: {e}")
        return "ごめんぽち、たぶんメモリーかAPI不足でこらえられないぽち。課金したら解決するにょ"
    finally:
        session.close()

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    source = event.source
    session_id = source.group_id if source.type == 'group' else source.user_id
    user_id = source.user_id
    message_text = event.message.text
    reply_token = event.reply_token

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        if message_text.lower() == 'pdf':
            liff_url = "https://starlit-alfajores-f1b64c.netlify.app/liff.html"
            reply_text = f"PDFをアップはここポチ！\n{liff_url}"
            line_bot_api.reply_message(
                ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=reply_text)])
            )
            return

        if message_text.startswith(("質問：", "質問:")):
            question = message_text.replace("質問：", "", 1).replace("質問:", "", 1).strip()
            # 質問応答は時間がかかるのでスレッドで実行
            thread = threading.Thread(target=answer_and_push_message, args=(question, user_id, session_id, reply_token))
            thread.start()
            return

        elif message_text == "DB確認":
            session = Session()
            doc_count = session.query(Document).count()
            session.close()
            reply_text = f"ぽちのメもは {doc_count} 件あるよ"
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=reply_text)]))
            return
        
        else:
            urls = re.findall(r'https?://\S+', message_text)
            commentary = re.sub(r'https?://\S+', '', message_text).strip()
            if commentary:
                store_message(user_id, commentary)
            if urls:
                for url in urls:
                    thread = threading.Thread(target=process_url_and_notify, args=(url, session_id))
                    thread.start()

# --- 新しいバックグラウンド処理用の関数 ---

def answer_and_push_message(question, user_id, session_id, reply_token):
    """質問応答をバックグラウンドで実行し、プッシュメッセージで結果を送信する"""
    logging.info(f"質問応答処理を開始: {question}")
    # まずユーザーに応答が遅れることを伝える
    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text="ちょっと考えてるから待つぽち...")])
            )
    except Exception as e:
        logging.error(f"質問応答の初期返信でエラー: {e}")

    # 重い処理を実行
    answer = answer_question(question, user_id, session_id)
    
    # 処理完了後、プッシュメッセージで回答を送信
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.push_message(PushMessageRequest(to=session_id, messages=[TextMessage(text=answer)]))
    logging.info(f"質問応答処理が完了、回答を送信しました。")


def process_url_and_notify(url, session_id):
    """URLの内容をDBに記憶し、簡単な通知を送る"""
    logging.info(f"URL処理を開始: {url}")
    scraped_data = scrape_website(url)
    if scraped_data and scraped_data['raw_text']:
        cleaned_text = clean_text(scraped_data['raw_text'])
        is_success = chunk_and_store_text(cleaned_text, scraped_data.get('title', 'タイトル不明'), url)
        if is_success:
             with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                message_text = f"覚えたによ！\n『{scraped_data.get('title', 'タイトル不明')}』"
                line_bot_api.push_message(PushMessageRequest(to=session_id, messages=[TextMessage(text=message_text)]))
    else:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.push_message(PushMessageRequest(to=session_id, messages=[TextMessage(text=f"【失敗】URLへのアクセスに失敗したぽち。\n{url}")]))

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    # ... (内容は変更なし) ...
    source = event.source
    session_id = source.group_id if source.type == 'group' else source.user_id
    user_id = source.user_id
    message_id = event.message.id
    reply_token = event.reply_token
    # 画像処理も重いのでスレッド化
    thread = threading.Thread(target=process_image_and_notify, args=(user_id, session_id, message_id, reply_token))
    thread.start()

def process_image_and_notify(user_id, session_id, message_id, reply_token):
    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_blob_api = MessagingApiBlob(api_client)
            try:
                line_bot_api.reply_message(
                    ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text="画像を認識中です...")])
                )
            except: pass
            
            image_data_bytes = line_bot_blob_api.get_message_content(message_id=message_id)
            img = Image.open(BytesIO(image_data_bytes))
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(["この画像を日本語で詳しく、見たままに説明してください。", img])
            image_description = response.text.strip()
            
            image_source = f"image_from_user:{user_id}"
            store_message(user_id, image_description, source=image_source)

            push_text = f"この画像わかったによ！\n\n【AIによる画像の説明】\n{image_description}"
            line_bot_api.push_message(
                PushMessageRequest(to=session_id, messages=[TextMessage(text=push_text)])
            )
    except Exception as e:
        logging.error(f"画像処理中にエラーが発生しました: {e}")
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.push_message(
                PushMessageRequest(to=session_id, messages=[TextMessage(text="うまく画像が見れないぽち。")])
            )


@app.route('/upload', methods=['POST'])
def handle_upload():
    # ... (内容は変更なし) ...
    try:
        if 'pdf_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'File part is missing'}), 400
        file = request.files['pdf_file']
        context_id = request.form.get('contextId')
        if file.filename == '' or not context_id:
            return jsonify({'status': 'error', 'message': 'File or contextId is missing'}), 400
        pdf_bytes = file.read()
        filename = file.filename
        thread = threading.Thread(target=process_pdf_and_notify, args=(pdf_bytes, filename, context_id))
        thread.start()
        return jsonify({'status': 'success', 'message': 'File upload received. Processing in background.'}), 200
    except Exception as e:
        logging.error(f"アップロード受付中にエラーが発生しました: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred on the server'}), 500

def process_pdf_and_notify(pdf_bytes, filename, context_id):
    logging.info(f"PDFバックグラウンド処理開始: {filename}")
    drive_link = ""
    try:
        drive_link = upload_to_google_drive_and_get_link(pdf_bytes, filename)
        if not drive_link:
            drive_link = "ファイルの共有リンク作成に失敗しました。"
        raw_text = ""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            raw_text += page.get_text()
        doc.close()
        if not raw_text.strip():
            logging.info("テキストが空のため、Gemini OCRに切り替えます。")
            temp_pdf_path = f"temp_{filename}"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_bytes)
            uploaded_file = genai.upload_file(path=temp_pdf_path, display_name=filename)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(["このPDFファイルに書かれているテキストを全て書き出して", uploaded_file])
            raw_text = response.text
            os.remove(temp_pdf_path)

        cleaned_text = clean_text(raw_text)
        is_success = chunk_and_store_text(cleaned_text, title=filename, source_url=drive_link)
        if is_success:
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                message_text = f"PDF「{filename}」を覚えたに！\n\nファイルリンク:\n{drive_link}"
                line_bot_api.push_message(PushMessageRequest(to=context_id, messages=[TextMessage(text=message_text)]))
    except Exception as e:
        logging.error(f"PDFバックグラウンド処理中にエラー: {e}")
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            message = TextMessage(text=f"PDF「{filename}」の処理中にエラーが発生に。\n\nファイルリンク:\n{drive_link}")
            line_bot_api.push_message(PushMessageRequest(to=context_id, messages=[message]))

def upload_to_google_drive_and_get_link(pdf_bytes, filename):
    try:
        logging.info("Google Driveへのアップロードを開始します。")
        creds_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not creds_json_str:
            raise ValueError("環境変数 'GOOGLE_CREDENTIALS_JSON' が設定されていません。")
        creds_info = json.loads(creds_json_str)
        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': filename, 'parents': [GOOGLE_DRIVE_FOLDER_ID]}
        media = MediaIoBaseUpload(BytesIO(pdf_bytes), mimetype='application/pdf')
        file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
        file_id = file.get('id')
        permission = {'type': 'anyone', 'role': 'reader'}
        service.permissions().create(fileId=file_id, body=permission).execute()
        logging.info("Google Driveへのアップロードと共有設定が完了しました。")
        return file.get('webViewLink')
    except Exception as e:
        logging.error(f"Google Driveへのアップロード中にエラー: {e}")
        return None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
