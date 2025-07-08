import os
import re
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from flask import Flask, request, abort

import json

from io import BytesIO
from PIL import Image

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
import threading  
import fitz      
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.auth.transport.requests import Request as GoogleAuthRequest

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
MAX_CHAT_HISTORY = 10 

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
            print(f"情報をDBに保存しました。source: {source_id}")
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

def check_and_prune_chat_history(session, session_id):
    """指定されたsession_idのチャット履歴が上限を超えていたら、古いものから削除する"""
    total_count = session.query(ChatHistory).filter(ChatHistory.session_id == session_id).count()
    if total_count > MAX_CHAT_HISTORY:
        items_to_delete_count = total_count - MAX_CHAT_HISTORY
        oldest_history = session.query(ChatHistory).filter(ChatHistory.session_id == session_id).order_by(ChatHistory.created_at.asc()).limit(items_to_delete_count).all()
        for history_item in oldest_history:
            session.delete(history_item)
        print(f"チャット履歴の上限を超えたため、{session_id} の古い履歴を {items_to_delete_count} 件削除しました。")


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
        # 新しい履歴を追加
        history_entry = ChatHistory(session_id=session_id, role=role, content=content)
        session.add(history_entry)
        
        # 履歴を追加した直後に、件数チェックと整理を行う
        check_and_prune_chat_history(session, session_id)
        
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

# テキストメッセージを処理する受付係
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    source = event.source
    session_id = source.group_id if source.type == 'group' else source.user_id
    user_id = source.user_id
    message_text = event.message.text
    reply_token = event.reply_token

    add_to_chat_history(session_id, 'user', message_text)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        
        # 1. まずキーワードをチェックする
        if message_text.lower() == 'pdf':
            liff_url = "https://starlit-alfajores-f1b64c.netlify.app/liff.html" #
            reply_text = f"PDFをアップは、ここから！\n{liff_url}"
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )
            return # 案内を送ったら、ここで処理を終了する
        # ★★★ ここまでが追加部分 ★★★

        # 2. キーワードに当てはまらない場合は、今までの処理を続ける
        if message_text.startswith(("質問：", "質問:")):
            question = message_text.replace("質問：", "", 1).replace("質問:", "", 1).strip()
            answer = answer_question(question, user_id, session_id)
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=answer)]))
            return

        elif message_text == "DB確認":
            session = Session()
            doc_count = session.query(Document).count()
            history_count = session.query(ChatHistory).count()
            reply_text = f"長期記憶(Documents)の件数: {doc_count} 件\n短期記憶(ChatHistory)の件数: {history_count} 件"
            session.close()
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=reply_text)]))
            return
        
        else:
            urls = re.findall(r'https?://\S+', message_text)
            commentary = re.sub(r'https?://\S+', '', message_text).strip()

            if commentary:
                store_message(user_id, commentary)
            
            if urls:
                try:
                    line_bot_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=reply_token,
                            messages=[TextMessage(text=f"メッセージと{len(urls)}件のURL、承知しました。内容を読んで記憶しますね。")]
                        )
                    )
                except Exception as e:
                    print(f"URL処理の初期返信でエラー（トークン切れや重複返信の可能性）: {e}")
                
                for url in urls:
                    thread = threading.Thread(target=process_url_and_notify, args=(url, session_id))
                    thread.start()
            

# URL処理をバックグラウンドで行うための関数
def process_url_and_notify(url, session_id):
    print(f"バックグラウンド処理開始: {url}")
    scraped_data = scrape_website(url)
    
    if scraped_data and scraped_data['raw_text']:
        cleaned_text = clean_text(scraped_data['raw_text'])
        # 1. まずDBにコンテンツを保存する
        is_success = chunk_and_store_text(cleaned_text, scraped_data['title'], url)
        
        summary = "" # 要約を格納する変数を初期化
        if is_success:
            # 2. DB保存が成功したら、Geminiで要約を試みる
            try:
                print("Gemini APIでURL内容の要約を生成しています...")
                # 長すぎるテキストは予期せぬエラーを防ぐため、ある程度の長さでカットする
                text_for_summary = cleaned_text[:15000] 
                
                summarize_prompt = f"""以下の記事の内容を、最も重要なポイントを3点に絞って、箇条書きで簡潔に要約してください。

# 記事本文
{text_for_summary}

# 要約
"""
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                response = model.generate_content(summarize_prompt)
                summary = response.text.strip()
                print("要約の生成に成功しました。")
            except Exception as e:
                print(f"要約の生成中にエラー: {e}")
                summary = "要約の生成に失敗しました。"

        # 3. LINEに通知メッセージを送信する
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            if is_success:
                # 成功した場合、メッセージにタイトルと要約を追加する
                title = scraped_data.get('title', 'タイトル不明')
                message_text = (
                    f"【完了】URLの内容を記憶しました！\n\n"
                    f"『{title}』\n\n"
                    f"【AIによる3行要約】\n{summary}\n\n"
                    f"URL:\n{url}"
                )
                line_bot_api.push_message(PushMessageRequest(to=session_id, messages=[TextMessage(text=message_text)]))
            else:
                # 失敗した場合は、従来通りのメッセージを送信
                line_bot_api.push_message(PushMessageRequest(to=session_id, messages=[TextMessage(text=f"【失敗】URLの読み込み・保存に失敗しました。\n{url}")]))

    else: # スクレイピング自体に失敗した場合
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.push_message(PushMessageRequest(to=session_id, messages=[TextMessage(text=f"【失敗】URLへのアクセスに失敗しました。\n{url}")]))
            
    print(f"バックグラウンド処理完了: {url}")

# 画像メッセージを処理する受付係
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    source = event.source
    session_id = source.group_id if source.type == 'group' else source.user_id
    user_id = source.user_id
    message_id = event.message.id
    reply_token = event.reply_token

    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_blob_api = MessagingApiBlob(api_client)
            
            # まずユーザーに、画像の処理を開始したことを素早く知らせる
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=reply_token,
                    messages=[TextMessage(text="画像を認識中です...")]
                )
            )

            # LINEのサーバーから画像データをバイトの塊として直接受け取る
            image_data_bytes = line_bot_blob_api.get_message_content(message_id=message_id)
            
            # 受け取ったバイトデータをPillowで画像として開く
            img = Image.open(BytesIO(image_data_bytes))

            # Geminiに画像を渡して、その説明を生成させる
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(["この画像を日本語で詳しく、見たままに説明してください。", img])
            image_description = response.text.strip()

            # ユーザーの発言として、チャット履歴にも保存
            history_text = f"（画像が投稿されました。画像の内容： {image_description}）"
            add_to_chat_history(session_id, 'user', history_text)
            
            # 長期記憶にも、説明文を一つの情報として保存
            image_source = f"image_from_user:{user_id}"
            store_message(user_id, history_text, source=image_source)

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

# 画像処理をバックグラウンドで行うための関数
def process_image_and_notify(user_id, session_id, image_bytes):
    print("バックグラウンドで画像処理を開始します。")
    try:
        img = Image.open(BytesIO(image_bytes))
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(["この画像を日本語で詳しく、見たままに説明してください。", img])
        image_description = response.text.strip()

        history_text = f"（画像が投稿されました。画像の内容： {image_description}）"
        add_to_chat_history(session_id, 'user', history_text)
        
        image_source = f"image_from_user:{user_id}"
        store_message(user_id, history_text, source=image_source)

        push_text = f"画像を記憶しました！\n\n【AIによる画像の説明】\n{image_description}"
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.push_message(PushMessageRequest(to=session_id, messages=[TextMessage(text=push_text)]))
        print("バックグラウンドの画像処理が成功しました。")

    except Exception as e:
        print(f"バックグラウンドの画像処理中にエラーが発生しました: {e}")
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.push_message(PushMessageRequest(to=session_id, messages=[TextMessage(text=f"画像の処理中にエラーが発生しました。")]))

# ### PDFアップロード関連の新しい関数群 ###
# ### PDFアップロード関連の修正後コード ###

def get_drive_service():
    """Google Drive APIのサービスオブジェクトを安全に取得する関数"""
    try:
        creds_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not creds_json_str:
            raise ValueError("環境変数 'GOOGLE_CREDENTIALS_JSON' が設定されていません。")

        creds_info = json.loads(creds_json_str)
        creds = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )

        # ★★★ ここが修正点 ★★★
        # 認証情報をリフレッシュして、有効なアクセストークンを確実に取得
        # 正しいリクエストオブジェクト（GoogleAuthRequest）を使用します。
        creds.refresh(GoogleAuthRequest())

        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"Google Driveサービスオブジェクトの作成中にエラー: {e}")
        return None

@app.route('/initiate-upload', methods=['POST'])
def initiate_upload():
    """LIFFからのリクエストを受け、Google Driveのアップロード専用URLを発行する"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({'status': 'error', 'message': 'Filename is required'}), 400

        print(f"アップロードセッション開始リクエストを受信: {filename}")

        # ★★★ ここからが修正部分 ★★★
        # 認証情報オブジェクトをこの場で直接作成する
        creds_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not creds_json_str:
            raise ValueError("環境変数 'GOOGLE_CREDENTIALS_JSON' が設定されていません。")
        
        creds_info = json.loads(creds_json_str)
        creds = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        creds.refresh(GoogleAuthRequest()) # トークンをリフレッシュ

        # 上記で作成した認証情報（creds）からアクセストークンを取得する
        access_token = creds.token
        # ★★★ ここまでが修正部分 ★★★

        file_metadata = {
            'name': filename,
            'parents': [GOOGLE_DRIVE_FOLDER_ID]
        }
        
        response = requests.post(
            'https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable',
            headers={
                'Authorization': f'Bearer {access_token}', # 正しいトークンを使用
                'Content-Type': 'application/json; charset=UTF-8',
            },
            data=json.dumps(file_metadata)
        )
        response.raise_for_status()
        
        upload_url = response.headers['Location']
        print("アップロード用URLの発行に成功しました。")
        
        return jsonify({'status': 'success', 'upload_url': upload_url}), 200

    except Exception as e:
        print(f"アップロードセッションの開始中にエラー: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to initiate session: {str(e)}'}), 500

def process_uploaded_pdf(drive_file_id, filename, context_id):
    """Google Drive上のPDFファイルを処理するバックグラウンドタスク"""
    print(f"バックグラウンド処理開始: {filename} (Drive File ID: {drive_file_id})")
    drive_link = f"https://drive.google.com/file/d/{drive_file_id}/view?usp=sharing"

    try:
        # 1. Google Driveからファイルをダウンロード (よりシンプルな方法に修正)
        print("Google Driveからファイルをダウンロードしています...")
        service = get_drive_service()
        if not service:
            raise Exception("Driveサービスを作成できませんでした。")
        
        media_request = service.files().get_media(fileId=drive_file_id)
        pdf_bytes = media_request.execute()

        # 2. テキスト抽出
        raw_text = ""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            raw_text += page.get_text()
        doc.close()

        # 3. テキストが空ならGemini OCR (より確実な方法に修正)
        if not raw_text.strip():
            print("テキストが空のため、Gemini OCRに切り替えます。")
            temp_pdf_path = f"temp_{filename}"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_bytes)
            
            uploaded_file = genai.upload_file(path=temp_pdf_path, display_name=filename)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(["このPDFファイルに書かれているテキストを全て書き出して", uploaded_file])
            raw_text = response.text
            os.remove(temp_pdf_path) # 一時ファイルを削除

        # 4. 要約作成 ...（以降の処理は変更なし）
        summary = ""
        try:
            summarize_prompt = f"以下の文章を、重要なポイントを3点に絞って箇条書きで要約してください。\n\n---\n\n{raw_text[:8000]}"
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(summarize_prompt)
            summary = response.text.strip()
        except Exception as e:
            summary = f"要約の生成に失敗しました。"

        # 5. DBに保存
        cleaned_text = clean_text(raw_text)
        is_success = chunk_and_store_text(cleaned_text, title=filename, source_url=drive_link)

        # 6. LINEに通知
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            message_text = f"PDF「{filename}」を記憶しました！\n\n【AIによる3行要約】\n{summary}\n\nファイルリンク:\n{drive_link}"
            line_bot_api.push_message(PushMessageRequest(to=context_id, messages=[TextMessage(text=message_text)]))

    except Exception as e:
        print(f"バックグラウンド処理中にエラー: {e}")
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            message = TextMessage(text=f"PDF「{filename}」の処理中にエラーが発生しました。\n\nファイルリンク:\n{drive_link}")
            line_bot_api.push_message(PushMessageRequest(to=context_id, messages=[message]))


def process_uploaded_pdf(drive_file_id, filename, context_id):
    """Google Drive上のPDFファイルを処理するバックグラウンドタスク"""
    print(f"バックグラウンド処理開始: {filename} (Drive File ID: {drive_file_id})")
    raw_text = ""
    drive_link = f"https://drive.google.com/file/d/{drive_file_id}/view?usp=sharing"

    try:
        # 1. Google Driveからファイルをダウンロード
        print("Google Driveからファイルをダウンロード中...")
        service = get_drive_service()
        request = service.files().get_media(fileId=drive_file_id)
        file_bytes = BytesIO()
        downloader = MediaIoBaseUpload(file_bytes, mimetype='application/pdf')
        # 正しくは MediaIoBaseDownload です。修正します。
        downloader = service.files().get_media(fileId=drive_file_id)
        file_bytes.write(downloader.execute())
        pdf_bytes = file_bytes.getvalue()

        # ...ここから先の処理は以前の process_pdf_and_notify とほぼ同じ...
        
        # 2. PyMuPDFでテキスト抽出
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            raw_text += page.get_text()
        doc.close()

        # 3. テキストが空ならGemini OCR (一時ファイルは不要になった)
        if not raw_text.strip():
            print("テキストが空のため、Gemini OCRに切り替えます...")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            # Driveからダウンロードしたバイトデータを直接は渡せないので、ファイルとして扱う
            # Gemini File APIを使う必要がありますが、ここでは簡略化のため画像として処理
            # Pillowで最初のページを画像に変換
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc.load_page(0) # 最初のページ
            pix = page.get_pixmap()
            img = Image.open(BytesIO(pix.tobytes()))
            
            response = model.generate_content(["この画像に書かれているテキストを全て書き出して", img])
            raw_text = response.text

        # 4. Gemini APIで要約作成
        summary = ""
        try:
            summarize_prompt = f"以下の文章を、重要なポイントを3点に絞って箇条書きで要約してください。\n\n---\n\n{raw_text[:8000]}"
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(summarize_prompt)
            summary = response.text.strip()
        except Exception as e:
            summary = "要約の生成に失敗しました。"

        # 5. テキストをDBに保存
        cleaned_text = clean_text(raw_text)
        is_success = chunk_and_store_text(cleaned_text, title=filename, source_url=drive_link)

        # 6. LINEに通知
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            message_text = f"PDF「{filename}」を記憶しました！\n\n【AIによる3行要約】\n{summary}\n\nファイルリンク:\n{drive_link}"
            line_bot_api.push_message(PushMessageRequest(to=context_id, messages=[TextMessage(text=message_text)]))

    except Exception as e:
        print(f"バックグラウンド処理中にエラー: {e}")
        # エラー通知
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            message_text = f"PDF「{filename}」の処理中にエラーが発生しました。\n\nファイルリンク:\n{drive_link}"
            line_bot_api.push_message(PushMessageRequest(to=context_id, messages=[TextMessage(text=message_text)]))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))