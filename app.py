import logging
import threading
import uuid
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

from document_processor import DocumentProcessor
from text_processor import TextProcessor
from vector_store import ChromaVectorStore
from mcp_protocol import ModelContextProtocol
from rag_pipeline import RAGPipeline

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

doc_proc = DocumentProcessor()
text_proc = TextProcessor()
rag = RAGPipeline()
mcp = ModelContextProtocol()
vector_db = None

upload_status = {}

def process_document_async(file_path, filename, request_id):
    try:
        app.logger.info(f"[{request_id}] Starting processing of {filename}")
        text = doc_proc.process_uploaded_file(file_path)
        app.logger.info(f"[{request_id}] Text extracted, {len(text)} chars")
        chunks = text_proc.split_text(text)
        app.logger.info(f"[{request_id}] Chunks: {len(chunks)}")
        embeddings = text_proc.generate_embeddings(chunks)
        app.logger.info(f"[{request_id}] Embeddings generated")

        global vector_db
        if vector_db is None:
            app.logger.info(f"[{request_id}] Initializing ChromaDB...")
            vector_db = ChromaVectorStore()

        doc_id = str(uuid.uuid4())
        vector_db.store_documents(chunks, embeddings, {'doc_id': doc_id})
        app.logger.info(f"[{request_id}] Upsert complete, doc_id={doc_id}")
        upload_status[request_id] = {'status': 'done', 'doc_id': doc_id}
    except Exception as e:
        app.logger.error(f"[{request_id}] Processing failed: {str(e)}")
        upload_status[request_id] = {'status': 'error', 'error': str(e)}
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        filename = file.filename
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        request_id = str(uuid.uuid4())
        thread = threading.Thread(
            target=process_document_async,
            args=(file_path, filename, request_id)
        )
        thread.start()
        app.logger.info(f"[{request_id}] Upload accepted, processing started")
        return jsonify({'request_id': request_id, 'message': 'Processing started'}), 202
    except Exception as e:
        logging.error(f'Upload initiation error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/status/<request_id>')
def get_status(request_id):
    status = upload_status.get(request_id, {'status': 'pending'})
    return jsonify(status)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data['question']
        style = data.get('style', 'default')
        q_embed = text_proc.generate_embeddings([question])[0]
        docs = vector_db.search_similar(q_embed)
        context = '\n'.join([d['metadata']['text'] for d in docs])
        prompt = mcp.get_context_prompt(style, question, context)
        answer = rag.generate_answer(prompt)
        return jsonify({'answer': answer})
    except Exception as e:
        logging.error(f'Ask error: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
