import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class TextProcessor:
    _model = None

    def __init__(self):
        pass

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            logger.info("Loading SentenceTransformer model...")
            cls._model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            logger.info("Model loaded successfully")
        return cls._model

    def split_text(self, text, chunk_size=500, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    def generate_embeddings(self, chunks):
        model = self._get_model()
        batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '8'))
        logger.info(f"Generating embeddings for {len(chunks)} chunks (batch_size={batch_size})")
        embeddings = model.encode(
            chunks,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings
