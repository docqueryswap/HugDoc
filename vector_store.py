import os
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    _client = None
    _collection = None
    _collection_name = "ragnoapi_docs"
    _embedding_dimension = 384

    def __init__(self, persist_directory="./chroma_db"):
        if ChromaVectorStore._client is None:
            logger.info(f"Initializing ChromaDB client with persistence at: {persist_directory}")
            ChromaVectorStore._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self._initialize_collection()

    def _initialize_collection(self):
        embedding_function = embedding_functions.DefaultEmbeddingFunction()
        try:
            existing_collections = ChromaVectorStore._client.list_collections()
            collection_exists = any(c.name == self._collection_name for c in existing_collections)
            if collection_exists:
                ChromaVectorStore._collection = ChromaVectorStore._client.get_collection(
                    name=self._collection_name,
                    embedding_function=embedding_function
                )
                count = ChromaVectorStore._collection.count()
                if count > 0:
                    sample = ChromaVectorStore._collection.peek(limit=1)
                    if sample['embeddings'] and sample['embeddings'][0]:
                        existing_dim = len(sample['embeddings'][0])
                        if existing_dim != self._embedding_dimension:
                            logger.warning("Collection dimension mismatch. Recreating...")
                            ChromaVectorStore._client.delete_collection(name=self._collection_name)
                            collection_exists = False
                if collection_exists:
                    logger.info(f"Using existing collection: {self._collection_name}")
                    return
            if not collection_exists:
                logger.info(f"Creating new collection: {self._collection_name}")
                ChromaVectorStore._collection = ChromaVectorStore._client.create_collection(
                    name=self._collection_name,
                    embedding_function=embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            ChromaVectorStore._collection = ChromaVectorStore._client.create_collection(
                name=self._collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )

    @property
    def collection(self):
        return ChromaVectorStore._collection

    def store_documents(self, chunks, vectors, metadata):
        if not chunks:
            return
        doc_id = metadata.get('doc_id', 'unknown')
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        metadatas = [{'text': chunk, **metadata} for chunk in chunks]
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=vectors.tolist() if hasattr(vectors, 'tolist') else vectors,
                metadatas=metadatas,
                documents=chunks
            )
            logger.info(f"Stored {len(chunks)} chunks with doc_id: {doc_id}")
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            raise

    def search_similar(self, query_vector, top_k=3):
        try:
            results = self.collection.query(
                query_embeddings=[query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
            matches = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    matches.append({
                        'id': doc_id,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': 1 - results['distances'][0][i] if results['distances'] else 0.0
                    })
            return matches
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
