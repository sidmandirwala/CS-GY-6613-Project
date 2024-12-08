# sample_retrieval.py

from qdrant_client import QdrantClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_API_KEY = None
COLLECTION_NAME = "qna_collection"

def retrieve_sample():
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,

    )
    
    try:
        # Perform a search with a dummy vector to retrieve the top 1 point
        dummy_vector = [0.0] * 384  # Assuming vector size is 384
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=dummy_vector,
            limit=1,
            with_payload=True
        )
        if results:
            sample_qna = results[0].payload
            logger.info(f"Sample Q&A:\nQ: {sample_qna['question']}\nA: {sample_qna['answer']}")
        else:
            logger.warning("No points retrieved from Qdrant.")
    except Exception as e:
        logger.error(f"Error retrieving sample from Qdrant: {e}")

if __name__ == "__main__":
    retrieve_sample()