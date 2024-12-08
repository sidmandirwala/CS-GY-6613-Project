# retrieval.py

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
logger = logging.getLogger(__name__)

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_API_KEY = None
COLLECTION_NAME = "qna_collection"

# Initialize embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize Qdrant client
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY
)

def retrieve_relevant_qna(query: str, top_k: int = 5):
    """
    Retrieve the top_k most relevant Q&A pairs from Qdrant based on the query.

    Args:
        query (str): The user's question.
        top_k (int): Number of top results to retrieve.

    Returns:
        List[Dict[str, str]]: List of relevant Q&A pairs.
    """
    logger.info(f"Received query: {query}")
    
    # Generate embedding for the query
    query_embedding = embedding_model.encode(query, convert_to_tensor=False).tolist()
    logger.debug(f"Generated embedding: {query_embedding[:5]}...")  # Log first 5 dimensions
    
    # Perform similarity search in Qdrant
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        logger.info(f"Qdrant search returned {len(results)} results.")
        
        qna_pairs = []
        for result in results:
            qna_pairs.append(result.payload)
            logger.debug(f"Retrieved Q&A pair: {result.payload}")
        
        if not qna_pairs:
            logger.warning("No Q&A pairs retrieved from Qdrant.")
        
        return qna_pairs
    except Exception as e:
        logger.error(f"Error retrieving from Qdrant: {e}")
        return []