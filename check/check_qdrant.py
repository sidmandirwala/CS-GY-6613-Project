# check_qdrant.py

from qdrant_client import QdrantClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_API_KEY = None  # Set if applicable
COLLECTION_NAME = "qna_collection"

def check_qdrant_points():
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
    )
    
    try:
        # Get collection information
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        
        # Access points_count directly
        point_count = collection_info.points_count
        logger.info(f"Collection '{COLLECTION_NAME}' has {point_count} points.")
    except Exception as e:
        logger.error(f"Error accessing Qdrant: {e}")

if __name__ == "__main__":
    check_qdrant_points()