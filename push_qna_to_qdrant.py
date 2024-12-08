# push_qna_to_qdrant.py

import csv
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CSV_FILE_PATH = "qna.csv"  # Path to your CSV file
QDRANT_HOST = "localhost"   # Host where Qdrant is running
QDRANT_PORT = 6333          # Correct port for Qdrant exposed by Docker
QDRANT_API_KEY = None       # Set if you have an API key for Qdrant
COLLECTION_NAME = "qna_collection"  # Name of the Qdrant collection

# Initialize the embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize Qdrant client
logger.info("Initializing Qdrant client.")
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
)

# Function to load Q&A from CSV
def load_qna_from_csv(csv_file: str) -> List[Dict[str, str]]:
    """
    Load question-answer pairs from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        List[Dict[str, str]]: List of Q&A pairs.
    """
    logger.info(f"Loading Q&A data from {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        qna_data = df.to_dict(orient="records")
        logger.info(f"Loaded {len(qna_data)} Q&A pairs.")
        return qna_data
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        raise

# Function to generate embeddings
def generate_embeddings(questions: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of questions.

    Args:
        questions (List[str]): List of questions.

    Returns:
        List[List[float]]: List of embedding vectors.
    """
    logger.info("Generating embeddings for questions.")
    try:
        embeddings = embedding_model.encode(questions, convert_to_tensor=False, show_progress_bar=True)
        embeddings = [embedding.tolist() for embedding in embeddings]
        logger.info("Embeddings generated successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

# Function to ensure Qdrant collection exists
def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, distance: str = "Cosine"):
    """
    Ensure that the specified Qdrant collection exists. If not, create it.

    Args:
        client (QdrantClient): Instance of QdrantClient.
        collection_name (str): Name of the collection.
        vector_size (int): Dimension of the vectors.
        distance (str): Distance metric ("Cosine", "Euclidean", "Dot").
    """
    logger.info(f"Checking if collection '{collection_name}' exists in Qdrant.")
    try:
        if collection_name not in client.get_collections().collections:
            logger.info(f"Collection '{collection_name}' does not exist. Creating it.")
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": vector_size,
                    "distance": distance
                }
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logger.error(f"Failed to ensure collection in Qdrant: {e}")
        raise

# Function to save Q&A to Qdrant
def save_qna_to_qdrant(qna_data: List[Dict[str, str]], embeddings: List[List[float]], collection_name: str):
    """
    Save the Q&A data along with their embeddings to Qdrant.

    Args:
        qna_data (List[Dict[str, str]]): List of Q&A pairs.
        embeddings (List[List[float]]): List of question embeddings.
        collection_name (str): Name of the Qdrant collection.
    """
    logger.info(f"Saving Q&A data to Qdrant collection '{collection_name}'.")
    points = []
    for idx, (qna, embedding) in enumerate(zip(qna_data, embeddings)):
        point = PointStruct(
            id=str(uuid.uuid4()),  # Unique identifier for each point
            vector=embedding,
            payload={
                "question": qna["Question"],
                "answer": qna["Answer"]
            }
        )
        points.append(point)
        if (idx + 1) % 1000 == 0:
            logger.info(f"Prepared {idx + 1} points.")

    try:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Successfully upserted {len(points)} points into '{collection_name}'.")
    except Exception as e:
        logger.error(f"Failed to upsert points into Qdrant: {e}")
        raise

# Main function to orchestrate the process
def main():
    # Step 1: Load Q&A from CSV
    qna_data = load_qna_from_csv(CSV_FILE_PATH)
    
    # Step 2: Generate embeddings for questions
    questions = [qna["Question"] for qna in qna_data]
    embeddings = generate_embeddings(questions)
    
    # Step 3: Ensure Qdrant collection exists
    # Assuming embedding size of 'all-MiniLM-L6-v2' is 384
    EMBEDDING_SIZE = len(embeddings[0]) if embeddings else 384
    ensure_collection(client, COLLECTION_NAME, vector_size=EMBEDDING_SIZE, distance="Cosine")
    
    # Step 4: Save Q&A with embeddings to Qdrant
    save_qna_to_qdrant(qna_data, embeddings, COLLECTION_NAME)

if __name__ == "__main__":
    main()