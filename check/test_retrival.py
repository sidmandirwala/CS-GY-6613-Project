# test_retrieval.py

from retrieval import retrieve_relevant_qna
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_retrieval():
    test_question = "what is ros2?"
    retrieved_qna = retrieve_relevant_qna(test_question, top_k=5)
    
    if retrieved_qna:
        logger.info(f"Retrieved {len(retrieved_qna)} Q&A pairs:")
        for idx, qna in enumerate(retrieved_qna, 1):
            logger.info(f"{idx}. Q: {qna['question']}\n   A: {qna['answer']}\n")
    else:
        logger.warning("No Q&A pairs retrieved.")

if __name__ == "__main__":
    test_retrieval()