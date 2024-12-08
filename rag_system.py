# rag_system.py

import requests
from retrieval import retrieve_relevant_qna
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
logger = logging.getLogger(__name__)

# Configuration
GPT2_API_URL = "http://localhost:5001/generate"  # Flask API endpoint

# Function to formulate the prompt
def formulate_prompt(retrieved_qna, user_query):
    """
    Create a prompt that includes retrieved Q&A pairs and the user query.

    Args:
        retrieved_qna (List[Dict[str, str]]): Retrieved Q&A pairs.
        user_query (str): The user's question.

    Returns:
        str: The formulated prompt.
    """
    prompt = "Here are some relevant Q&A pairs:\n\n"
    for qna in retrieved_qna:
        prompt += f"Q: {qna['question']}\nA: {qna['answer']}\n\n"
    prompt += f"Now, answer the following question:\n\n{user_query}"
    logger.debug(f"Formulated prompt:\n{prompt}")
    return prompt

# Function to generate answer using GPT-2 model via Flask API
def generate_answer(prompt, max_tokens=150):
    """
    Generate an answer to the prompt using the GPT-2 model.

    Args:
        prompt (str): The formulated prompt.
        max_tokens (int): Maximum number of tokens in the generated answer.

    Returns:
        str: The generated answer.
    """
    logger.info("Sending prompt to GPT-2 model.")
    logger.debug(f"Prompt: {prompt}")
    
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_tokens
    }
    try:
        response = requests.post(GPT2_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        answer = result.get('text', '').strip()
        logger.info("Received answer from GPT-2 model.")
        logger.debug(f"Answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "I'm sorry, I couldn't generate an answer at this time."

# Function to handle the RAG process
def rag(user_query):
    """
    Handle the Retrieval-Augmented Generation process.

    Args:
        user_query (str): The user's question.

    Returns:
        str: The generated answer.
    """
    logger.info(f"Processing RAG for query: {user_query}")
    
    # Step 1: Retrieve relevant Q&A pairs
    retrieved_qna = retrieve_relevant_qna(user_query)
    if not retrieved_qna:
        logger.warning("No relevant Q&A pairs found.")
        return "I'm sorry, I couldn't find any relevant information to answer your question."

    # Step 2: Formulate the prompt
    prompt = formulate_prompt(retrieved_qna, user_query)

    # Step 3: Generate the answer using GPT-2 model
    answer = generate_answer(prompt)

    logger.info("RAG process completed.")
    return answer

# Example usage
if __name__ == "__main__":
    user_query = "what is ros2?"
    answer = rag(user_query)
    print(f"Q: {user_query}\nA: {answer}\n")