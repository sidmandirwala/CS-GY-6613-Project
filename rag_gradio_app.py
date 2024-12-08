import gradio as gr
import requests

# Define the FastAPI endpoint
API_URL = "http://localhost:8000/ask"

# Define pre-populated questions
QUESTIONS = [
    "What is ROS2?",
]

# Function to call the FastAPI endpoint
def ask_rag(question):
    payload = {"question": question}
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"Error: {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
with gr.Blocks() as rag_app:
    gr.Markdown("## RAG System Interface")
    
    with gr.Row():
        question_input = gr.Dropdown(
            choices=QUESTIONS, label="Select a Question"
        )
        
    submit_button = gr.Button("Ask")
    answer_output = gr.Textbox(
        label="Answer", interactive=False
    )

    submit_button.click(
        ask_rag, inputs=question_input, outputs=answer_output
    )

# Launch the app
rag_app.launch(server_name="0.0.0.0", server_port=7860)