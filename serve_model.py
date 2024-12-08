# serve_model.py

from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the fine-tuned model and tokenizer
model_name = "prathamssaraf/finetuned-gpt2_50_new-peft"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 150)  # Updated parameter

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,      # Use max_new_tokens
            num_beams=5,                        # Enable beam search
            no_repeat_ngram_size=2,
            early_stopping=True                 # Valid with num_beams > 1
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)