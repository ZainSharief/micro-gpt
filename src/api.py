import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.pre_train.model import GPTModel
from src.tokenizer import GPTtokenizer
from src.config import config

device = 'cpu'
torch.manual_seed(411)

if torch.mps.is_available():
    device = 'mps'
    torch.mps.manual_seed(411)

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed(411)

app = Flask(__name__)
CORS(app)

@app.route('/generate-response/<context>', methods=['GET'])
def generate_response(context):
    response = base_model.generate(tokeniser, context, temperature=config.temperature, k=config.k, max_new_tokens=100, device=device)
    return jsonify(response), 200

tokeniser = GPTtokenizer()
base_model = GPTModel(tokeniser.vocab_size, config.embedding_dim, config.context_size, config.num_heads, config.num_layers, device=device, dropout=config.dropout)
base_model.load_state_dict(torch.load(config.model_path, map_location='cpu', weights_only=True)['model_state_dict'])
base_model.eval()

if __name__ == '__main__':
    
    app.run(debug=True)