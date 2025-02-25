import torch 

from ..src.tokenizer import GPTtokenizer
from ..src.model import GPTModel
from ..src.config import config

device = 'cpu'
torch.manual_seed(411)

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed(411)

tokeniser = GPTtokenizer()
model = GPTModel(tokeniser.vocab_size, config.embedding_dim, config.context_size, config.num_heads, config.num_layers, device=device, dropout=config.dropout)
m = model.to(device)

checkpoint = torch.load(config.model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def main():

    conversation = ''
    while True:

        user_text = input('YOU: ')

        if user_text != '':

            conversation += user_text 

            model_text = model.generate(
                tokeniser=tokeniser, 
                text=conversation, 
                temperature=config.temperature, 
                k=config.k,
                max_new_tokens=100,
            )

            conversation += model_text

            print('MicroGPT: ' + model_text)

if __name__ == '__main__':
    main()