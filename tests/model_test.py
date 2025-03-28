import torch 

from src.tokenizer import GPTtokenizer
from src.pre_train.model import GPTModel
from src.config import config

device = 'cpu'
torch.manual_seed(411)

if torch.mps.is_available():
    device = 'mps'
    torch.mps.manual_seed(411)

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed(411)

tokeniser = GPTtokenizer()
model = GPTModel(tokeniser.vocab_size, config.embedding_dim, config.context_size, config.num_heads, config.num_layers, device=device, dropout=config.dropout)
m = model.to(device)

final_weights = torch.load(config.base_model_path, map_location=device)
model.load_state_dict(final_weights)
model.eval()

def main():
    with torch.no_grad():
        print(model.generate(tokeniser, 'The best way to greet someone is to say', temperature=config.temperature, k=config.k, max_new_tokens=100, device=device))
    
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
                device=device
            )

            conversation += model_text

            print('MicroGPT: ' + model_text)
            print(tokeniser.encode(model_text))

if __name__ == '__main__':
    main()