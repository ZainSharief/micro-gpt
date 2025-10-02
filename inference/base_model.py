import torch 

from microgpt.tokenizer.tokenizer import GPTtokenizer
from microgpt.model.model import PretrainModel
from microgpt.config import Config

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config()
    tokenizer = GPTtokenizer()
    model = PretrainModel(config).to(device)
    checkpoint = torch.load('weights/base_model_checkpoint_190000.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    with torch.no_grad():

        while True:
            prompt = input("query: ")
            print(model.generate(tokenizer, prompt, temperature=config.temperature, k=config.k, max_new_tokens=100, device=device))
   
if __name__ == '__main__':
    main()