import torch 

from microgpt.tokenizer.tokenizer import GPTtokenizer
from microgpt.model.model import PretrainModel
from microgpt.config import Config

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config()
    tokenizer = GPTtokenizer()
    checkpoint = torch.load('weights/base_model_checkpoint_190000.pth', map_location=device)
    model = PretrainModel(config, checkpoint['model_state_dict'], train=False).to(device)
    model.eval()

    with torch.no_grad():

        while True:
            prompt = input("query: ")
            print(model.generate(tokenizer, prompt, max_new_tokens=200, device=device))
   
if __name__ == '__main__':
    main()