import torch 
from microgpt.config import Config
from microgpt.tokenizer import GPTtokenizer
from microgpt.model import PretrainModel

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config()
    tokenizer = GPTtokenizer()
    checkpoint = torch.load('model/pretrain_model.pth', map_location=device)

    model = PretrainModel(config, checkpoint['model_state_dict']).to(device)
    model.eval()

    with torch.no_grad():

        while True:
            prompt = input("query: ")
            print(model.generate(tokenizer, prompt, max_new_tokens=100, device=device))
   
if __name__ == '__main__':
    main()