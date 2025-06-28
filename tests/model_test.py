import torch 

from microgpt.tokenizer import GPTtokenizer
from microgpt.model.model import GPTModel
from microgpt.config import Config

def main():

    device = 'cpu'
    torch.manual_seed(411)

    if torch.mps.is_available():
        device = 'mps'
        torch.mps.manual_seed(411)

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(411)

    config = Config()
    tokeniser = GPTtokenizer()
    model = GPTModel(config)
    m = model.to(device)

    final_weights = torch.load(config.base_model_path, map_location=device)
    model.load_state_dict(final_weights['model_state_dict'], strict=True)
    model.eval()

    with torch.no_grad():
        prompt = input("Query: ")
        if prompt != '':
            print(model.generate(tokeniser, prompt, temperature=config.temperature, k=config.k, max_new_tokens=100, device=device))
   
if __name__ == '__main__':
    main()