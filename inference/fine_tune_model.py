import torch 

from microgpt.tokenizer import GPTtokenizer
from microgpt.model.model import FinetuneModel
from microgpt.config import Config

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config()
    tokenizer = GPTtokenizer()
    model = FinetuneModel(config).to(device)
    checkpoint = torch.load('weights/hh_rlhf_chosen_finetune.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True) 
    model.eval()

    conversation = tokenizer.user_token + "You are a helpful assisstant. Please answer my questions as best as you can." + tokenizer.end_user_token + tokenizer.assistant_token + tokenizer.end_assistant_token
    with torch.no_grad():
        while True:

            conversation += tokenizer.user_token
            prompt = input("query: ")
            conversation += prompt + tokenizer.end_user_token

            conversation += tokenizer.assistant_token
            output = model.generate(tokenizer, text=conversation, max_new_tokens=100, device=device)
            conversation += output + tokenizer.end_assistant_token

            print('assisstant: ' + output)
    
if __name__ == '__main__':
    main()