import torch 

from microgpt.tokenizer import GPTtokenizer
from microgpt.model import GPTModel
from microgpt.config import Config

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config()
    tokenizer = GPTtokenizer()
    model = GPTModel(config, use_lora=True).to(device)
    checkpoint = torch.load('weights/fine_tuned_checkpoint_8.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True) 
    model.eval()

    conversation = ''
    with torch.no_grad():

        while True:

            conversation += tokenizer.user_token
            prompt = input("query: ")
            conversation += prompt + tokenizer.end_user_token

            conversation += tokenizer.assistant_token
            output = model.generate(tokenizer, conversation, temperature=config.temperature, k=config.k, max_new_tokens=100, device=device)
            conversation += output + tokenizer.end_assistant_token

            print('assisstant: ' + output)
   
if __name__ == '__main__':
    main()