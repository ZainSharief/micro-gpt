import torch 
from microgpt.config import Config
from microgpt.tokenizer import GPTtokenizer
from microgpt.model import FinetuneModel

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config()
    tokenizer = GPTtokenizer()
    checkpoint = torch.load('weights/hh_rlhf_chosen_finetune.pth', weights_only=True)
    model = FinetuneModel(config, checkpoint['model_state_dict'], train=False).to(device)
    model.eval()

    conversation = ''
    with torch.no_grad():
        while True:

            conversation += tokenizer.user_token
            prompt = input("user: ")
            conversation += prompt + tokenizer.end_user_token

            conversation += tokenizer.assistant_token
            output = model.generate(tokenizer, text=conversation, max_new_tokens=100, device=device)
            conversation += output + tokenizer.end_assistant_token

            print('assistant: ' + output)
    
if __name__ == '__main__':
    main()