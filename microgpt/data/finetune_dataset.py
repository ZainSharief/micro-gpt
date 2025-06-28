import torch
from datasets import load_dataset
from torch.utils.data import Dataset

class OpenAssistantDataset(Dataset):
    def __init__(self, tokeniser, context_size, device, split='train', pad_token_id=0):
        self.context_size = context_size
        self.device = device

        data = load_dataset(
            'OpenAssistant/oasst1', 
            split=split, 
            trust_remote_code=True
        )

        self.dataset = self.build_conversations(data, context_size, tokeniser, pad_token_id)
    
    def build_conversations(self, dataset, context_size, tokeniser, pad_token_id):

        conversation_dataset = []

        # Extracts all the "parents" and stores children messages with their parent IDs
        parents = []
        parent_ids = {}
        for data in dataset:
            if data['parent_id'] is None:  
                parents.append(data)
            else:
                parent_ids[data['parent_id']] = data

        for parent in parents:

            # Adds the tokenised parent message to the conversation
            conversation = tokeniser.encode(parent['text']).squeeze(0)
            conversation = torch.cat((conversation, torch.tensor([tokeniser.eos_token], dtype=torch.long)))
            loss_value = 0 if parent['role'] == 'prompter' else 1
            loss_mask = torch.full((conversation.size(0) + 1,), fill_value=loss_value, dtype=torch.bool) 
            
            while parent['message_id'] in parent_ids:
                
                # Adds the tokenised child message to the conversation
                # and updates the parent to the next message in the chain
                parent = parent_ids[parent['message_id']]
                tokenised_text = tokeniser.encode(parent['text']).squeeze(0)
                conversation = torch.cat((conversation, tokenised_text))
                conversation = torch.cat((conversation, torch.tensor([tokeniser.eos_token], dtype=torch.long)))
                loss_value = 0 if parent['role'] == 'prompter' else 1
                loss_mask = torch.cat((loss_mask, torch.full((tokenised_text.size(0) + 1,), fill_value=loss_value, dtype=torch.bool)))

            # Splits the conversation into samples of size context_size + 1
            conversation_samples, loss_mask_mask_samples = self.batch_data(conversation, loss_mask, context_size, pad_token_id)    

            conversation_dataset += [{'tokens' : conv, 'loss_mask' : mask} for conv, mask in zip(conversation_samples, loss_mask_mask_samples)]
        
        return conversation_dataset

    def batch_data(self, conversation, loss_mask, context_size, pad_token_id):

        conversation_samples = list(conversation.split(context_size + 1, dim=0))
        loss_mask_samples = list(loss_mask.split(context_size + 1, dim=0))

        size = conversation_samples[-1].size(0)
        if size < context_size + 1:
            conversation_samples[-1] = torch.cat((conversation_samples[-1], torch.full((context_size + 1 - size,), pad_token_id, dtype=torch.long)))
            loss_mask_samples[-1] = torch.cat((loss_mask_samples[-1], torch.zeros(context_size + 1 - size, dtype=torch.bool)))

        return conversation_samples, loss_mask_samples

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        sample = self.dataset[idx]
        tokens = sample['tokens']
        pad_mask = sample['loss_mask']

        x = tokens[:self.context_size]
        y = tokens[1:]
        pad_mask = pad_mask[:self.context_size]

        return x.to(self.device), y.to(self.device), pad_mask.to(self.device)