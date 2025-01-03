from re import sub
import torch

def get_batch(dataset, tokeniser, batch_size, context_size, device, iter=None, train=True):

    data = 'train' if train else 'validation'

    if iter is None:
        num_rows = 11118 if data == 'train' else 1000
        conversation = dataset[data][torch.randint(num_rows, (1,))]['dialog'][0]

    else:
        conversation = dataset[data][iter]['dialog']

    text = ''
    for line in conversation:

        line = line.strip()
        line = sub(r'\s([,.?!])', r'\1', line)
        line = sub(r"(\b\w+)\s'\s(\w+)", r"\1'\2", line)
        line = sub(r" ’ ", r"'", line)

        text += line + '\n'

    data = tokeniser.encode(text)[0, :]
    idx = torch.randint(len(data), (batch_size,))
    data = torch.cat([data, torch.zeros(context_size, dtype=torch.long)])

    x = torch.stack([data[i:i+context_size] for i in idx])
    y = torch.stack([data[i+1:i+context_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

def get_text(dataset, train=True):
    data = 'train' if train else 'validation'
    text = ''
    for conversation in dataset[data]:
        conversation = conversation['dialog']
        for line in conversation:

            line = line.strip()
            line = sub(r'\s([,.?!])', r'\1', line)
            line = sub(r"(\b\w+)\s'\s(\w+)", r"\1'\2", line)
            line = sub(r" ’ ", r"'", line)

            text += line + '\n'

    return text