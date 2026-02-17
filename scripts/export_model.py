import torch
import os
import argparse
from microgpt import Config
from dataclasses import asdict

def export_model(checkpoint_path, output_path):
    """
    Exports the model checkpoint to:
        'model_state_dict': fp16 weights
        'config': model config
    """
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' not in checkpoint:
        raise Exception(f'{checkpoint_path} does not contain model_state_dict. Only contains {checkpoint.keys()}')

    state_dict = checkpoint['model_state_dict']
    config = asdict(Config())

    # remove prefixes added by torch.compile
    temp_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('_orig_mod.', '') 
        temp_state_dict[name] = v
    state_dict = temp_state_dict

    new_checkpoint = {
        'model_state_dict': state_dict,
        'config': config
    }

    for key, tensor in state_dict.items():
        state_dict[key] = tensor.half()

    torch.save(new_checkpoint, output_path)
    
    orig_size = os.path.getsize(checkpoint_path) / (1024*1024)
    new_size = os.path.getsize(output_path) / (1024*1024)
    print(f"Reduced size from {orig_size:.2f} MB to {new_size:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to training checkpoint')
    parser.add_argument('--out', type=str, required=True, help='Path to save clean model')
    args = parser.parse_args()

    export_model(args.ckpt, args.out)