from torch import nn
from model.LoRA import LoRA_ViT_timm
import timm
import torch

def find_module(model, target_module_name):
    for name, module in model.named_modules():
        if name.endswith(target_module_name):
            return module
    raise ValueError(f"Module '{target_module_name}' not found")

def find_param(model, target_param_name):
    for name, param in model.named_parameters():
        if name.endswith(target_param_name):
            return param
    raise ValueError(f"Param '{target_param_name}' not found")


class CryptoViT(nn.Module):
    def __init__(self, model, key_dict):
        super(CryptoViT, self).__init__()
        self.model = model
        self.key_dict = key_dict


    def encrypt_patch_emb(self):
        with torch.no_grad():
            # get E i.e., the params of patch embedding layer or linear projection
            patch_embed_proj = find_module(self.model, target_module_name="patch_embed.proj")
            E = patch_embed_proj.weight     # E shape [768, 3, 16, 16]
            original_shape = E.shape

            # flatten -> shuffle(encrypt) -> reshape
            flat_E = torch.flatten(E, start_dim=1)
            shuffled_E = flat_E[:, self.key_dict["key_1"]]
            reshaped_E = shuffled_E.view(original_shape)

            # replace the original E with its encrypted
            patch_embed_proj.weight.copy_(reshaped_E)

    def encrypt_pos_emb(self):
        with torch.no_grad():
            pos_embed = find_param(self.model, target_param_name="pos_embed")

            cls_token_pos = pos_embed[:, :1, :]  # pos embedding for class token [1, 1, 768]
            p_token_pos = pos_embed[:, 1:, :]  # pos embeddings for patch token [1, 196, 768]

            shuffled_patch_pos = p_token_pos[:, self.key_dict["key_2"], :]  # shuffle
            new_pos_embed = torch.cat([cls_token_pos, shuffled_patch_pos], dim=1)
            pos_embed.copy_(new_pos_embed)


    def forward(self, x):
        return self.model(x)

def create_crypto_ViT(model, key_dict, patch_emb_encrypted=False, pos_emb_encrypted=False):
    model = CryptoViT(model, key_dict)
    if patch_emb_encrypted:
        print("Patch Embedding Encrypted...")
        model.encrypt_patch_emb()
    if pos_emb_encrypted:
        print("Positional Embedding Encrypted...")
        model.encrypt_pos_emb()
    return model


if __name__ == "__main__":
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model = create_crypto_ViT(model)