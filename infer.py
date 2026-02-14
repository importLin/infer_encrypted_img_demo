from model import create_crypto_ViT
from image_cipher import load_key_dict
import timm
import torch
import torchvision.transforms as T
from PIL import Image

def load_img_tensor(img_path, size):
    """Load image as tensor (1, C, H, W)."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    if img.size[0] != size or img.size[1] != size:
        ValueError("Image size must be equal to the specified one")
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def compare_predictions(model_a, model_b, img_a, img_b):
    """Compare classification results between two models on two images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a.eval()
    model_b.eval()
    model_a.to(device)
    model_b.to(device)

    img_a = img_a.to(device)
    img_b = img_b.to(device)

    with torch.no_grad():
        pred_a = model_a(img_a).argmax(dim=1).item()
        pred_b = model_b(img_b).argmax(dim=1).item()

    return {
        "pred_ori_model": pred_a,
        "pred_enc_model": pred_b,
        "match": pred_a == pred_b,
    }

if __name__ == "__main__":
    key_path = "key_dicts/key_dict_42.npz"
    key_dict = load_key_dict(key_path)      # load key used for image encryption.
    ori_img = load_img_tensor("./sample/imgnet_ori_224x224.png", size=224)      # load original image
    enc_img = load_img_tensor("./sample/imgnet_dec.png", size=224)      # load encrypted image

    # for simplicity, initialize ViT model using timm
    ori_model = timm.create_model("vit_base_patch16_224", pretrained=True)

    # create vit with encrypted embedding params i.e., CryptoViT
    enc_model = timm.create_model("vit_base_patch16_224", pretrained=True)
    enc_model = create_crypto_ViT(enc_model, key_dict, patch_emb_encrypted=True, pos_emb_encrypted=True)

    # compare the predictions from unencrypted model and encrypted model
    res = compare_predictions(ori_model, enc_model, ori_img, enc_img)
    print("Unencrypted model prediction: ", res["pred_ori_model"])
    print("Encrypted model prediction: ", res["pred_enc_model"])

