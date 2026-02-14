import numpy as np
from pathlib import Path
from PIL import Image
import os
import matplotlib.pyplot as plt

def gen_key_dict(img_height=224, img_width=224, img_channel=3, p_size=16, seed=42):
    rng = np.random.default_rng(seed)

    num_blocks = (img_height // p_size) * (img_width // p_size)
    block_pixels = p_size * p_size * img_channel

    return {
        "key_1": rng.permutation(block_pixels),  # key for pixel shuffling
        "key_2": rng.permutation(num_blocks),  # key for block scrambling
        "config": {
            "img_height": img_height,
            "img_width": img_width,
            "img_channel": img_channel,
            "p_size": p_size,
            "seed": seed,
        }
    }


def save_key_dict(key_dict, path):
    """Save key dict to .npz file."""
    np.savez(
        Path(path),
        key_1=key_dict["key_1"],
        key_2=key_dict["key_2"],
        config=key_dict["config"],
    )


def load_key_dict(path):
    """Load key dict from .npz file."""
    data = np.load(Path(path), allow_pickle=True)
    return {
        "key_1": data["key_1"],
        "key_2": data["key_2"],
        "config": data["config"].item(),
    }


class PixelShuffler:
    def __init__(self, key_dict):
        self.key_dict = key_dict
        self.config = key_dict["config"]

    def pad_and_resize(self, img):
        """Pad shorter edge to match aspect ratio, then resize to config dimensions."""
        c, h, w = img.shape
        target_h, target_w = self.config["img_height"], self.config["img_width"]

        if h == target_h and w == target_w:
            return img

        target_ratio = target_w / target_h
        if w / h < target_ratio:
            new_w = int(h * target_ratio)
            pad = new_w - w
            img = np.pad(img, ((0, 0), (0, 0), (pad // 2, pad - pad // 2)))
        elif w / h > target_ratio:
            new_h = int(w / target_ratio)
            pad = new_h - h
            img = np.pad(img, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)))

        img_pil = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))
        img_resized = np.array(img_pil.resize((target_w, target_h), Image.BILINEAR))
        return img_resized.transpose(2, 0, 1)

    def _to_blocks(self, img):
        """(C,H,W) -> (num_blocks, C*p*p)"""
        c, h, w = img.shape
        p = self.config["p_size"]
        return img.reshape(c, h // p, p, w // p, p).transpose(1, 3, 0, 2, 4).reshape(-1, c * p * p)

    def _from_blocks(self, blocks):
        """(num_blocks, C*p*p) -> (C,H,W)"""
        cfg = self.config
        p, c = cfg["p_size"], cfg["img_channel"]
        nh, nw = cfg["img_height"] // p, cfg["img_width"] // p
        return blocks.reshape(nh, nw, c, p, p).transpose(2, 0, 3, 1, 4).reshape(c, cfg["img_height"], cfg["img_width"])

    def _ensure_chw(self, img):
        """Ensure (C,H,W) format."""
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[2] in [1, 3, 4]:
            img = img.transpose(2, 0, 1)
        return img

    def encrypt(self, img):
        """Encrypt image. Input: (C,H,W) or (H,W,C). Output: (C,H,W)."""
        img = self._ensure_chw(img)
        blocks = self._to_blocks(img)
        blocks = blocks[:, self.key_dict["key_1"]]  # shuffle pixels in each block
        blocks = blocks[self.key_dict["key_2"]]  # shuffle block positions
        return self._from_blocks(blocks)

    def decrypt(self, img):
        """Decrypt image. Input: (C,H,W) or (H,W,C). Output: (C,H,W)."""
        img = self._ensure_chw(img)
        blocks = self._to_blocks(img)
        blocks = blocks[np.argsort(self.key_dict["key_2"])]  # inverse block shuffle
        blocks = blocks[:, np.argsort(self.key_dict["key_1"])]  # inverse pixel shuffle
        return self._from_blocks(blocks)

    def save(self, img, path):
        """Save (C,H,W) image to file."""
        img = img.transpose(1, 2, 0).astype(np.uint8)
        if img.shape[2] == 1:
            img = img.squeeze(2)
        Image.fromarray(img).save(Path(path))


def visualize_encryption(img_path, key_dict, figsize=(6, 6), dpi=300, fontsize=16):
    """Visualize original, encrypted, and decrypted images sequentially."""
    shuffler = PixelShuffler(key_dict)

    img_original = np.array(Image.open(img_path).convert("RGB")).transpose(2, 0, 1)
    img_processed = shuffler.pad_and_resize(img_original)
    encrypted = shuffler.encrypt(img_processed)
    decrypted = shuffler.decrypt(encrypted)

    titles = ["Original", "Padding + Resize", "Encrypted", "Decrypted"]
    images = [img_original, img_processed, encrypted, decrypted]

    for title, im in zip(titles, images):
        plt.figure(figsize=figsize)
        plt.imshow(im.transpose(1, 2, 0).astype(np.uint8))
        plt.title(title, fontsize=fontsize + 2)
        plt.xlabel("Width", fontsize=fontsize)
        plt.ylabel("Height", fontsize=fontsize)
        plt.xticks(fontsize=fontsize - 2)
        plt.yticks(fontsize=fontsize - 2)
        plt.tight_layout()
        # plt.savefig(f"{title.lower()}.png", dpi=dpi)
        plt.show()


if __name__ == "__main__":
    seed = 42
    key_dict_path = "./key_dicts/key_dict_42.npz"

    ori_input_path = "./sample/imgnet_ori_224x224.png"
    enc_output_path = "./sample/imgnet_enc.png"
    dec_output_path = "./sample/imgnet_dec.png"

    if os.path.isfile(key_dict_path):
        key_dict = load_key_dict(key_dict_path)
    else:
        key_dict = gen_key_dict(img_height=224, img_width=224, p_size=16, seed=seed)

    visualize_encryption(img_path=ori_input_path, key_dict=key_dict, figsize=(6, 6), dpi=300)

    # # if you want to save the res produced by your images, uncomment the line below
    # shuffler = PixelShuffler(key_dict)
    # img_original = np.array(Image.open(ori_input_path).convert("RGB")).transpose(2, 0, 1)
    # img_processed = shuffler.pad_and_resize(img_original)      # image size must match ViT's input size (224, 224)
    #
    # encrypted_img = shuffler.encrypt(img_processed)
    # shuffler.save(encrypted_img, enc_output_path)
    #
    # decrypted_img = shuffler.decrypt(encrypted_img)
    # shuffler.save(decrypted_img, dec_output_path)
