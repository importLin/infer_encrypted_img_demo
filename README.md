A simple implementation via [timm](https://github.com/huggingface/pytorch-image-models) for privacy-preserving image classification using Vision Transformer (ViT) 
---

## How It Works


1. **`image_cipher.py`** — Key generation, image encryption & decryption via block-wise pixel shuffling
2. **`model.py`** — `CryptoViT` wraps a timm ViT and encrypts its patch/positional embeddings to match the encrypted image
3. **`infer.py`** — Loads both a standard ViT and a CryptoViT, runs inference, and compares predictions

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

Each script has a self-contained demo under `if __name__ == "__main__"`. Run them in order:

```bash
# Step 1: Visualize image encryption & decryption
python image_cipher.py

# Step 2: Compare predictions between standard ViT and CryptoViT
python infer.py
```

---

## Citation

If you find this project helpful, please consider citing the following works:

```bibtex
@article{kiya2023image,
  title={Image and model transformation with secret key for vision transformer},
  author={Kiya, Hitoshi and Iijima, Ryota and Maungmaung, Aprilpyone and Kinoshita, Yuma},
  journal={IEICE TRANSACTIONS on Information and Systems},
  volume={106},
  number={1},
  pages={2--11},
  year={2023},
  publisher={The Institute of Electronics, Information and Communication Engineers}
}
```

## License

MIT License
