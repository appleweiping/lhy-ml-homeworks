"""Pre-download all datasets used by the homeworks.

All datasets are freely available and auto-download on first run of each HW, so
this script is optional — it just fetches everything up-front into each HW's
`data/` directory (which is gitignored). Run:

    python scripts/download_data.py

Datasets:
  - CIFAR-10       (torchvision)   -> hw03, hw08, hw09, hw10, hw11
  - FashionMNIST   (torchvision)   -> hw04, hw13
  - MNIST          (torchvision)   -> hw06, hw14
  - Omniglot       (torchvision)   -> hw15
  - Multi30k       (HF datasets)   -> hw05
  - bert-tiny      (HF transformers)-> hw07

The official gated Kaggle datasets (food-11, VoxCeleb, Crypko, DRCD, ...) are NOT
downloaded here — see each HW's README for the reason and the freely-downloadable
real dataset used in their place.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def d(hw):
    p = os.path.join(ROOT, hw, "data")
    os.makedirs(p, exist_ok=True)
    return p


def main():
    import torchvision
    print("CIFAR-10 ...")
    torchvision.datasets.CIFAR10(d("hw03-cnn"), train=True, download=True)
    torchvision.datasets.CIFAR10(d("hw03-cnn"), train=False, download=True)
    print("FashionMNIST ...")
    torchvision.datasets.FashionMNIST(d("hw04-self-attention"), train=True, download=True)
    torchvision.datasets.FashionMNIST(d("hw13-compression"), train=True, download=True)
    print("MNIST ...")
    torchvision.datasets.MNIST(d("hw14-lifelong"), train=True, download=True)
    torchvision.datasets.MNIST(d("hw14-lifelong"), train=False, download=True)
    print("Omniglot ...")
    torchvision.datasets.Omniglot(d("hw15-meta"), background=True, download=True)
    try:
        print("Multi30k (HF) ...")
        from datasets import load_dataset
        load_dataset("bentrevett/multi30k", split="train[:10]")
        print("bert-tiny (HF) ...")
        from transformers import BertTokenizerFast
        BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
    except Exception as e:
        print("HF assets skipped (offline?):", e)
    print("done.")


if __name__ == "__main__":
    main()
