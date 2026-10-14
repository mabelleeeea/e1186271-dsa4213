import json
import os
import re
from collections import Counter

import torch
from datasets import load_dataset

DATA_DIR = "data"
VOCAB_SIZE = 20000                       # including special tokens
SPECIALS = ["<pad>", "<unk>", "<eos>"]
PAD, UNK, EOS = 0, 1, 2
# WikiText writes possessives/contractions as "city 's", "didn 't": keep "'s", "'t"
# etc. as single tokens (as GloVe does) instead of splitting off the apostrophe.
TOKEN_RE = re.compile(r"'(?:s|t|re|ve|ll|d|m)\b|[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def clean(line: str) -> str:
    line = line.strip().lower()
    return line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(clean(text))


def read_split(texts) -> list[list[str]]:
    paras = []
    for text in texts:
        s = text.strip()
        if not s or s.startswith("="):
            continue
        toks = tokenize(s)
        if toks:
            paras.append(toks + ["<eos>"])
    return paras


def build_vocab(train_paras):
    counts = Counter(t for p in train_paras for t in p if t != "<eos>")
    words = [w for w, _ in counts.most_common(VOCAB_SIZE - len(SPECIALS))]
    itos = SPECIALS + words
    return itos, {w: i for i, w in enumerate(itos)}


def encode(paras, stoi) -> list[int]:
    return [stoi.get(t, UNK) for p in paras for t in p]


def load_vocab():
    with open(os.path.join(DATA_DIR, "vocab.json")) as f:
        itos = json.load(f)
    return itos, {w: i for i, w in enumerate(itos)}


def load_split(name: str) -> torch.Tensor:
    """name in {'train', 'valid', 'test'} -> 1-D LongTensor of token ids."""
    return torch.load(os.path.join(DATA_DIR, f"{name}.pt"))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    paras = {
        "train": read_split(ds["train"]["text"]),
        "valid": read_split(ds["validation"]["text"]),
        "test": read_split(ds["test"]["text"]),
    }
    itos, stoi = build_vocab(paras["train"])
    with open(os.path.join(DATA_DIR, "vocab.json"), "w") as f:
        json.dump(itos, f)

    stats = {"vocab_size": len(itos)}
    for split, p in paras.items():
        ids = encode(p, stoi)
        torch.save(torch.tensor(ids, dtype=torch.long), os.path.join(DATA_DIR, f"{split}.pt"))
        n = len(ids)
        stats[split] = {
            "paragraphs": len(p),
            "tokens": n,
            "unk_rate": round(sum(i == UNK for i in ids) / n, 4),
        }
    with open(os.path.join(DATA_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()