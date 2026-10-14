"""
Part II: build the two FROZEN input-embedding tables
python embeddings.py --which w2v     # skip-gram trained on  training split only
python embeddings.py --which glove   # public GloVe 6B 100d (Wikipedia + Gigaword)

Each writes embeddings/<name>.npy with shape (|V|, 100), row i = vector for vocab id i,
and results/emb_<name>.json (coverage, timing, nearest-neighbour sanity check).

run: PYTHONHASHSEED=42 python embeddings.py --which w2v).
"""
import argparse
import json
import os
import time

import numpy as np

from config import SEED
from data import EOS, PAD, load_split, load_vocab

DIM = 100
PROBE_WORDS = ["city", "film", "war", "bird", "election", "music", "king", "river"]


def fill_missing(E, found_mask, rng):
    sigma = float(E[found_mask].std())
    missing = ~found_mask
    E[missing] = rng.normal(0.0, sigma, size=(missing.sum(), E.shape[1]))
    E[PAD] = 0.0
    return sigma


def neighbours(E, itos, stoi, k=5):
    norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    out = {}
    for w in PROBE_WORDS:
        if w in stoi:
            sims = norm @ norm[stoi[w]]
            sims[stoi[w]] = -1
            out[w] = [itos[i] for i in np.argsort(-sims)[:k]]
    return out


def build_w2v(itos, stoi, rng):
    from gensim.models import Word2Vec

    ids = load_split("train").tolist()
    sentences, cur = [], []
    for i in ids:                       # split the stream into paragraphs at <eos>
        if i == EOS:
            if cur:
                sentences.append(cur)
            cur = []
        else:
            cur.append(itos[i])
    if cur:
        sentences.append(cur)

    t0 = time.perf_counter()
    model = Word2Vec(sentences, vector_size=DIM, window=5, sg=1, negative=5,
                     min_count=1, epochs=5, workers=1, seed=SEED)
    train_time = time.perf_counter() - t0

    E = np.zeros((len(itos), DIM), dtype=np.float32)
    found = np.zeros(len(itos), dtype=bool)
    for w, i in stoi.items():
        if w in model.wv:
            E[i] = model.wv[w]
            found[i] = True
    meta = {"method": "skip-gram (gensim Word2Vec)", "window": 5, "negative": 5,
            "epochs": 5, "train_tokens": len(ids), "train_time_s": train_time}
    return E, found, meta


def build_glove(itos, stoi, rng):
    import gensim.downloader as api

    t0 = time.perf_counter()
    kv = api.load("glove-wiki-gigaword-100")      # ~130MB, cached in ~/gensim-data
    load_time = time.perf_counter() - t0

    E = np.zeros((len(itos), DIM), dtype=np.float32)
    found = np.zeros(len(itos), dtype=bool)
    for w, i in stoi.items():
        if w in kv.key_to_index:
            E[i] = kv[w]
            found[i] = True
    meta = {"source": "GloVe 6B 100d (glove-wiki-gigaword-100)",
            "pretraining_tokens": "~6 billion (Wikipedia 2014 + Gigaword 5)",
            "load_time_s": load_time}
    return E, found, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["w2v", "glove"])
    args = ap.parse_args()

    itos, stoi = load_vocab()
    rng = np.random.default_rng(SEED)
    E, found, meta = (build_w2v if args.which == "w2v" else build_glove)(itos, stoi, rng)

    real_words = [i for i, w in enumerate(itos) if not w.startswith("<")]
    covered = int(found[real_words].sum())
    train_ids = load_split("train").numpy()
    is_word = np.array([not w.startswith("<") for w in itos])
    word_tokens = train_ids[is_word[train_ids]]       # ignore <unk>/<eos> occurrences
    token_cov = float(found[word_tokens].mean())      # share of training TOKENS covered
    missing_examples = [itos[i] for i in real_words if not found[i]][:25]
    sigma = fill_missing(E, found, rng)

    os.makedirs("embeddings", exist_ok=True)
    np.save(f"embeddings/{args.which}.npy", E)

    meta.update({
        "vocab_size": len(itos),
        "word_types_covered": covered,
        "word_type_coverage": covered / len(real_words),
        "train_token_coverage": token_cov,
        "missing_examples": missing_examples,
        "missing_init": f"N(0, {sigma:.4f}^2); <pad> = 0",
        "nearest_neighbours": neighbours(E, itos, stoi),
    })
    os.makedirs("results", exist_ok=True)
    with open(f"results/emb_{args.which}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()