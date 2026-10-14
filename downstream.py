"""
Part III

Task: AG News (4 classes: World, Sports, Business, Sci/Tech).
  - train/val: 10,000 articles sampled from AG News train (seeded), split 8,000 / 2,000
  - test: the official AG News test set (7,600 articles)

For each (language model, pooling) pair:
  1. Tokenise with the SAME tokenizer/vocab as Parts I-II (OOV -> <unk>), truncate
     to 64 tokens (the LMs' context length), right-pad with <pad>.
  2. Run the FROZEN LM, take per-token hidden states (before the output layer).
  3. Pool to one fixed-length vector:
       mean = average of hidden states over real tokens (padding excluded)
       last = hidden state at the last real token
     Right-padding + causal models => real-token states never see the padding.
  4. Standardise features (fit on train only) and fit a logistic regression.
     C chosen from a small grid by validation macro-F1; then evaluated once on test.
Same classifier family, grid and data splits for every representation.

Baseline (--bag_baselines): average the raw input-embedding vectors of the
article (no LM at all) and use the same classifier, to test whether the LM's
hidden states add anything beyond the embeddings themselves.

Usage:
  python downstream.py  # LSTM vs Transformer, mean pooling
  python downstream.py --runs lstm_random transformer_random lstm_w2v lstm_glove 
                       --pooling mean last 
                       --bag_baselines lstm_random lstm_w2v lstm_glove # full comparison
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from config import SEED
from data import PAD, UNK, load_vocab, tokenize
from models import build_model
from train_lm import get_device

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
MAX_LEN = 64
C_GRID = [0.01, 0.1, 1.0, 10.0]


def load_ag(n_trainval=10000, n_val=2000):
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(ds["train"]))[:n_trainval]
    tr = ds["train"].select(idx[n_val:].tolist())
    va = ds["train"].select(idx[:n_val].tolist())
    te = ds["test"]
    return ({"text": tr["text"], "label": tr["label"]},
            {"text": va["text"], "label": va["label"]},
            {"text": te["text"], "label": te["label"]})


def encode(texts, stoi):
    X = torch.full((len(texts), MAX_LEN), PAD, dtype=torch.long)
    lengths, n_tok, n_unk, n_trunc = [], 0, 0, 0
    for k, t in enumerate(texts):
        ids = [stoi.get(w, UNK) for w in tokenize(t)] or [UNK]
        n_tok += len(ids)
        n_unk += sum(i == UNK for i in ids)
        if len(ids) > MAX_LEN:
            n_trunc += 1
            ids = ids[:MAX_LEN]
        X[k, :len(ids)] = torch.tensor(ids)
        lengths.append(len(ids))
    stats = {"unk_rate": n_unk / n_tok, "truncated_share": n_trunc / len(texts),
             "mean_tokens": n_tok / len(texts)}
    return X, torch.tensor(lengths), stats


def load_lm(run, V, device):
    with open(f"results/{run}.json") as f:
        a = json.load(f)["args"]
    emb = None if a["emb"] == "random" else torch.zeros(V, a["emb_dim"])
    model = build_model(a["model"], V, emb_dim=a["emb_dim"], hidden=a["hidden"],
                        layers=a["layers"], dropout=a["dropout"], heads=a["heads"],
                        ff=a["ff"], max_len=a["seq_len"], emb_weights=emb)
    model.load_state_dict(torch.load(f"checkpoints/{run}.pt", map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def features(model, X, lengths, device, bs=256, bag=False):
    """bag=True: skip the LM and pool the raw input-embedding vectors instead."""
    out = {"mean": [], "last": []}
    for i in range(0, len(X), bs):
        x, L = X[i:i + bs].to(device), lengths[i:i + bs].to(device)
        h = model.emb(x) if bag else model.hidden_states(x)        # (B, T, H)
        mask = (x != PAD).unsqueeze(-1).float()
        out["mean"].append(((h * mask).sum(1) / mask.sum(1)).cpu())
        out["last"].append(h[torch.arange(len(x), device=device), L - 1].cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def fit_eval(Ftr, ytr, Fva, yva, Fte, yte):
    scaler = StandardScaler().fit(Ftr)
    Ftr, Fva, Fte = scaler.transform(Ftr), scaler.transform(Fva), scaler.transform(Fte)
    val_scores = {}
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED).fit(Ftr, ytr)
        val_scores[C] = f1_score(yva, clf.predict(Fva), average="macro")
    best_C = max(val_scores, key=val_scores.get)
    clf = LogisticRegression(C=best_C, max_iter=2000, random_state=SEED).fit(Ftr, ytr)
    pred = clf.predict(Fte)
    return pred, {"best_C": best_C, "val_macro_f1_by_C": val_scores,
                  "test_accuracy": accuracy_score(yte, pred),
                  "test_macro_f1": f1_score(yte, pred, average="macro"),
                  "confusion_matrix": confusion_matrix(yte, pred).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["lstm_random", "transformer_random"])
    ap.add_argument("--pooling", nargs="+", default=["mean"], choices=["mean", "last"])
    ap.add_argument("--bag_baselines", nargs="*", default=[],
                    help="runs whose input embeddings to average directly, with no LM "
                         "(e.g. lstm_random lstm_w2v lstm_glove)")
    args = ap.parse_args()

    device = get_device()
    itos, stoi = load_vocab()
    tr, va, te = load_ag()
    Xtr, Ltr, s_tr = encode(tr["text"], stoi)
    Xva, Lva, _ = encode(va["text"], stoi)
    Xte, Lte, s_te = encode(te["text"], stoi)
    ytr, yva, yte = map(np.array, (tr["label"], va["label"], te["label"]))
    print("train encoding:", s_tr, "\ntest encoding:", s_te)

    results, preds = {}, {}
    for run in args.runs:
        with open(f"results/{run}.json") as f:
            lm_ppl = json.load(f)["test_ppl"]
        model = load_lm(run, len(itos), device)
        t0 = time.perf_counter()
        F = [features(model, X, L, device) for X, L in ((Xtr, Ltr), (Xva, Lva), (Xte, Lte))]
        feat_time = time.perf_counter() - t0
        for pool in args.pooling:
            name = f"{run}/{pool}"
            pred, r = fit_eval(F[0][pool], ytr, F[1][pool], yva, F[2][pool], yte)
            r.update({"lm_test_ppl": lm_ppl, "feature_dim": F[0][pool].shape[1],
                      "feature_extraction_time_s": feat_time})
            results[name], preds[name] = r, pred
            print(f"{name:28s} acc {r['test_accuracy']:.4f} | macro-F1 "
                  f"{r['test_macro_f1']:.4f} | C={r['best_C']} | LM ppl {lm_ppl:.1f}")

    # Baseline: mean of the raw input-embedding vectors (no LM). Same classifier.
    # Asks whether the LM's hidden states add anything over the embeddings alone.
    for run in args.bag_baselines:
        model = load_lm(run, len(itos), device)
        F = [features(model, X, L, device, bag=True) for X, L in ((Xtr, Ltr), (Xva, Lva), (Xte, Lte))]
        name = f"bag-of-embeddings({run})/mean"
        pred, r = fit_eval(F[0]["mean"], ytr, F[1]["mean"], yva, F[2]["mean"], yte)
        r.update({"lm_test_ppl": None, "feature_dim": F[0]["mean"].shape[1]})
        results[name], preds[name] = r, pred
        print(f"{name:28s} acc {r['test_accuracy']:.4f} | macro-F1 "
              f"{r['test_macro_f1']:.4f} | C={r['best_C']} | (no LM)")

    # Error analysis: misclassified examples per representation, plus disagreements
    rng = np.random.default_rng(SEED)
    errors = {}
    for name, pred in preds.items():
        wrong = np.flatnonzero(pred != yte)
        pick = rng.choice(wrong, size=min(10, len(wrong)), replace=False)
        errors[name] = [{"text": te["text"][i], "true": LABELS[yte[i]],
                         "pred": LABELS[pred[i]]} for i in pick]
    if len(args.runs) >= 2:      # compare the first two LMs under the first pooling
        a, b = (f"{r}/{args.pooling[0]}" for r in args.runs[:2])
        only_a = np.flatnonzero((preds[a] == yte) & (preds[b] != yte))
        only_b = np.flatnonzero((preds[b] == yte) & (preds[a] != yte))
        errors["disagreement_counts"] = {f"only {a} correct": int(len(only_a)),
                                         f"only {b} correct": int(len(only_b))}
        errors[f"only {b} correct (examples)"] = [
            {"text": te["text"][i], "true": LABELS[yte[i]], f"{a} said": LABELS[preds[a][i]]}
            for i in rng.choice(only_b, size=min(8, len(only_b)), replace=False)]
        errors[f"only {a} correct (examples)"] = [
            {"text": te["text"][i], "true": LABELS[yte[i]], f"{b} said": LABELS[preds[b][i]]}
            for i in rng.choice(only_a, size=min(8, len(only_a)), replace=False)]

    os.makedirs("results", exist_ok=True)
    with open("results/downstream.json", "w") as f:
        json.dump({"labels": LABELS, "splits": {"train": len(ytr), "val": len(yva),
                   "test": len(yte)}, "encoding": {"train": s_tr, "test": s_te},
                   "results": results}, f, indent=2)
    with open("results/downstream_errors.json", "w") as f:
        json.dump(errors, f, indent=2)
    print("\nSaved results/downstream.json and results/downstream_errors.json")


if __name__ == "__main__":
    main()