"""
Training script for every neural LM run in Parts I and II.

Part I :  python train_lm.py --model rnn
          python train_lm.py --model lstm
          python train_lm.py --model transformer
Part II:  python train_lm.py --model lstm --emb w2v      (needs embeddings/w2v.npy)
          python train_lm.py --model lstm --emb glove    (needs embeddings/glove.npy)
          (the random-embedding LSTM from Part I is reused as the third setting)
Smoke test: python train_lm.py --model lstm --epochs 1 --max_batches 30
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import BANNED_TOKENS, GEN_LEN, PROMPTS, SEED, TEMPERATURE, TOP_K
from data import PAD, UNK, load_split, load_vocab, tokenize
from models import build_model, count_params


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def make_windows(ids: torch.Tensor, L: int):
    n = len(ids)
    starts = list(range(0, n - 1, L))
    X = torch.full((len(starts), L), PAD, dtype=torch.long)
    Y = torch.full((len(starts), L), PAD, dtype=torch.long)
    for k, s in enumerate(starts):
        y = ids[s + 1:s + L + 1]
        m = len(y)
        X[k, :m] = ids[s:s + m]
        Y[k, :m] = y
    return X, Y


@torch.no_grad()
def evaluate(model, X, Y, bs, device):
    """Average next-token cross-entropy (nats), padding excluded."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(X), bs):
        x, y = X[i:i + bs].to(device), Y[i:i + bs].to(device)
        logits = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                 ignore_index=PAD, reduction="sum").item()
        count += (y != PAD).sum().item()
    return total / count, count


@torch.no_grad()
def generate(model, prompt_ids, L, banned, device, rng):
    """Top-k sampling with temperature; same settings as the n-gram model."""
    model.eval()
    ids = list(prompt_ids)
    for _ in range(GEN_LEN):
        x = torch.tensor([ids[-L:]], device=device)
        logits = model(x)[0, -1].float().cpu() / TEMPERATURE
        logits[banned] = -float("inf")
        top = torch.topk(logits, TOP_K)
        probs = torch.softmax(top.values, dim=0)
        choice = torch.multinomial(probs, 1, generator=rng).item()
        ids.append(top.indices[choice].item())
    return ids[len(prompt_ids):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rnn", "lstm", "transformer"])
    ap.add_argument("--emb", default="random", choices=["random", "w2v", "glove"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb_dim", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max_batches", type=int, default=None, help="smoke test only")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    run = f"{args.model}_{args.emb}"
    print(f"Run {run} on {device}")

    itos, stoi = load_vocab()
    V = len(itos)
    Xtr, Ytr = make_windows(load_split("train"), args.seq_len)
    Xva, Yva = make_windows(load_split("valid"), args.seq_len)
    Xte, Yte = make_windows(load_split("test"), args.seq_len)

    emb_weights = None
    if args.emb != "random":
        emb_weights = torch.tensor(np.load(f"embeddings/{args.emb}.npy"), dtype=torch.float32)
        assert emb_weights.shape == (V, args.emb_dim), emb_weights.shape

    model = build_model(args.model, V, emb_dim=args.emb_dim, hidden=args.hidden,
                        layers=args.layers, dropout=args.dropout, heads=args.heads,
                        ff=args.ff, max_len=args.seq_len, emb_weights=emb_weights).to(device)
    params = count_params(model)
    print(params)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = f"checkpoints/{run}.pt"
    gen_cpu = torch.Generator().manual_seed(SEED)
    history, best_val, best_epoch, train_time = [], float("inf"), -1, 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen_cpu)
        tot, cnt = 0.0, 0
        sync(device)
        t0 = time.perf_counter()
        for b, i in enumerate(range(0, len(perm), args.batch_size)):
            if args.max_batches and b >= args.max_batches:
                break
            idx = perm[i:i + args.batch_size]
            x, y = Xtr[idx].to(device), Ytr[idx].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            n_tok = (y != PAD).sum().item()
            tot += loss.item() * n_tok
            cnt += n_tok
        sync(device)
        epoch_time = time.perf_counter() - t0
        train_time += epoch_time

        val_ce, _ = evaluate(model, Xva, Yva, 128, device)
        history.append({"epoch": epoch, "train_loss": tot / cnt, "val_loss": val_ce,
                        "epoch_time_s": epoch_time})
        print(f"epoch {epoch:2d} | train {tot / cnt:.3f} | val {val_ce:.3f} "
              f"(ppl {math.exp(val_ce):.1f}) | {epoch_time:.0f}s")
        if val_ce < best_val:
            best_val, best_epoch = val_ce, epoch
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))

    sync(device)
    t0 = time.perf_counter()
    test_ce, n_scored = evaluate(model, Xte, Yte, 128, device)
    sync(device)
    infer_time = time.perf_counter() - t0

    banned = [stoi[t] for t in BANNED_TOKENS]
    rng = torch.Generator().manual_seed(SEED)
    samples = {}
    for prompt in PROMPTS:
        p_ids = [stoi.get(t, UNK) for t in tokenize(prompt)]
        out = generate(model, p_ids, args.seq_len, banned, device, rng)
        samples[prompt] = " ".join(itos[i] for i in out)
        print(f"[{prompt}] -> {samples[prompt]}")

    results = {
        "run": run, "args": vars(args), "device": str(device), "params": params,
        "history": history, "best_epoch": best_epoch,
        "test_cross_entropy": test_ce, "test_ppl": math.exp(test_ce),
        "test_tokens_scored": n_scored,
        "train_time_s": train_time,
        "inference_time_s": infer_time,
        "inference_tokens_per_s": n_scored / infer_time,
        "samples": samples,
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/{run}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{run}: test ppl {results['test_ppl']:.2f} | best epoch {best_epoch} | "
          f"train {train_time / 60:.1f} min | {results['inference_tokens_per_s']:.0f} tok/s")


if __name__ == "__main__":
    main()
