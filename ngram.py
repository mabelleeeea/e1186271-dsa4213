import json
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np

from config import BANNED_TOKENS, GEN_LEN, PROMPTS, SEED, TEMPERATURE, TOP_K
from data import load_split, load_vocab, tokenize, UNK


class KneserNeyTrigram:
    def __init__(self, vocab_size: int, D: float = 0.75):
        self.V = vocab_size
        self.D = D

    def fit(self, ids: list[int]):
        tri = defaultdict(Counter)                     # (u,v) -> {w: c(u,v,w)}
        for u, v, w in zip(ids, ids[1:], ids[2:]):
            tri[(u, v)][w] += 1
        self.tri = tri
        self.tri_tot = {k: sum(c.values()) for k, c in tri.items()}

        cont2 = defaultdict(Counter)                   # v -> {w: N(.,v,w)}
        for (u, v), c in tri.items():
            for w in c:
                cont2[v][w] += 1
        self.cont2 = cont2
        self.cont2_tot = {v: sum(c.values()) for v, c in cont2.items()}

        bigram_types = set(zip(ids, ids[1:]))
        cont1 = Counter(w for _, w in bigram_types)    # w -> N(.,w)
        self.cont1 = cont1
        self.cont1_tot = sum(cont1.values())
        self.cont1_types = len(cont1)

        # Vectorised P1 (used for generation)
        p1 = np.full(self.V, self.D * self.cont1_types / self.cont1_tot / self.V)
        for w, n in cont1.items():
            p1[w] += max(n - self.D, 0) / self.cont1_tot
        self.p1_vec = p1
        return self

    def num_entries(self) -> int:
        """'Model size' for an n-gram = number of stored count entries."""
        return (sum(len(c) for c in self.tri.values())
                + sum(len(c) for c in self.cont2.values()) + len(self.cont1))

    # ---------- scalar probabilities (evaluation) ----------
    def p1(self, w):
        D = self.D
        return (max(self.cont1.get(w, 0) - D, 0)
                + D * self.cont1_types / self.V) / self.cont1_tot

    def p2(self, w, v):
        tot = self.cont2_tot.get(v)
        if not tot:
            return self.p1(w)
        c = self.cont2[v]
        return max(c.get(w, 0) - self.D, 0) / tot + self.D * len(c) / tot * self.p1(w)

    def p3(self, w, u, v):
        tot = self.tri_tot.get((u, v))
        if not tot:
            return self.p2(w, v)
        c = self.tri[(u, v)]
        return max(c.get(w, 0) - self.D, 0) / tot + self.D * len(c) / tot * self.p2(w, v)

    def cross_entropy(self, ids: list[int]) -> float:
        """Average NLL (nats) over tokens 1..N-1 — the common eval protocol."""
        nll = -math.log(self.p2(ids[1], ids[0]))
        for i in range(2, len(ids)):
            nll -= math.log(self.p3(ids[i], ids[i - 2], ids[i - 1]))
        return nll / (len(ids) - 1)

    # ---------- full distribution (generation) ----------
    def dist(self, u, v) -> np.ndarray:
        D = self.D
        p = self.p1_vec.copy()
        if self.cont2_tot.get(v):
            c, tot = self.cont2[v], self.cont2_tot[v]
            p *= D * len(c) / tot
            for w, n in c.items():
                p[w] += max(n - D, 0) / tot
        if self.tri_tot.get((u, v)):
            c, tot = self.tri[(u, v)], self.tri_tot[(u, v)]
            p *= D * len(c) / tot
            for w, n in c.items():
                p[w] += max(n - D, 0) / tot
        return p

    def generate(self, prompt_ids, n_tokens, top_k, temperature, banned, rng):
        out = list(prompt_ids)
        for _ in range(n_tokens):
            p = self.dist(out[-2], out[-1])
            p[banned] = 0.0
            top = np.argpartition(p, -top_k)[-top_k:]
            logits = np.log(p[top] + 1e-12) / temperature
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            out.append(int(rng.choice(top, p=probs)))
        return out


def main():
    itos, stoi = load_vocab()
    train = load_split("train").tolist()
    valid = load_split("valid").tolist()
    test = load_split("test").tolist()

    t0 = time.perf_counter()
    model = KneserNeyTrigram(len(itos)).fit(train)
    train_time = time.perf_counter() - t0

    # Tune D on validation only (counts don't depend on D, so fit once)
    tuning = {}
    for D in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        model.D = D                      # scalar eval reads D directly; counts reused
        tuning[D] = math.exp(model.cross_entropy(valid))
        print(f"D={D:.2f}  valid ppl={tuning[D]:.2f}")
    best_D = min(tuning, key=tuning.get)
    model.D = best_D
    model.fit(train)  # rebuild p1_vec with the chosen D

    t0 = time.perf_counter()
    test_ce = model.cross_entropy(test)
    infer_time = time.perf_counter() - t0
    n_scored = len(test) - 1

    banned = [stoi[t] for t in BANNED_TOKENS]
    rng = np.random.default_rng(SEED)
    samples = {}
    for prompt in PROMPTS:
        p_ids = [stoi.get(t, UNK) for t in tokenize(prompt)]
        gen = model.generate(p_ids, GEN_LEN, TOP_K, TEMPERATURE, banned, rng)
        samples[prompt] = " ".join(itos[i] for i in gen[len(p_ids):])
        print(f"\n[{prompt}] -> {samples[prompt]}")

    results = {
        "model": "trigram interpolated Kneser-Ney",
        "best_D": best_D,
        "valid_ppl_by_D": tuning,
        "test_cross_entropy": test_ce,
        "test_ppl": math.exp(test_ce),
        "train_time_s": train_time,
        "stored_entries": model.num_entries(),
        "inference_time_s": infer_time,
        "inference_tokens_per_s": n_scored / infer_time,
        "inference_device": "CPU (pure Python)",
        "samples": samples,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/ngram.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTest ppl {results['test_ppl']:.2f} | train {train_time:.1f}s | "
          f"{results['inference_tokens_per_s']:.0f} tok/s | {results['stored_entries']:,} entries")


if __name__ == "__main__":
    main()
