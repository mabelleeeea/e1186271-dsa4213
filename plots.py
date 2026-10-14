"""
Figures and summary tables for the report.

  python plots.py

Writes:
  figures/part1_loss_curves.png   train/val loss of RNN, LSTM, Transformer (+ trigram val line)
  figures/part2_loss_curves.png   train/val loss of the three LSTM embedding settings
  results/tables.md               Part I / II / III numbers in Markdown tables

"""
import json
import math
import os

import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]     # fixed categorical order
INK, MUTED, GRID = "#2b2b2a", "#6f6e69", "#e4e3df"

PART1 = [("rnn_random", "RNN"), ("lstm_random", "LSTM"), ("transformer_random", "Transformer")]
PART2 = [("lstm_random", "Random (trainable)"), ("lstm_w2v", "Skip-gram (frozen)"),
         ("lstm_glove", "GloVe (frozen)")]


def load(name):
    p = f"results/{name}.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def curves(runs, title, out, ngram_val=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    ends = []                                   # [y_true, y_label, text, x]
    for (run, label), color in zip(runs, COLORS):
        r = load(run)
        if r is None:
            continue
        ep = [h["epoch"] for h in r["history"]]
        val = [h["val_loss"] for h in r["history"]]
        ax.plot(ep, [h["train_loss"] for h in r["history"]], color=color, lw=1.6, ls="--")
        ax.plot(ep, val, color=color, lw=2, marker="o", ms=4, label=label)
        ends.append([val[-1], val[-1], f"{label}  {val[-1]:.2f}", ep[-1]])
    # Direct end-of-line labels, nudged apart so they never overlap
    if ends:
        lo, hi = ax.get_ylim()
        gap = 0.05 * (hi - lo)
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
        for y, y_lab, text, x in ends:
            ax.annotate(text, (x, y), xytext=(x + 0.25, y_lab), textcoords="data",
                        va="center", fontsize=8, color=INK)
    if ngram_val is not None:
        last_ep = max(e[3] for e in ends) if ends else 1
        ax.plot([1, last_ep], [ngram_val, ngram_val], color=MUTED, lw=1.2, ls=":")
        ax.annotate(f"KN trigram (val)  {ngram_val:.2f}", (1, ngram_val), xytext=(0, -4),
                    textcoords="offset points", va="top", fontsize=8, color=MUTED)
    ax.plot([], [], color=MUTED, lw=1.6, ls="--", label="train (dashed)")
    ax.set_xlabel("Epoch", color=INK)
    ax.set_ylabel("Cross-entropy (nats / token)", color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    n_ep = max((len(load(r)["history"]) for r, _ in runs if load(r)), default=1)
    ax.set_xticks(range(1, n_ep + 1))
    ax.set_xlim(0.6, n_ep + 2.4)              # room on the right for direct labels
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def lm_table(runs, extra_cols=False):
    head = "| Model | Test PPL | Test CE | Best epoch | Params (total / trainable) | Train time (min) | Inference (tok/s) |"
    lines = [head, "|" + "---|" * 7]
    for run, label in runs:
        r = load(run)
        if r is None:
            continue
        p = r["params"]
        lines.append(f"| {label} | {r['test_ppl']:.1f} | {r['test_cross_entropy']:.3f} | "
                     f"{r['best_epoch']}/{len(r['history'])} | {p['total'] / 1e6:.2f}M / "
                     f"{p['trainable'] / 1e6:.2f}M | {r['train_time_s'] / 60:.1f} | "
                     f"{r['inference_tokens_per_s']:,.0f} |")
    return lines


def main():
    os.makedirs("figures", exist_ok=True)
    ng = load("ngram")
    ng_val = math.log(ng["valid_ppl_by_D"][str(ng["best_D"])]) if ng else None

    curves(PART1, "Part I: training (dashed) and validation (solid) loss",
           "figures/part1_loss_curves.png", ng_val)
    curves(PART2, "Part II: LSTM with three input-embedding settings",
           "figures/part2_loss_curves.png")

    out = ["## Part I", ""]
    if ng:
        out.append(f"KN trigram: test PPL {ng['test_ppl']:.1f} (CE {ng['test_cross_entropy']:.3f}), "
                   f"D={ng['best_D']}, {ng['stored_entries']:,} stored counts, "
                   f"fit {ng['train_time_s']:.1f}s, {ng['inference_tokens_per_s']:,.0f} tok/s on CPU")
        out.append("")
    out += lm_table(PART1) + ["", "## Part II", ""] + lm_table(PART2)
    for name in ("w2v", "glove"):
        e = load(f"emb_{name}")
        if e:
            t = e.get("train_time_s", e.get("load_time_s"))
            out.append(f"\n{name}: type coverage {e['word_type_coverage']:.1%}, token coverage "
                       f"{e['train_token_coverage']:.1%}, missing init {e['missing_init']}, "
                       f"{'train' if 'train_time_s' in e else 'load'} time {t:.1f}s")

    d = load("downstream")
    if d:
        out += ["", "## Part III", "",
                "| Representation | Accuracy | Macro-F1 | Best C | LM test PPL |", "|---|---|---|---|---|"]
        for name, r in d["results"].items():
            ppl = "—" if r["lm_test_ppl"] is None else f"{r['lm_test_ppl']:.1f}"
            out.append(f"| {name} | {r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} | "
                       f"{r['best_C']} | {ppl} |")
    with open("results/tables.md", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()