# for Parts I-III: RNN, LSTM, Transformer (causal).

import torch
import torch.nn as nn


class LMBase(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, dropout, emb_weights=None):
        super().__init__()
        if emb_weights is None:
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        else:  # Part II: pre-trained table, kept entirely fixed
            self.emb = nn.Embedding.from_pretrained(emb_weights, freeze=True, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def hidden_states(self, x):
        raise NotImplementedError

    def forward(self, x): # x: (B, T) token ids
        return self.out(self.drop(self.hidden_states(x))) # (B, T, V) logits


class RNNLM(LMBase):
    def __init__(self, vocab_size, emb_dim=100, hidden=256, layers=2, dropout=0.3, emb_weights=None, **_):
        super().__init__(vocab_size, emb_dim, hidden, dropout, emb_weights)
        self.rnn = nn.RNN(emb_dim, hidden, layers, batch_first=True,
                          dropout=dropout, nonlinearity="tanh")

    def hidden_states(self, x):
        h, _ = self.rnn(self.drop(self.emb(x)))
        return h


class LSTMLM(LMBase):
    def __init__(self, vocab_size, emb_dim=100, hidden=256, layers=2, dropout=0.3, emb_weights=None, **_):
        super().__init__(vocab_size, emb_dim, hidden, dropout, emb_weights)
        self.rnn = nn.LSTM(emb_dim, hidden, layers, batch_first=True, dropout=dropout)

    def hidden_states(self, x):
        h, _ = self.rnn(self.drop(self.emb(x)))
        return h


class TransformerLM(LMBase):
    def __init__(self, vocab_size, emb_dim=100, hidden=256, layers=2, dropout=0.3,
                 heads=4, ff=1024, max_len=64, emb_weights=None, **_):
        super().__init__(vocab_size, emb_dim, hidden, dropout, emb_weights)
        self.in_proj = nn.Linear(emb_dim, hidden)          # 100 -> d_model
        self.pos = nn.Embedding(max_len, hidden)           # learned positions
        layer = nn.TransformerEncoderLayer(hidden, heads, ff, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.max_len = max_len

    def hidden_states(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.in_proj(self.drop(self.emb(x))) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.norm(h)


MODELS = {"rnn": RNNLM, "lstm": LSTMLM, "transformer": TransformerLM}


def build_model(name, vocab_size, **kwargs):
    return MODELS[name](vocab_size, **kwargs)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = model.emb.weight.numel()
    return {"total": total, "trainable": trainable, "embedding": emb,
            "non_embedding": total - emb}
