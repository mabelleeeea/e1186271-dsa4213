## Part I

KN trigram: test PPL 210.0 (CE 5.347), D=0.9, 1,890,099 stored counts, fit 2.0s, 927,136 tok/s on CPU

| Model | Test PPL | Test CE | Best epoch | Params (total / trainable) | Train time (min) | Inference (tok/s) |
|---|---|---|---|---|---|---|
| RNN | 185.4 | 5.223 | 10/10 | 7.36M / 7.36M | 19.1 | 47,979 |
| LSTM | 200.1 | 5.299 | 10/10 | 8.03M / 8.03M | 21.3 | 33,769 |
| Transformer | 157.3 | 5.058 | 10/10 | 8.76M / 8.76M | 25.3 | 31,866 |

## Part II

| Model | Test PPL | Test CE | Best epoch | Params (total / trainable) | Train time (min) | Inference (tok/s) |
|---|---|---|---|---|---|---|
| Random (trainable) | 200.1 | 5.299 | 10/10 | 8.03M / 8.03M | 21.3 | 33,769 |
| Skip-gram (frozen) | 171.8 | 5.147 | 10/10 | 8.03M / 6.03M | 20.9 | 36,003 |
| GloVe (frozen) | 186.6 | 5.229 | 10/10 | 8.03M / 6.03M | 20.6 | 35,023 |

w2v: type coverage 100.0%, token coverage 100.0%, missing init N(0, 0.1946^2); <pad> = 0, train time 25.7s

glove: type coverage 97.7%, token coverage 99.7%, missing init N(0, 0.5227^2); <pad> = 0, load time 31.3s

## Part III

| Representation | Accuracy | Macro-F1 | Best C | LM test PPL |
|---|---|---|---|---|
| lstm_random/mean | 0.5880 | 0.5869 | 1.0 | 200.1 |
| lstm_random/last | 0.4312 | 0.4283 | 0.01 | 200.1 |
| transformer_random/mean | 0.7209 | 0.7202 | 1.0 | 157.3 |
| transformer_random/last | 0.6346 | 0.6322 | 0.01 | 157.3 |
| lstm_w2v/mean | 0.7537 | 0.7525 | 0.1 | 171.8 |
| lstm_w2v/last | 0.7067 | 0.7065 | 0.1 | 171.8 |
| lstm_glove/mean | 0.8039 | 0.8034 | 0.1 | 186.6 |
| lstm_glove/last | 0.7397 | 0.7371 | 0.01 | 186.6 |
| bag-of-embeddings(lstm_random)/mean | 0.5609 | 0.5601 | 0.01 | — |
| bag-of-embeddings(lstm_w2v)/mean | 0.7784 | 0.7778 | 0.01 | — |
| bag-of-embeddings(lstm_glove)/mean | 0.8646 | 0.8643 | 0.1 | — |
