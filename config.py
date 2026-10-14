# Settings shared by every model, so comparisons stay matched.

SEED = 42

PROMPTS = [
    "the team scored",
    "after the election , the",
    "the film was directed",
    "the bird was found in",
    "the experiment caused",
]
GEN_LEN = 30          # new tokens per prompt
TOP_K = 40
TEMPERATURE = 0.8
BANNED_TOKENS = ["<pad>", "<unk>"]   # never sampled, for every model

