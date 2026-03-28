# Programmer: Nicholas Vuletich
# Date: 03-16-2026
# File run_attention_example.py

import numpy as np
from transformer.transformer_block_prenorm import transformer_block_prenorm

#----------Vocabulary----------
vocab = {
    "the": 0,
    "cat": 1,
    "sat": 2,
    "on": 3,
    "mat": 4
}

sentence = ["the", "cat", "sat", "on", "the", "mat"]

token_ids = [vocab[word] for word in sentence]

#----------Embedding----------

