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

vocab_size = len(vocab)
embed_dim = 8

# Random Embedding matrix
E = np.random.randn(vocab_size, embed_dim)

X = np.array([E[token] for token in token_ids])

#----------Init Weights----------
hidden = 16

Wq = np.random.randn(embed_dim,embed_dim)
Wk = np.random.randn(embed_dim,embed_dim)
Wv = np.random.randn(embed_dim,embed_dim)

W1 = np.random.randn(embed_dim,hidden)
b1 = np.random.randn(hidden)

W2 = np.random.randn(hidden,embed_dim)
b2 = np.random.randn(embed_dim)

gamma1 = np.ones(embed_dim)
beta1 = np.zeros(embed_dim)

gamma2 = np.ones(embed_dim)
beta2 = np.zeros(embed_dim)

#----------transformer----------
output, weights = transformer_block_prenorm(X, Wq, Wk, Wv, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2)

#----------Printing----------
print("Sentence: ", sentence)
print("Token IDs: ", token_ids)
print("Input Shape: ", X.shape)
print("Output Shape", output.shape)
print()
print("Output Vectors (first 2 tokens): ")
print(output[:2])
print()
print("Attention Matrix:")
print(weights)