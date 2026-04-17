# Programmer: Nicholas Vuletich
# Date: 03-16-2026
# File run_attention_example.py

import numpy as np
from nn_math.softmax import softmax
from transformer.transformer_block_prenorm import transformer_block_prenorm
from transformer.mha_block import MHA_transformer_block_prenorm

np.random.seed(42)

#----------Load dataset----------
with open("data/big_simple_text.txt", "r") as f:
    text = f.read().lower()

words = text.split()

#----------Vocabulary----------
vocab = {word: i for i, word in enumerate(set(words))}
idx_to_word = {i: word for word, i in vocab.items()}

vocab_size = len(vocab)

#----------Making sequences----------
sequence_length = 4

inputs = []
targets = []

for i in range(len(words) - sequence_length):
    seq = words[i:i+sequence_length]
    target = words[i+sequence_length]

    inputs.append([vocab[w] for w in seq])
    targets.append(vocab[target])


#----------Embedding----------

vocab_size = len(vocab)
embed_dim = 8

# Random Embedding matrix
E = np.random.randn(vocab_size, embed_dim)

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

W_out = np.random.randn(embed_dim, vocab_size)

#----------Training Loop----------
learning_rate = 0.01

for epoch in range(3000):
    total_loss = 0

    for seq, target in zip(inputs, targets):

        # Convert tokens to embeddings
        X = np.array([E[token] for token in seq])

        # Forward pass
        output, _ = MHA_transformer_block_prenorm(
            X, Wq, Wk, Wv,
            W1, b1,
            W2, b2,
            gamma1, beta1,
            gamma2, beta2
        )

        logits = output @ W_out
        probs = softmax(logits)

        # The last token is the only one to predict the next word
        p = probs[-1]

        # Loss
        loss = -np.log(p[target] + 1e-9)
        total_loss += loss

        # Gradient output layuer only
        grad = p.copy()
        grad[target] -= 1

        dW_out = np.outer(output[-1], grad)

        # Update
        W_out -= learning_rate * dW_out

    if epoch % 100 == 0:
        avg_loss = total_loss / len(inputs)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

np.save("W_out.npy", W_out)
np.save("E.npy", E)
print("Training complete.\n")

#----------Test predictions----------
print("\n---TEST---\n")

for seq in inputs[:5]:
    X = np.array([E[token] for token in seq])

    output, _ = MHA_transformer_block_prenorm(
        X, Wq, Wk, Wv,
        W1, b1,
        W2, b2,
        gamma1, beta1,
        gamma2, beta2
    )

    logits = output @ W_out
    probs = softmax(logits)

    pred = np.argmax(probs[-1])

    words_seq = [idx_to_word[i] for i in seq]

    print(f"Input: {words_seq}")
    print(f"Predicted next: {idx_to_word[pred]}")
    print()

#----------Printing visuals----------

print("\n---Attention Breakdown (Demo)---\n")

demo_seq = ["the", "cat", "sat", "on", "the"]
demo_tokens = [vocab[w] for w in demo_seq]
X_demo = np.array([E[token] for token in demo_tokens])

_, weights = MHA_transformer_block_prenorm( X_demo,Wq, Wk, Wv, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2 )

for i, row in enumerate(weights):
    current_word = demo_seq[i]
    print(f"Word: '{current_word}' looks at:")

    for j, score in enumerate(row):
        target_word = demo_seq[j]
        print(f"  {target_word:>5} -> {score*100:5.1f}%")
    print()
