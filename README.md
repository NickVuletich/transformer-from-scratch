# Transformer From Scratch

This project implements the mathematics behind the Transformer architecture from scratch using **NumPy**.

The goal is to deeply understand the **linear algebra and core mechanisms** behind modern AI models rather than relying on high-level frameworks like PyTorch or TensorFlow.

Modern AI systems such as **GPT, BERT, and LLaMA** are built using the Transformer architecture.  
This repository rebuilds the core components step-by-step to expose the mathematics behind:

- attention mechanisms
- feedforward networks
- normalization layers
- residual connections

---

# Architecture Overview

A transformer block consists of two main components:

```text
    Input
      ↓
Self Attention
      ↓
Residual Connection
      ↓
Layer Normalization
      ↓
Feedforward Network (MLP)
      ↓
Residual Connection
      ↓
Layer Normalization
```

Mathematically:
```text
Z1 = LayerNorm(X + Attention(X))
Z2 = LayerNorm(Z1 + MLP(Z1))
```
---

# Implemented Components

The following building blocks are implemented from scratch:

### Core Math Layers
- Linear Layer
- Softmax
- Layer Normalization

### Activation Functions
- ReLU
- GELU
- tanh

### Transformer Components
- Self Attention
- Feedforward Network (MLP)
- Transformer Block

---

# Project Structure
```text
transformer-from-scratch/

  nn_math/
    linear.py
    softmax.py
    layernorm.py
    activations.py
    
  attention/
    self_attention.py
  
  transformer/
    mlp.py
    transformer_block.py
  
  tests/
    test_activations.py
    test_linear.py
    test_softmax.py
    test_layernorm.py
    test_attention.py
    test_mlp.py
    test_all.py
```


---

# Running Tests

To run all tests:
```bash
python -m tests.test_all
```


---

# Purpose

The goal of this project is **educational**.

Instead of using prebuilt deep learning layers, each component is implemented directly from its mathematical definition to understand:

- how attention works internally
- how token relationships are computed
- how transformer blocks process information

---
# Author

## Nicholas Vuletich

