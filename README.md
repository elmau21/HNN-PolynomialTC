# Hamiltonian Neural Networks for Path Finding Optimization

## Overview
This project implements a novel approach to optimizing polynomial time complexity in Hamiltonian path finding using Hamiltonian Neural Networks (HNN). Inspired by the anime "Science Fell in Love, So I Tried to Prove It" (理系が恋に落ちたので証明してみた), this implementation combines principles from classical mechanics with modern deep learning techniques.

## Features
- **Hamiltonian Neural Network Implementation**
  - Custom energy conservation mechanisms
  - Symplectic integration using leapfrog method
  - Attention mechanism for graph understanding
  - Advanced gradient flow analysis

- **Path Finding Capabilities**
  - Finds Hamiltonian paths in graphs
  - Optimizes polynomial time complexity
  - Validates path correctness
  - Visualizes results

## Mathematical Foundation
The implementation is based on Hamiltonian mechanics principles:
```python
H(q,p) = T(p) + V(q)  # Hamiltonian
dq/dt = ∂H/∂p        # Hamilton's equations
dp/dt = -∂H/∂q
```

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Networkx

## Installation
```bash
git clone https://github.com/elmau21/HNN-PolynomialTC.git
cd hamiltonian-nn
```

## Usage
Basic usage example:
```python
from hamiltonian_nn import EnhancedHamiltonianNN, HamiltonianConfig

# Create configuration
config = HamiltonianConfig(
    input_dim=n_vertices,
    hidden_dim=64,
    time_steps=100,
    use_attention=True
)
