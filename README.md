# Math Modeling Lab

A repository for mathematical modeling experiments, focusing on **bifurcation analysis** and **Physics-Informed Neural Networks (PINNs)**.

## Structure

```
├── Bifurcation_Analysis/   # Jupyter notebooks for bifurcation analysis
├── Papers/                 # Reference papers and publications
├── PINN/                   # Physics-Informed Neural Networks implementations
│   └── LM-r1/              # Levenberg-Marquardt based experiments
└── Report/                 # Lab reports and documentation
```

## Overview

### Bifurcation Analysis
Notebooks exploring dynamical systems and bifurcation diagrams.

### PINN
Neural network approaches for solving differential equations and detecting bifurcation points. Includes:
- Training scripts with Adam optimizer
- Levenberg-Marquardt based methods
- Pretrained model checkpoints (`.pth`)

## Requirements

- Python 3.x
- PyTorch
- Jupyter Notebook
- NumPy, Matplotlib
