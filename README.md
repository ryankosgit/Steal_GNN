# Steal_GNN

This is an implementation of StealGNN described in the paper "Unveiling the Secrets without Data: Can Graph Neural Networks Be Exploited through Data-Free Model Extraction Attacks?" (Zhuang, et al).

# Project Structure

```text
stealgnn/
├── attacks/
│   ├── __init__.py
│   ├── type_i_ii.py         # Type-I and Type-II attack logic
│   └── type_iii.py          # Type-III attack logic
├── data_loader.py           # Loads datasets (Cora, Pubmed, etc.)
├── models.py                # GCN and generator architectures
├── run_attack.py            # Attack execution logic
├── train.py                 # Victim training loop
main.py                      # CLI entry point to launch attacks
README.md
requirements.txt             
```

# Files & Folder Info

## main.py

The primary script to launch experiments. It parses command-line arguments, loads datasets, initializes models and generators, and executes the specified attack while logging performance metrics.


