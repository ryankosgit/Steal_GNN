# StealGNN

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

## attacks/
- `type_i_ii`: Implements the Type I _or_ Type II attack, depends on user entry
- `type_iii`: Implements the Type III attack

In Type I, the gradients are propogated through the surrogate model and the gradient approximation of the victim model.

In Type II, the gradients are propogated through one surrogate model.

In Type III, the gradients are passed back through two surrogate models. 

# Datasets used
- Cora
- Amazon Computers
- Pubmed
- OGB-Arxiv

# Evaluation
- Accuracy: Agreement with true test labels
- Fidelity: Agreement with the victim model’s predictions on the test set, indicating how well the attack reproduced the original model's behavior

# License
This project is licensed under the MIT License. Reference the ``LICENSE`` file in the repository for more detail.

# Reference
Zhuang, Y., Shi, C., Zhang, M., Chen, J., Lyu, L., Zhou, P., Sun, L. (2024) Unveiling the Secrets without Data: Can Graph Neural Networks Be Exploited through Data-Free Model Extraction Attacks?
https://pure.psu.edu/en/publications/unveiling-the-secrets-without-data-can-graph-neural-networks-be-e
