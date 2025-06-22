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

# Run
```bash
pip install -r requirements.txt
```
^Installs the libraries

```bash
python main.py --dataset <dataset> --attack_type <attack_type> 
```
^Runs the attack

```<attack_type>``` can either be 1, 2, or 3

```<dataset>``` can either be ```cora```, ```ogb-arxiv```, ```pubmed```, or ```computers```

# Datasets used
- Cora
- Computers
- Pubmed
- OGB-Arxiv

# Evaluation
- Accuracy: Agreement with true test labels
- Fidelity: Agreement with the victim model’s predictions on the test set, indicating how well the attack reproduced the original model's behavior

# Result
This section documents the STEALGNN results aligned with the experimental tables in the original paper. Values marked with `~` are approximate, taken from logs or estimation.

### Cora

| Attack Type | Victim Acc | Surrogate Acc | Fidelity | Final Gen Loss | Final Surr Loss | Notes         |
|-------------|-------------|----------------|----------|----------------|------------------|---------------|
| Type I      | ~0.80       | ~0.75          | ~0.86    | ~-0.99         | 0.0002           | queries = 1000 |
| Type II     | ~0.80       | ~0.75          | ~0.84    | ~-0.96         | 0.0003           | queries = 1000 |
| Type III    | ~0.80       | ~0.12          | ~0.11    | -0             | Both 1.94        | queries = 1000 |

### Pubmed

| Attack Type | Victim Acc | Surrogate Acc | Fidelity | Final Gen Loss | Final Surr Loss | Notes |
|-------------|-------------|----------------|----------|----------------|------------------|-------|
| Type I      | ~0.65       | ~0.65          | ~0.70    | ~-0.198        | 0.6              |       |
| Type II     | ~0.787      | 0.766          | ~0.94    | ~-0.35         | 0.12             |       |
| Type III    | ~0.79       | N/A            | N/A      | N/A            | N/A              |       |

### Computers

| Attack Type | Victim Acc | Surrogate Acc | Fidelity | Final Gen Loss | Final Surr Loss | Notes         |
|-------------|-------------|----------------|----------|----------------|------------------|---------------|
| Type I      | ~0.65       | ~0.38          | 0.55     | -0.30          | 0.28             | queries = 500 |
| Type II     | ~0.64       | ~0.40          | ~0.50    | -0.12          | 0.02             | queries = 500 |
| Type III    | N/A         | N/A            | N/A      | N/A            | N/A              | Out of memory |



# License
This project is licensed under the MIT License. Reference the ``LICENSE`` file in the repository for more detail.

# Reference
Zhuang, Y., Shi, C., Zhang, M., Chen, J., Lyu, L., Zhou, P., Sun, L. (2024) Unveiling the Secrets without Data: Can Graph Neural Networks Be Exploited through Data-Free Model Extraction Attacks?
https://pure.psu.edu/en/publications/unveiling-the-secrets-without-data-can-graph-neural-networks-be-e
