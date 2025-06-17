import argparse
import torch
from stealgnn.data_loader import load_dataset
from stealgnn.models import GCN, CosineGenerator, FullParameterizationGenerator
from stealgnn.train import train_victim
from stealgnn.attacks.type_i_ii import TypeI_IIAttack
from stealgnn.attacks.type_iii import TypeIIIAttack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['cora', 'ogb-arxiv', 'Pubmed', 'Computers'])
    parser.add_argument('--attack_type', type=str, required=True, choices=['type-i', 'type-ii', 'type-iii'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    data, in_dim, hid_dim, num_classes = load_dataset(args.dataset)
    data = data.to(args.device)
    num_nodes = data.num_nodes
    feature_dim = in_dim

    victim = GCN(in_dim, 256, num_classes).to(args.device)
    surrogate = GCN(in_dim, 256, num_classes).to(args.device)
    victim = train_victim(victim, data)
    victim.eval()

    if args.attack_type in ['type-i', 'type-ii']:
        generator = CosineGenerator(num_nodes, feature_dim, threshold=0.1)
        attack = TypeI_IIAttack(generator, surrogate, victim, args.device,
                                noise_dim=64, num_nodes=num_nodes, feature_dim=feature_dim)
        surrogate_model, gen_losses, surr_losses = attack.attack(num_queries=1000, warmup_steps=20)
        acc, fid = attack.evaluate(surrogate_model, data)

    elif args.attack_type == 'type-iii':
        generator = FullParameterizationGenerator(num_nodes, feature_dim)
        surrogate_1 = GCN(feature_dim, 64, num_classes)
        surrogate_2 = GCN(feature_dim, 64, num_classes)
        attack = TypeIIIAttack(generator, surrogate_1, surrogate_2, args.device,
                               num_classes=num_classes, feature_dim=feature_dim,
                               num_nodes=num_nodes, threshold=0.1)
        final_surr, gen_losses, loss1, loss2 = attack.attack(num_queries=600)
        acc, fid = attack.evaluate(final_surr, victim, data)

    print(f"Surrogate Accuracy: {acc:.4f}, Fidelity: {fid:.4f}")

if __name__ == '__main__':
    main()
