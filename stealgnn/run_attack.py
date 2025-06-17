if attack_type == '1':
    model = TypeI_IIAttack(generator, surrogate, victim, device,
                       noise_dim=noise_dim, num_nodes=num_nodes, feature_dim=feature_dim)
    print("Type-I Attack running")
    surrogate_model, gen_losses, surr_losses = model.attack(num_queries=1000, warmup_steps=20)
    print(f"Final Gen Loss: {gen_losses[-1]:.4f}, Final Surr Loss: {surr_losses[-1]:.4f}")
    surr_acc, surr_fid = model.evaluate(surrogate_model, data)
    print(f"Surrogate Accuracy: {surr_acc:.4f}, Fidelity: {surr_fid:.4f}")
elif attack_type == '2':
    model = TypeI_IIAttack(generator, surrogate, victim, device,
                           noise_dim=noise_dim, num_nodes=num_nodes, feature_dim=feature_dim)
    print("Type-II Attack running")
    surrogate_model, gen_losses, surr_losses = model.attack(num_queries=1000, warmup_steps=15)
    print(f"Final Gen Loss: {gen_losses[-1]:.4f}, Final Surr Loss: {surr_losses[-1]:.4f}")
    surr_acc, surr_fid = model.evaluate(surrogate_model, data)
    print(f"Surrogate Accuracy: {surr_acc:.4f}, Fidelity: {surr_fid:.4f}")

elif attack_type == '3':
    generator = FullParameterizationGenerator(num_nodes, feature_dim)
    surrogate_1 = GCN(feature_dim, 64, num_classes)
    surrogate_2 = GCN(feature_dim, 64, num_classes)

    model = TypeIIIAttack(generator, surrogate_1, surrogate_2, device,
                          num_classes=num_classes, threshold=0.1,
                          feature_dim=feature_dim,
                          num_nodes=num_nodes)

    print("Type-III Attack running")
    final_surr, gen_losses, loss1, loss2 = model.attack(num_queries=600)
    print(f"Final Gen Loss: {gen_losses[-1]:.4f}, Surrogate1 Loss: {loss1[-1]:.4f}, Surrogate2 Loss: {loss2[-1]:.4f}")
    surr_acc, surr_fid = model.evaluate(final_surr, victim, data)
    print(f"Surrogate Accuracy: {surr_acc:.4f}, Fidelity: {surr_fid:.4f}")
    surr_acc, surr_fid = model.evaluate(final_surr, victim, data)
