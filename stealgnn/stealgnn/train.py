print("Training the victim model...")
num_victim_epochs = 200
for epoch in range(num_victim_epochs):
    victim.train()
    victim_optimizer.zero_grad()

    out = victim(data.x, data.edge_index)

    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    victim_optimizer.step()

    if (epoch + 1) % 20 == 0:
        victim.eval()
        with torch.no_grad():
            pred = victim(data.x, data.edge_index).argmax(dim=1)
            mask_to_eval = data.val_mask if hasattr(data, 'val_mask') else data.test_mask
            correct = (pred[mask_to_eval] == data.y[mask_to_eval]).sum()
            acc = int(correct) / int(mask_to_eval.sum())
            print(f"Epoch {epoch+1}/{num_victim_epochs}, Victim Training Loss: {loss.item():.4f}, Victim Val/Test Acc: {acc:.4f}")
