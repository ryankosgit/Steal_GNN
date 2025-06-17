class TypeIIIAttack:
    def __init__(self, generator, surrogate_1, surrogate_2, device,
                 num_classes, feature_dim, num_nodes, threshold,
                 generator_lr=5e-5, surrogate_lr=1e-3,
                 n_generator_steps=2, n_surrogate_steps=5):

        self.device = device
        self.generator = generator.to(device)
        self.surrogate_1 = surrogate_1.to(device)
        self.surrogate_2 = surrogate_2.to(device)

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_nodes = num_nodes
        self.threshold = threshold

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=generator_lr)
        self.surrogate_optimizer_1 = torch.optim.Adam(self.surrogate_1.parameters(), lr=surrogate_lr)
        self.surrogate_optimizer_2 = torch.optim.Adam(self.surrogate_2.parameters(), lr=surrogate_lr)
        self.criterion = nn.CrossEntropyLoss()

        self.n_generator_steps = n_generator_steps
        self.n_surrogate_steps = n_surrogate_steps

    def gen_graph(self):
        features, edge_index = self.generator(self.threshold)
        return features.to(self.device), edge_index.to(self.device)

    def train_generator(self):
        self.generator.train()
        self.surrogate_1.eval()
        self.surrogate_2.eval()

        self.generator_optimizer.zero_grad()

        features, edge_index = self.gen_graph()

        logits_1 = torch.softmax(self.surrogate_1(features, edge_index), dim=1)
        logits_2 = torch.softmax(self.surrogate_2(features, edge_index), dim=1)

        disagreement = torch.mean(torch.std(torch.stack([logits_1, logits_2], dim=1), dim=1))
        loss = -disagreement

        loss.backward()
        self.generator_optimizer.step()
        return loss.item()

    def train_surrogates(self):
        self.surrogate_1.train()
        self.surrogate_2.train()

        features, edge_index = self.gen_graph()
        labels = torch.randint(0, self.num_classes, (self.num_nodes,)).to(self.device)


        self.surrogate_optimizer_1.zero_grad()
        output_1 = self.surrogate_1(features, edge_index)
        loss_1 = self.criterion(output_1, labels)
        loss_1.backward()
        self.surrogate_optimizer_1.step()


        self.surrogate_optimizer_2.zero_grad()
        output_2 = self.surrogate_2(features, edge_index)
        loss_2 = self.criterion(output_2, labels)
        loss_2.backward()
        self.surrogate_optimizer_2.step()

        return loss_1.item(), loss_2.item()

    def attack(self, num_queries):
        gen_losses = []
        surr_losses_1 = []
        surr_losses_2 = []

        for _ in tqdm(range(num_queries // (self.n_generator_steps + self.n_surrogate_steps))):
            for _ in range(self.n_generator_steps):
                g_loss = self.train_generator()
                gen_losses.append(g_loss)

            for _ in range(self.n_surrogate_steps):
                loss1, loss2 = self.train_surrogates()
                surr_losses_1.append(loss1)
                surr_losses_2.append(loss2)

        return self.surrogate_1, gen_losses, surr_losses_1, surr_losses_2


    def evaluate(self, surrogate, victim, data):
      surrogate.eval()
      victim.eval()
      with torch.no_grad():
          surrogate_preds = surrogate(data.x, data.edge_index).argmax(dim=1)
          victim_preds = victim(data.x, data.edge_index).argmax(dim=1)
          acc = (surrogate_preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
          fidelity = (surrogate_preds[data.test_mask] == victim_preds[data.test_mask]).float().mean().item()
      return acc, fidelity

"""**ATTACK**"""

victim = GCN(in_channels=in_dim, hidden_channels=256, out_channels=num_classes).to(device)
surrogate = GCN(in_channels=in_dim, hidden_channels=256, out_channels=num_classes).to(device)

victim_optimizer = torch.optim.Adam(victim.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

num_nodes = data.num_nodes
feature_dim = in_dim
noise_dim = 64
generator = CosineGenerator(num_nodes=num_nodes, feature_dim=feature_dim, threshold=0.1)

print(f"Dataset={dataset_name}")

if dataset_name == 'Computers':
  data = RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)(data)
  print(f"train_mask in data? {'train_mask' in data}")
