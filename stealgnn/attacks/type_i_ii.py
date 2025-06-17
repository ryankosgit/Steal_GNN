class TypeI_IIAttack:
    def __init__(self, generator, surrogate, victim, device,
                 noise_dim, num_nodes, feature_dim,
                 generator_lr=1e-6, surrogate_lr=0.001,
                 n_generator_steps=2, n_surrogate_steps=5):

        self.device = device
        self.generator = generator.to(device)
        self.surrogate_model = surrogate.to(device)
        self.victim_model = victim.to(device)

        self.noise_dim = noise_dim
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=generator_lr)
        self.surrogate_optimizer = optim.Adam(self.surrogate_model.parameters(), lr=surrogate_lr)
        self.criterion = nn.CrossEntropyLoss()

        self.n_generator_steps = n_generator_steps
        self.n_surrogate_steps = n_surrogate_steps

    def gen_graph(self):
        features, edge_index = self.generator(self.generator.threshold)
        return features.to(self.device), edge_index.to(self.device)

    def train_generator(self):
      self.generator.train()
      self.surrogate_model.eval()

      self.generator_optimizer.zero_grad()

      features, edge_index = self.gen_graph()

      victim_logits = self.victim_model(features, edge_index).detach()
      surrogate_logits = self.surrogate_model(features, edge_index)

      victim_probs = F.softmax(victim_logits, dim=1)
      log_surr_probs = F.log_softmax(surrogate_logits, dim=1)

      loss = -F.kl_div(log_surr_probs, victim_probs, reduction='batchmean')

      loss.backward()
      self.generator_optimizer.step()

      return loss.item()


    def train_surrogate(self):
        self.surrogate_model.train()

        self.surrogate_optimizer.zero_grad()

        features, edge_index = self.gen_graph()
        victim_output = self.victim_model(features, edge_index).detach()
        hard_labels = victim_output.argmax(dim=1)

        output = self.surrogate_model(features, edge_index)

        loss = self.criterion(output, hard_labels)
        loss.backward()
        self.surrogate_optimizer.step()

        return loss.item()

    def attack(self, num_queries, warmup_steps):
      gen_losses = []
      surr_losses = []

      print(f"Warmup Phase: Training surrogate for {warmup_steps} steps...")
      for step in range(warmup_steps):
          loss = self.train_surrogate()
          if step % 2 == 0:
              print(f"  Warmup step {step}: Surrogate loss = {loss:.4f}")

      print("Starting adversarial training...")
      for _ in tqdm(range(num_queries // (self.n_generator_steps + self.n_surrogate_steps))):
          for _ in range(self.n_generator_steps):
              gen_loss = self.train_generator()
              gen_losses.append(gen_loss)

          for _ in range(self.n_surrogate_steps):
              surr_loss = self.train_surrogate()
              surr_losses.append(surr_loss)

      return self.surrogate_model, gen_losses, surr_losses


    def evaluate(self, model, data):
        model.eval()
        self.victim_model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)
            labels = data.y
            acc = (preds[data.test_mask] == labels[data.test_mask]).float().mean().item()

            with torch.no_grad():
                victim_preds = self.victim_model(data.x, data.edge_index).argmax(dim=1)
            fidelity = (preds[data.test_mask] == victim_preds[data.test_mask]).float().mean().item()

        return acc, fidelity
