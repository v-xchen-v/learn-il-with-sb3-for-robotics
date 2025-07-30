import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


class BCDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = torch.tensor(observations, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32 if actions.ndim == 2 else torch.long)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256, 256], is_discrete=False):
        super().__init__()
        self.is_discrete = is_discrete
        layers = []
        dims = [obs_dim] + hidden_sizes
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        logits_or_action = self.net(obs)
        if self.is_discrete:
            return logits_or_action  # logits for classification
        else:
            return logits_or_action  # actions for regression


class BehaviorCloning:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        is_discrete: bool = False,
        hidden_sizes: List[int] = [256, 256],
        lr: float = 1e-3,
        device: str = "cpu"
    ):
        self.device = device
        self.is_discrete = is_discrete
        self.policy = MLPPolicy(obs_dim, act_dim, hidden_sizes, is_discrete).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.loss_fn = (
            nn.CrossEntropyLoss() if is_discrete else nn.MSELoss()
        )

    def train(
        self,
        observations,
        actions,
        batch_size=64,
        epochs=10,
        shuffle=True,
        verbose=True,
    ):
        dataset = BCDataset(observations, actions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(epochs):
            total_loss = 0.0
            for obs_batch, act_batch in dataloader:
                obs_batch = obs_batch.to(self.device)
                act_batch = act_batch.to(self.device)

                pred = self.policy(obs_batch)
                if self.is_discrete:
                    loss = self.loss_fn(pred, act_batch)
                else:
                    loss = self.loss_fn(pred, act_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * obs_batch.size(0)

            avg_loss = total_loss / len(dataset)
            if verbose:
                print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        self.policy.eval()
        with torch.no_grad():
            obs = obs.to(self.device)
            output = self.policy(obs)
            if self.is_discrete:
                return torch.argmax(output, dim=-1)
            else:
                return output

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
