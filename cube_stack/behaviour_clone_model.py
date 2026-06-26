import torch
import torch.nn as nn

class BehaviorCloningBaseline(nn.Module):
    def __init__(self, input_dim=32, horizon=8, action_dim=7):
        super(BehaviorCloningBaseline, self).__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),           # add normalization inside network
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, horizon * action_dim),
        )
        
    def forward(self, state):
        flat_actions = self.mlp(state)
        batch_size = state.size(0)
        action_sequence = flat_actions.view(batch_size, self.horizon, self.action_dim)
        return action_sequence


if __name__ == "__main__":
    # Quick check to ensure shapes match up:
    model = BehaviorCloningBaseline()
    sample_input = torch.randn(1, 32)
    sample_output = model(sample_input)
    print(sample_output.shape)  # Expected: torch.Size([1, 8, 7])