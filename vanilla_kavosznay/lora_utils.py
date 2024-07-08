from model import MLP_NS
import torch.nn as nn
from evaluation import model_testing
import torch
from hyper_lora_utils import MLP_LoRA, CombinedModel
from data_utils import prepare_data
from train_utils import train
from model import MLP_NS
import torch
import os

class MLP_NS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_NS, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        return x


class MLP_LoRA(nn.Module):
    def __init__(self, base_model, rank=4):
        super(MLP_LoRA, self).__init__()
        self.base_model = base_model
        self.rank = rank

        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Define the LoRA weights
        self.lora_layers = nn.ModuleList()
        for layer in self.base_model.children():
            if isinstance(layer, nn.Linear):
                A = nn.Linear(layer.in_features, rank, bias=False)
                B = nn.Linear(rank, layer.out_features, bias=False)
                # Initialize weights of B to zero
                nn.init.zeros_(B.weight)
                self.lora_layers.append(A)
                self.lora_layers.append(B)

    def forward(self, x):
        lora_output = x
        base_layers = list(self.base_model.children())  # Convert generator to list

        for i, layer in enumerate(base_layers):
            if isinstance(layer, nn.Linear):
                # Ensure no gradients are computed for the original forward pass of the base model
                with torch.no_grad():
                    base_output = layer(x)

                # LoRA adjustment
                A = self.lora_layers[2 * i](lora_output)
                B = self.lora_layers[2 * i + 1](A)
                x = base_output + B  # Combine the original output and LoRA adjustment

                # Update lora_output for the next layer
                lora_output = torch.tanh(x) if i < len(base_layers) - 1 else x

        return x

device = 'mps' if torch.backends.mps.is_built() else 'cuda'
base_model = MLP_NS(2, 64, 3)
base_model.load_state_dict(torch.load("./checkpoint/model_checkpoint_Reynolds20.pth", map_location=device))
# Move the model to the appropriate device
base_model.to(device)
# Define LoRA model
lora_model = MLP_LoRA(base_model, rank=4).to(device)

device = 'mps' if torch.backends.mps.is_built() else 'cuda'

grid_size = 200
Re = 100
num_epochs = 10000
initial_lr = 1e-2

# Prepare data
x, y, u, v, p = prepare_data(grid_size, Re, device, 0.8)

base_model = MLP_NS(2, 64, 3)
base_model.load_state_dict(torch.load("./checkpoint/model_checkpoint_Reynolds20.pth", map_location=device))
# Move the model to the appropriate device
base_model.to(device)
# Define LoRA model
lora_model = MLP_LoRA(base_model, rank=1).to(device)
# Ensure there are parameters in lora_layers
assert sum(p.numel() for p in lora_model.lora_layers.parameters()) > 0, "lora_layers has no parameters"
print(lora_model.lora_layers[0])
train(lora_model, x, y, u, v, p, Re, num_epochs, initial_lr, lora_model.lora_layers)
u_loss, v_loss, p_loss, f_loss, g_loss, continuity_loss = model_testing(lora_model, grid_size=100, Re=100, device=device)
'without lora: Epoch: 100, Loss: 0.1488321126, Current LR: 1.0e-03, Forward Time: 0.0081s, Backward Time: 0.0552s, Update Time: 0.0019s'
'with lora basemodel_param frozen: Epoch: 100, Loss: 8.1476831436, Current LR: 1.0e-03, Forward Time: 0.0170s, Backward Time: 0.0934s, Update Time: 0.0020s'
save_path = "./lora_model_trained.pth"
torch.save(lora_model.state_dict(), save_path)
print("Model saved to", save_path)