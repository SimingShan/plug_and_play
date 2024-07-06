import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from plot_uvp import plot_uvp

device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda')
print(f"Training the model with {device}")


# Define the Base Network (PINN)
def network():
    net = nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 3)
    ).to(device)
    return net


base_net = network()
base_net.load_state_dict(torch.load('model.pt', map_location=device))


# Define the Hypernetwork with dynamic rank input
class Hypernetwork(nn.Module):
    def __init__(self, rank):
        super(Hypernetwork, self).__init__()
        self.rank = rank
        # Calculate total number of parameters based on the rank
        total_params = (2 * rank + rank * 64) + 4 * (64 * rank + rank * 64) + (64 * rank + rank * 3)
        self.fc = nn.Linear(1, total_params)  # Output: Parameters for LoRA

    def forward(self, reynolds):
        params = self.fc(reynolds)
        return params


# Combine the Hypernetwork with LoRA
class LoRAWithHypernetwork(nn.Module):
    def __init__(self, base_network, hypernetwork):
        super(LoRAWithHypernetwork, self).__init__()
        self.base_network = base_network
        self.hypernetwork = hypernetwork

    def forward(self, x, reynolds):
        params = self.hypernetwork(reynolds)
        rank = self.hypernetwork.rank

        # Reshape the parameters into low-rank matrices for each layer
        split_sizes = [2 * rank + rank * 64, 64 * rank + rank * 64, 64 * rank + rank * 64, 64 * rank + rank * 64,
                       64 * rank + rank * 64, 64 * rank + rank * 3]
        param_splits = torch.split(params, split_sizes, dim=1)

        # Store the modified weights
        modified_weights = []

        # Layer 1: 2*1 + 1*64
        L1 = param_splits[0][:, :2 * rank].view(2, rank)
        L2 = param_splits[0][:, 2 * rank:].view(rank, 64)
        modified_weights.append(self.base_network[0].weight + torch.matmul(L1, L2).t())

        # Layers 2-5: 64*1 + 1*64
        for i in range(1, 5):
            L1 = param_splits[i][:, :64 * rank].view(64, rank)
            L2 = param_splits[i][:, 64 * rank:].view(rank, 64)
            modified_weights.append(self.base_network[2 * i].weight + torch.matmul(L1, L2).t())

        # Layer 6: 64*1 + 1*3
        L1 = param_splits[5][:, :64 * rank].view(64, rank)
        L2 = param_splits[5][:, 64 * rank:].view(rank, 3)
        modified_weights.append(self.base_network[10].weight + torch.matmul(L1, L2).t())

        # Apply the modified weights
        out = x
        out = torch.matmul(out, modified_weights[0].t()) + self.base_network[0].bias
        out = self.base_network[1](out)
        for i in range(1, 5):
            out = torch.matmul(out, modified_weights[i].t()) + self.base_network[2 * i].bias
        out = self.base_network[2 * i + 1](out)
        out = torch.matmul(out, modified_weights[5].t()) + self.base_network[10].bias

        return out


# Define the Kovasznay analytical solution function
def Kovasznay(x1, x2, Re):
    Nu = 1. / Re
    lambda1 = 1. / (2. * Nu) - np.sqrt(1. / (4. * Nu * Nu) + 4. * np.pi * np.pi)
    u = 1. - np.exp(lambda1 * x1) * np.cos(2. * np.pi * x2)
    v = lambda1 / (2. * np.pi) * np.exp(lambda1 * x1) * np.sin(2. * np.pi * x2)
    p = 0.5 * (1. - np.exp(2. * lambda1 * x1))
    return u, v, p


# Generate data for Re = 40
x = np.linspace(0, 1.0, 100)
y = np.linspace(0, 1.0, 100)
X, Y = np.meshgrid(x, y)
x_flat = X.flatten()
y_flat = Y.flatten()
Re = 40
u, v, p = Kovasznay(X, Y, Re)
u = u.flatten()
v = v.flatten()
p = p.flatten()

# Use only 10% of data
n_samples = len(x_flat)
indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
x_flat_train = x_flat[indices]
y_flat_train = y_flat[indices]
u_train = u[indices]
v_train = v[indices]
p_train = p[indices]

# Prepare the training data
train_data = torch.tensor(np.vstack((x_flat_train, y_flat_train)).T, dtype=torch.float32).to(device)
train_targets = torch.tensor(np.vstack((u_train, v_train, p_train)).T, dtype=torch.float32).to(device)
train_loader = torch.utils.data.DataLoader(list(zip(train_data, train_targets)), batch_size=32, shuffle=True)

# Define loss function and optimizer for LoRA training
criterion = nn.MSELoss()
rank = 1  # Change this value for different ranks
hypernetwork = Hypernetwork(rank).to(device)
model = LoRAWithHypernetwork(base_net, hypernetwork).to(device)
lora_optimizer = optim.Adam(model.hypernetwork.parameters(), lr=1e-3)

# Freeze the base network's parameters
for param in base_net.parameters():
    param.requires_grad = False

# Training loop for LoRA weights
model.train()
for epoch in range(10000):
    for x, target in train_loader:
        reynolds = torch.tensor([[Re]], dtype=torch.float32).to(device)  # Reynolds number input
        lora_optimizer.zero_grad()
        output = model(x, reynolds)
        loss = criterion(output, target)
        loss.backward()
        lora_optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Optionally, save the LoRA model
torch.save(hypernetwork.state_dict(), 'lora_hypernetwork_re40.pt')

# Plotting the results
plot_uvp(X, Y, u, v, p, model, device, Re)