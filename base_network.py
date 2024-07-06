import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from plot_uvp import plot_uvp
device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda')
print(f"Training the model with {device}")
# Function to compute Kovasznay flow analytical solution
def Kovasznay(x1, x2, Re):
    Nu = 1. / Re
    lambda1 = 1. / (2. * Nu) - np.sqrt(1. / (4. * Nu * Nu) + 4. * np.pi * np.pi)
    u = 1. - np.exp(lambda1 * x1) * np.cos(2. * np.pi * x2)
    v = lambda1 / (2. * np.pi) * np.exp(lambda1 * x1) * np.sin(2. * np.pi * x2)
    p = 0.5 * (1. - np.exp(2. * lambda1 * x1))
    return u, v, p

class NavierStokes():
    def __init__(self, X, Y, p, u, v, re):
        self.re = re
        self.device = device
        self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(self.device)
        self.y = torch.tensor(Y, dtype=torch.float32, requires_grad=True).to(self.device)
        self.p = torch.tensor(p, dtype=torch.float32, requires_grad=True).to(self.device)
        self.u = torch.tensor(u, dtype=torch.float32, requires_grad=True).to(self.device)
        self.v = torch.tensor(v, dtype=torch.float32, requires_grad=True).to(self.device)
        self.null = torch.zeros((self.x.shape[0], 1)).to(self.device)
        self.network()
        self.mse = nn.MSELoss()
        self.iter = 0

    def network(self):
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        ).to(self.device)

    def function(self, x, y):
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        input_tensor = torch.cat((x, y), dim=1)
        res = self.net(input_tensor)
        u, v, p = res[:, 0:1], res[:, 1:2], res[:, 2:3]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f = u * u_x + v * u_y + p_x - 1 / self.re * (u_xx + u_yy)
        g = u * v_x + v * v_y + p_y - 1 / self.re * (v_xx + v_yy)

        continuity = u_x + v_y
        return u, v, p, f, g, continuity

    def closure(self):
        self.optimizer.zero_grad()
        u_prediction, v_prediction, p_prediction, f_prediction, g_prediction, continuity_prediction = self.function(self.x, self.y)
        u_target = self.u.view(-1, 1)
        v_target = self.v.view(-1, 1)
        p_target = self.p.view(-1, 1)
        u_loss = self.mse(u_prediction, u_target)
        v_loss = self.mse(v_prediction, v_target)
        p_loss = self.mse(p_prediction, p_target)
        f_loss = self.mse(f_prediction, self.null)
        g_loss = self.mse(g_prediction, self.null)
        continuity_loss = self.mse(continuity_prediction, self.null)
        total_loss = u_loss + v_loss + f_loss + g_loss + p_loss + continuity_loss
        total_loss.backward()
        self.iter += 1
        if self.iter % 1000 == 0:
            print('Iteration: {:}, Loss: {:0.10f}'.format(self.iter, total_loss.item()))
        return total_loss

    def train_adam(self, num_epochs=30000, initial_lr=1e-3):
        self.optimizer = optim.Adam(self.net.parameters(), lr=initial_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1000,
                                                         min_lr=1e-7, verbose=True)
        self.net.train()
        best_loss = float('inf')
        best_iter = 0
        for epoch in range(num_epochs):
            loss = self.closure()
            self.optimizer.step()
            if epoch >= 10000:
                scheduler.step(loss)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_iter = epoch
            if epoch - best_iter > 1000:
                print(f"Early stopping at epoch {epoch} with best loss {best_loss:.6f}")
                break
    def train_lbfgs(self):
        self.optimizer = optim.LBFGS(self.net.parameters(), max_iter=50000, tolerance_grad=1e-9, tolerance_change=1e-9, history_size=100)
        self.net.train()
        def closure():
            self.optimizer.zero_grad()
            loss = self.closure()
            return loss
        self.optimizer.step(closure)


# Generate data
x = np.linspace(0, 1.0, 100)
y = np.linspace(0, 1.0, 100)
X, Y = np.meshgrid(x, y)
x_flat = X.flatten()
y_flat = Y.flatten()
Re = 20
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

# Use reduced dataset for training
pinn = NavierStokes(x_flat_train, y_flat_train, p_train, u_train, v_train, Re)
#pinn.train_adam()
#pinn.train_lbfgs()
#torch.save(pinn.net.state_dict(), 'model.pt')
pinn.net.load_state_dict(torch.load('model.pt', map_location=device))
pinn.net.eval()
# Print the number of parameters
total_params = sum(p.numel() for p in pinn.net.parameters())
print(f"Number of parameters in the network: {total_params}")
# Generate full grid for evaluation
u_out, v_out, p_out, f_out, g_out, continuity = pinn.function(
    torch.tensor(X.flatten(), dtype=torch.float32, requires_grad=True).to(device),
    torch.tensor(Y.flatten(), dtype=torch.float32, requires_grad=True).to(device))

# Calculate and print the MSE
u_target = torch.tensor(u, dtype=torch.float32).view(-1, 1).to(device)
v_target = torch.tensor(v, dtype=torch.float32).view(-1, 1).to(device)
p_target = torch.tensor(p, dtype=torch.float32).view(-1, 1).to(device)
u_loss = pinn.mse(u_out, u_target).item()
v_loss = pinn.mse(v_out, v_target).item()
p_loss = pinn.mse(p_out, p_target).item()
print(f'MSE for u: {u_loss:.10f}')
print(f'MSE for v: {v_loss:.10f}')
print(f'MSE for p: {p_loss:.10f}')

plot_uvp(x_flat, y_flat, u_out, v_out, p_out, u_target, v_target, p_target, 'kovasznay')