import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = torch.device('mps' if torch.has_mps else 'cpu')
print(f"The model is tested on {device}")

# Function to compute Kovasznay flow analytical solution
def Kovasznay(x1, x2, Re):
    Nu = 1. / Re
    lambda1 = 1. / (2. * Nu) - np.sqrt(1. / (4. * Nu * Nu) + 4. * np.pi * np.pi)
    u = 1. - np.exp(lambda1 * x1) * np.cos(2. * np.pi * x2)
    v = lambda1 / (2. * np.pi) * np.exp(lambda1 * x1) * np.sin(2. * np.pi * x2)
    p = 0.5 * (1. - np.exp(2. * lambda1 * x1))
    return u, v, p

class NavierStokes(nn.Module):
    def __init__(self):
        super(NavierStokes, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, x, y):
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        input_tensor = torch.cat((x, y), dim=1)
        res = self.net(input_tensor)
        u, v, p = res[:, 0:1], res[:, 1:2], res[:, 2:3]
        return u, v, p

# Load the trained model
model = NavierStokes().to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()

# Generate test data
x_test = np.linspace(0, 1.0, 100)
y_test = np.linspace(0, 1.0, 100)
X_test, Y_test = np.meshgrid(x_test, y_test)
x_flat_test = X_test.flatten()
y_flat_test = Y_test.flatten()

# Convert test data to tensors
x_tensor_test = torch.tensor(x_flat_test, dtype=torch.float32, requires_grad=True).to(device)
y_tensor_test = torch.tensor(y_flat_test, dtype=torch.float32, requires_grad=True).to(device)

# Make predictions
with torch.no_grad():
    u_pred, v_pred, p_pred = model(x_tensor_test, y_tensor_test)

# Convert predictions to numpy arrays for plotting
u_pred = u_pred.cpu().numpy().reshape(100, 100)
v_pred = v_pred.cpu().numpy().reshape(100, 100)
p_pred = p_pred.cpu().numpy().reshape(100, 100)

# Plot the results
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.imshow(u_pred, extent=(x_test.min(), x_test.max(), y_test.min(), y_test.max()), origin='lower', cmap='jet')
plt.title('Predicted u')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(v_pred, extent=(x_test.min(), x_test.max(), y_test.min(), y_test.max()), origin='lower', cmap='jet')
plt.title('Predicted v')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(p_pred, extent=(x_test.min(), x_test.max(), y_test.min(), y_test.max()), origin='lower', cmap='jet')
plt.title('Predicted p')
plt.colorbar()

plt.show()