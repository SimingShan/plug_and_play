import numpy as np
import matplotlib.pyplot as plt
import torch

def Kovasznay(x1, x2, Re):
    Nu = 1. / Re
    lambda1 = 1. / (2. * Nu) - np.sqrt(1. / (4. * Nu * Nu) + 4. * np.pi * np.pi)
    u = 1. - np.exp(lambda1 * x1) * np.cos(2. * np.pi * x2)
    v = lambda1 / (2. * np.pi) * np.exp(lambda1 * x1) * np.sin(2. * np.pi * x2)
    p = 0.5 * (1. - np.exp(2. * lambda1 * x1))
    return u, v, p


def prepare_data(grid_size, Re, device, percentage):
    x = np.linspace(0, 1.0, grid_size)
    y = np.linspace(0, 1.0, grid_size)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    u, v, p = Kovasznay(X, Y, Re)  # Ensure this function returns numpy arrays
    u, v, p = u.flatten(), v.flatten(), p.flatten()
    # Select a subset of data based on the percentage
    total_samples = x_flat.shape[0]
    num_samples = int(total_samples * percentage)
    indices = np.random.choice(total_samples, num_samples, replace=False)

    # Convert selected data to tensors and move to the specified device
    x_selected = x_flat[indices]
    y_selected = y_flat[indices]
    x_selected = torch.tensor(x_selected, dtype=torch.float32, requires_grad=True).to(device)
    y_selected = torch.tensor(y_selected, dtype=torch.float32, requires_grad=True).to(device)
    u_selected = torch.tensor(u.flatten()[indices], dtype=torch.float32, requires_grad=True).to(device)
    v_selected = torch.tensor(v.flatten()[indices], dtype=torch.float32, requires_grad=True).to(device)
    p_selected = torch.tensor(p.flatten()[indices], dtype=torch.float32, requires_grad=True).to(device)
    return x_selected, y_selected, u_selected, v_selected, p_selected

def plot_uvp(x, y, u_out, v_out, p_out, u_true, v_true, p_true, filename):
    # Detach, move to CPU and convert to numpy arrays
    #x = x.detach().cpu().numpy()
    #y = y.detach().cpu().numpy()
    u_out = u_out.detach().cpu().numpy()
    v_out = v_out.detach().cpu().numpy()
    p_out = p_out.detach().cpu().numpy()
    u_true = u_true.detach().cpu().numpy()
    v_true = v_true.detach().cpu().numpy()
    p_true = p_true.detach().cpu().numpy()

    # Reshape to 2D grids for plotting
    x_grid = x.reshape(100, 100)
    y_grid = y.reshape(100, 100)
    u_out_grid = u_out.reshape(100, 100)
    v_out_grid = v_out.reshape(100, 100)
    p_out_grid = p_out.reshape(100, 100)
    u_true_grid = u_true.reshape(100, 100)
    v_true_grid = v_true.reshape(100, 100)
    p_true_grid = p_true.reshape(100, 100)

    # Calculate residuals
    u_residual = u_true_grid - u_out_grid
    v_residual = v_true_grid - v_out_grid
    p_residual = p_true_grid - p_out_grid

    # Create a 3x3 plot
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust spacing
    # Plot u_pred, u_true, and u_residual
    axes[0, 0].contourf(x_grid, y_grid, u_out_grid, levels=20, cmap='yuanshen')
    axes[0, 0].set_title('Predicted u')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')

    axes[0, 1].contourf(x_grid, y_grid, u_true_grid, levels=20, cmap='yuanshen')
    axes[0, 1].set_title('True u')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')

    u_contour = axes[0, 2].contourf(x_grid, y_grid, u_residual, levels=20, cmap='yuanshen')
    axes[0, 2].set_title('Residual u')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    fig.colorbar(u_contour, ax=axes[0, 2])

    # Plot v_pred, v_true, and v_residual
    axes[1, 0].contourf(x_grid, y_grid, v_out_grid, levels=20, cmap='yuanshen')
    axes[1, 0].set_title('Predicted v')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')

    axes[1, 1].contourf(x_grid, y_grid, v_true_grid, levels=20, cmap='yuanshen')
    axes[1, 1].set_title('True v')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')

    v_contour = axes[1, 2].contourf(x_grid, y_grid, v_residual, levels=20, cmap='yuanshen')
    axes[1, 2].set_title('Residual v')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    fig.colorbar(v_contour, ax=axes[1, 2])

    # Plot p_pred, p_true, and p_residual
    axes[2, 0].contourf(x_grid, y_grid, p_out_grid, levels=20, cmap='yuanshen')
    axes[2, 0].set_title('Predicted p')
    axes[2, 0].set_xlabel('x')
    axes[2, 0].set_ylabel('y')

    axes[2, 1].contourf(x_grid, y_grid, p_true_grid, levels=20, cmap='yuanshen')
    axes[2, 1].set_title('True p')
    axes[2, 1].set_xlabel('x')
    axes[2, 1].set_ylabel('y')

    p_contour = axes[2, 2].contourf(x_grid, y_grid, p_residual, levels=20, cmap='yuanshen')
    axes[2, 2].set_title('Residual p')
    axes[2, 2].set_xlabel('x')
    axes[2, 2].set_ylabel('y')
    fig.colorbar(p_contour, ax=axes[2, 2])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()