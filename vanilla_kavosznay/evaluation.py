from data_utils import prepare_data
from train_utils import compute_loss, forward
import matplotlib.pyplot as plt

def model_testing(model,grid_size, Re, device):
    # prepare the whole dataset
    x, y, u, v, p = prepare_data(grid_size, Re, device, 1)
    # pass the whole grid to the model
    u_pred, v_pred, p_pred, f_pred, g_pred, continuity_pred = forward(model, x, y, Re)
    u_loss, v_loss, p_loss, f_loss, g_loss, continuity_loss = compute_loss(u_pred,
                                                                           v_pred, p_pred, f_pred, g_pred,
                                                                           continuity_pred, u, v, p)
    plot_uvp(x, y, u_pred, v_pred, p_pred, u, v, p, filename="Kovasznay")
    print(f"The U_loss on the test set is: {u_loss:.10f}")
    print(f"The V_loss on the test set is: {v_loss:.10f}")
    print(f"The P_loss on the test set is: {p_loss:.10f}")
    print(f"The sum of MSE loss on the test set is: {(u_loss + v_loss + p_loss):.10f}")
    return u_loss, v_loss, p_loss, f_loss, g_loss, continuity_loss

def plot_uvp(x, y, u_out, v_out, p_out, u_true, v_true, p_true, filename):
    # Detach, move to CPU and convert to numpy arrays
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
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
    axes[0, 0].contourf(x_grid, y_grid, u_out_grid, levels=20, cmap='viridis')
    axes[0, 0].set_title('Predicted u')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')

    axes[0, 1].contourf(x_grid, y_grid, u_true_grid, levels=20, cmap='viridis')
    axes[0, 1].set_title('True u')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')

    u_contour = axes[0, 2].contourf(x_grid, y_grid, u_residual, levels=20, cmap='viridis')
    axes[0, 2].set_title('Residual u')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    fig.colorbar(u_contour, ax=axes[0, 2])

    # Plot v_pred, v_true, and v_residual
    axes[1, 0].contourf(x_grid, y_grid, v_out_grid, levels=20, cmap='viridis')
    axes[1, 0].set_title('Predicted v')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')

    axes[1, 1].contourf(x_grid, y_grid, v_true_grid, levels=20, cmap='viridis')
    axes[1, 1].set_title('True v')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')

    v_contour = axes[1, 2].contourf(x_grid, y_grid, v_residual, levels=20, cmap='viridis')
    axes[1, 2].set_title('Residual v')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    fig.colorbar(v_contour, ax=axes[1, 2])

    # Plot p_pred, p_true, and p_residual
    axes[2, 0].contourf(x_grid, y_grid, p_out_grid, levels=20, cmap='viridis')
    axes[2, 0].set_title('Predicted p')
    axes[2, 0].set_xlabel('x')
    axes[2, 0].set_ylabel('y')

    axes[2, 1].contourf(x_grid, y_grid, p_true_grid, levels=20, cmap='viridis')
    axes[2, 1].set_title('True p')
    axes[2, 1].set_xlabel('x')
    axes[2, 1].set_ylabel('y')

    p_contour = axes[2, 2].contourf(x_grid, y_grid, p_residual, levels=20, cmap='viridis')
    axes[2, 2].set_title('Residual p')
    axes[2, 2].set_xlabel('x')
    axes[2, 2].set_ylabel('y')
    fig.colorbar(p_contour, ax=axes[2, 2])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()