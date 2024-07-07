from data_utils import prepare_data
from train_utils import compute_loss
def model_testing(model,grid_size, Re, device):
    # prepare the whole dataset
    x, y, u, v, p = prepare_data(grid_size, Re, device, 1)
    # pass the whole grid to the model
    u_loss, v_loss, p_loss, f_loss, g_loss, continuity_loss = compute_loss(model, x, y, u, v, p, Re)
    print(f"The U_loss on the test set is: {u_loss:.10f}")
    print(f"The V_loss on the test set is: {v_loss:.10f}")
    print(f"The P_loss on the test set is: {p_loss:.10f}")
    print(f"The sum of MSE loss on the test set is: {(u_loss + v_loss + p_loss):.10f}")
    return u_loss, v_loss, p_loss, f_loss, g_loss, continuity_loss
