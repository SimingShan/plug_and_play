import torch
import torch.nn as nn
import torch.optim as optim
import time
def ns_pde_loss(x, y, u, v, p, re):
    # Compute gradients
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # Navier-Stokes equations and continuity equation

    f = u * u_x + v * u_y + p_x - 1 / re * (u_xx + u_yy)
    g = u * v_x + v * v_y + p_y - 1 / re * (v_xx + v_yy)

    continuity = u_x + v_y

    return f, g, continuity

def compute_loss(model, x, y, u_target, v_target, p_target, re):
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    input_tensor = torch.cat((x, y), dim=1)
    output = model(input_tensor)
    u_pred, v_pred, p_pred = output[:, 0:1], output[:, 1:2], output[:, 2:3]
    u_pred = u_pred.view(-1,1)
    v_pred = v_pred.view(-1, 1)
    p_pred = p_pred.view(-1, 1)
    f_pred, g_pred, continuity_pred = ns_pde_loss(x, y, u_pred, v_pred, p_pred, re)

    mse = nn.MSELoss()
    u_loss = mse(u_pred, u_target.view(-1,1))
    v_loss = mse(v_pred, v_target.view(-1,1))
    p_loss = mse(p_pred, p_target.view(-1,1))
    null = torch.zeros((x.shape[0], 1)).to(output.device)
    f_loss = mse(f_pred, null)
    g_loss = mse(g_pred, null)
    continuity_loss = mse(continuity_pred, null)
    '''
    print(f'u loss {u_loss}')
    print(f'v loss {v_loss}')
    print(f'p loss {p_loss}')
    print(f'f loss {f_loss}')
    print(f'g loss {g_loss}')
    print(f'con loss {continuity_loss}')
    '''
    return u_loss , v_loss , p_loss , f_loss , g_loss , continuity_loss


def log_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            print(f'{name}: grad_mean={grad_mean:.6e}, grad_std={grad_std:.6e}')


def train(model, x, y, u_target, v_target, p_target, re, num_epochs, initial_lr):
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=50000, tolerance_grad=1e-9, tolerance_change=1e-9,
                                  history_size=100)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1000,
                                                     min_lr=1e-7)
    model.train()
    best_loss = float('inf')
    best_iter = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        optimizer.zero_grad()
        u_loss, v_loss, p_loss, f_loss, g_loss, continuity_loss = compute_loss(model, x, y, u_target, v_target, p_target, re)
        loss = u_loss + v_loss + p_loss + f_loss + g_loss + continuity_loss
        loss.backward()
        optimizer.step()

        end_time = time.time()
        epoch_time = end_time - start_time

        if epoch >= 1000:
            scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_iter = epoch

        if epoch - best_iter > 1000:
            print(f"Early stopping at epoch {epoch} with best loss {best_loss:.10f}")
            break

        if (epoch + 1) % 1000 == 0:
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch + 1}, Loss: {loss.item():.10f}, Current LR: {current_lr:.1e}, Training time for epoch: {epoch_time:.2f} seconds")

    def closure():
        optimizer_lbfgs.zero_grad()
        u_loss, v_loss, p_loss, f_loss, g_loss, continuity_loss = compute_loss(model, x, y, u_target, v_target, p_target, re)
        loss = u_loss + v_loss + p_loss + f_loss + g_loss + continuity_loss
        loss.backward()
        return loss

    print("Starting L-BFGS optimization...")
    for i in range(500):  # Adjust the number of iterations or condition based on your requirements
        start_time = time.time()
        loss = optimizer_lbfgs.step(closure)
        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"L-BFGS Iteration: {i + 1}, Loss: {loss.item():.10f}, Training time for iteration: {epoch_time:.2f} seconds")
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_iter = i

        if i - best_iter > 50:  # Adjust the early stopping condition as needed
            print(f"Early stopping at L-BFGS iteration {i + 1} with best loss {best_loss:.10f}")
            break


