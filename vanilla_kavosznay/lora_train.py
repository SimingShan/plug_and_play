device = 'mps' if torch.backends.mps.is_built() else 'cuda'
base_model = MLP_NS(2, 64, 3)
base_model.load_state_dict(torch.load("./checkpoint/model_checkpoint_Reynolds20.pth", map_location=device))
# Move the model to the appropriate device
base_model.to(device)
# Define LoRA model
lora_model = MLP_LoRA(base_model, rank=4).to(device)

device = 'mps' if torch.backends.mps.is_built() else 'cuda'

grid_size = 100
Re = 100
num_epochs = 10000
initial_lr = 1e-3

# Prepare data
x, y, u, v, p = prepare_data(grid_size, Re, device, 0.1)

base_model = MLP_NS(2, 64, 3)
base_model.load_state_dict(torch.load("./checkpoint/model_checkpoint_Reynolds20.pth", map_location=device))
# Move the model to the appropriate device
base_model.to(device)
# Define LoRA model
lora_model = MLP_LoRA(base_model, rank=4).to(device)
# Ensure there are parameters in lora_layers
assert sum(p.numel() for p in lora_model.lora_layers.parameters()) > 0, "lora_layers has no parameters"
print(lora_model.lora_layers[0])
train(lora_model, x, y, u, v, p, Re, num_epochs, initial_lr, lora_model.lora_layers)
u_loss, v_loss, p_loss, f_loss, g_loss, continuity_loss = model_testing(lora_model, grid_size=100, Re=100, device=device)

save_path = "./lora_model_trained.pth"
torch.save(lora_model.state_dict(), save_path)
print("Model saved to", save_path)