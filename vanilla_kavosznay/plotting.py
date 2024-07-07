from model import MLP_NS
from evaluation import model_testing, plot_uvp
from train_utils import forward
import torch

# Load the checkpoint
#device = 'mps' if torch.backends.mps.is_built() else 'cuda'
device = 'mps'
model = MLP_NS()
model.load_state_dict(torch.load("./checkpoint/model_checkpoint.pth", map_location=device))
# Move the model to the appropriate device
model.to(device)
model.eval()

# Print the device
model_device = next(model.parameters()).device
print(f"Model is loaded on device: {model_device}")
u_loss, v_loss, p_loss, f_loss, g_loss, continuity_loss = model_testing(model, grid_size=100, Re=20, device=device)
