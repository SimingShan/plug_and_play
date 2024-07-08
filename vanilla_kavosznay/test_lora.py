from lora_utils import MLP_LoRA, MLP_NS
import torch
from evaluation import model_testing
# Load pretrained model
device = 'mps' if torch.backends.mps.is_built() else 'cuda'
base_model = MLP_NS(2, 64, 3).to(device)
base_model.load_state_dict(torch.load("./checkpoint/model_checkpoint_Reynolds20.pth", map_location=device))
lora_model = MLP_LoRA(base_model).to(device)
lora_model.load_state_dict(torch.load("./lora_model_trained.pth"))
lora = lora_model.lora_layers
u_loss, v_loss, p_loss, f_loss, g_loss, continuity_loss = model_testing(lora_model.base_model, grid_size=100, Re=20, device=device)
u_loss, v_loss, p_loss, f_loss, g_loss, continuity_loss = model_testing(lora_model, grid_size=100, Re=100, device=device)
