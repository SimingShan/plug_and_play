from model import MLP_NS
from train_utils import train
from data_utils import prepare_data
from evaluation import model_testing
import torch
import os
def main():
    # Store configuration settings here
    device = 'mps' if torch.backends.mps.is_built() else 'cpu'  # Ensure compatibility with available devices
    grid_size = 100
    Re = 20
    num_epochs = 30000
    initial_lr = 1e-3

    # Prepare data
    x, y, u, v, p = prepare_data(grid_size, Re, device, 0.1)

    # Initialize model
    model = MLP_NS().to(device)

    # Train model
    train(model, x, y, u, v, p, Re, num_epochs, initial_lr)

    # Save model checkpoint
    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_checkpoint.pth'))
    print(f"Model parameters saved to {os.path.join(checkpoint_dir, 'model_checkpoint.pth')}")

    model.eval()
    model_testing(model, grid_size, Re, device)

if __name__ == '__main__':
    main()