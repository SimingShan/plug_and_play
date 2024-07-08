from model import MLP_NS
import torch.nn as nn
import torch
def sum_layer_dims(model):
    total_input_dim = 0
    total_output_dim = 0
    layer_dims = []

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            total_input_dim += layer.in_features
            total_output_dim += layer.out_features
            layer_dims.append((layer.in_features, layer.out_features))
            print(f"Layer: {name}, Input Dim: {layer.in_features}, Output Dim: {layer.out_features}")

    total_dim = total_input_dim + total_output_dim
    return total_dim, layer_dims


# Define the simplified LoRA MLP model
class MLP_LoRA(nn.Module):
    def __init__(self, base_model, rank=4):
        super(MLP_LoRA, self).__init__()
        self.base_model = base_model
        self.rank = rank

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.lora_layers = nn.ModuleList()
        self.layer_indices = []  # to track where the linear layers are

        for name, layer in self.base_model.named_modules():
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                self.lora_layers.append(nn.Linear(in_features, rank, bias=False))
                self.lora_layers.append(nn.Linear(rank, out_features, bias=False))
                self.layer_indices.append(name)  # save the layer name or index

    def forward(self, x):
        lora_output = x
        linear_layer_index = 0

        def apply_lora_adjustments(x, lora_output, linear_layer_index):
            A = self.lora_layers[2 * linear_layer_index](lora_output)
            B = self.lora_layers[2 * linear_layer_index + 1](A)
            return x + B

        for name, layer in self.base_model.named_modules():
            if isinstance(layer, nn.Sequential):
                for inner_name, inner_layer in layer.named_children():
                    if isinstance(inner_layer, nn.Linear):
                        print(f"Processing layer {inner_name}")
                        # Original forward pass
                        x = inner_layer(x)

                        # LoRA adjustment
                        x = apply_lora_adjustments(x, lora_output, linear_layer_index)
                        lora_output = torch.tanh(self.lora_layers[2 * linear_layer_index](lora_output)) if linear_layer_index < len(self.lora_layers) // 2 - 1 else lora_output
                        linear_layer_index += 1
                    else:
                        x = inner_layer(x)
            else:
                if isinstance(layer, nn.Linear):
                    print(f"Processing layer {name}")
                    # Original forward pass
                    x = layer(x)

                    # LoRA adjustment
                    x = apply_lora_adjustments(x, lora_output, linear_layer_index)
                    lora_output = torch.tanh(self.lora_layers[2 * linear_layer_index](lora_output)) if linear_layer_index < len(self.lora_layers) // 2 - 1 else lora_output
                    linear_layer_index += 1

        return x
'''
class MLP_LoRA(nn.Module):
    def __init__(self, base_model, input_feature):
        super(MLP_LoRA, self).__init__()
        self.base_model = base_model
        self.input_feature = input_feature
        self.output_feature, self.dim_sep = sum_layer_dims(base_model)
        self.net = nn.Sequential(
            nn.Linear(self.input_feature, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, self.output_feature)
        )
    def forward(self, input_tensor):
        return self.net(input_tensor)
'''
class CombinedModel(nn.Module):
    def __init__(self, base_model, lora_model, Re):
        super(CombinedModel, self).__init__()
        self.Re = Re
        self.base_model = base_model
        self.lora_model = lora_model

        # Store layer dimensions
        _, self.layer_dims = sum_layer_dims(self.base_model)

    def add_on_base(self, lora_output):
        current_index = 0
        for (in_dim, out_dim), layer in zip(self.layer_dims, self.base_model.children()):
            if isinstance(layer, nn.Linear):
                # Extract LoRA parameters for this layer
                A_size = in_dim
                B_size = out_dim
                A = lora_output[current_index:current_index + A_size].view(A_size, 1)
                current_index += A_size
                B = lora_output[current_index:current_index + B_size].view(1, B_size)
                current_index += B_size
                alpha = lora_output[current_index]
                current_index += 1
                # Adjust layer weights
                with torch.no_grad():
                    adjusted_weight = layer.weight + alpha * torch.matmul(A, B)
                layer.weight.copy_(adjusted_weight)

    def forward(self, x):
        # Generate LoRA adjustments
        lora_input = torch.tensor([[self.Re]], dtype=torch.float32).to(x.device)
        lora_output = self.lora_model(lora_input)

        # Apply LoRA adjustments to the base model
        self.add_on_base(lora_output)

        # Forward pass through the adjusted base model
        return self.base_model(x)
