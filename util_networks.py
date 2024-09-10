import timm
import torch
import torch.nn as nn
from kan import *
import torch.nn.functional as F
import torch.nn.functional as F

import torch
import torch.nn as nn
import timm

# Define the MLP class
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(input_size // 2, num_classes)
        )
    def forward(self, x):
        return self.mlp(x)

class KANvolver(nn.Module):
    def __init__(self, layers_hidden, polynomial_order=2, base_activation=nn.ReLU, use_feature_extractor=True, flat_features=None):
        super(KANvolver, self).__init__()
       #self.layers_hidden = layers.hidden
        self.use_feature_extractor = use_feature_extractor
        self.polynomial_order = polynomial_order
        self.base_activation = base_activation()
        if self.use_feature_extractor:
            # Feature extractor with Convolutional layers
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 3 input channels (RGB), 16 output channels
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            # Calculate the flattened feature size after convolutional layers
            if flat_features is None:
                flat_features = 32 * 56 * 56  # Adjusted for input size 224x224
            else:
                assert flat_features is not None, "flat_features must be provided if use_feature_extractor is False"
        self.layers_hidden = [flat_features] + layers_hidden
        self.base_weights = nn.ModuleList()
        self.poly_weights = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for in_features, out_features in zip(self.layers_hidden[:-1], self.layers_hidden[1:]):
            self.base_weights.append(nn.Linear(in_features, out_features))
            self.poly_weights.append(nn.Linear(in_features * (polynomial_order + 1), out_features))
            self.batch_norms.append(nn.BatchNorm1d(out_features))

    def compute_efficient_monomials(self, x, order):
        powers = torch.arange(order + 1, device=x.device, dtype=x.dtype)
        x_expanded = x.unsqueeze(-1).repeat(1, 1, order + 1)
        return torch.pow(x_expanded, powers)
       
    def forward(self, x):
        if self.use_feature_extractor:
            x = self.feature_extractor(x)
        for base_weight, poly_weight, batch_norm in zip(self.base_weights, self.poly_weights, self.batch_norms):
            x = x.view(x.size(0), -1)  # Flatten the features from the conv layers
            base_output = base_weight(x)
            monomial_basis = self.compute_efficient_monomials(x, self.polynomial_order)
            monomial_basis = monomial_basis.view(x.size(0), -1)
            poly_output = poly_weight(monomial_basis)
            x = self.base_activation(batch_norm(base_output + poly_output))
        
        return x
        
# Define the Image Classifier class
class ImageClassifier(nn.Module):
    def __init__(self, backbone_model_name, num_classes=13, pretrained=True, network_type='mlp'):
        super(ImageClassifier, self).__init__()
        self.network_type = network_type
       
       # Load the backbone model
        self.backbone = timm.create_model(backbone_model_name, pretrained=pretrained)
        self.backbone.head = nn.Identity()  # Remove the classification head
        # Calculate the input size of the MLP based on the backbone model's output
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224)
            backbone_output = self.backbone(sample_input)
            backbone_output_shape = backbone_output.shape[1]
       # Initialize the appropriate classifier
        if network_type == 'mlp':
            self.classifier = MLP(backbone_output_shape, num_classes)
        elif network_type == 'kanvolver_kan':
            self.kan = KANvolver(layers_hidden=[128, 64, num_classes], use_feature_extractor=True, flat_features=None)
        elif network_type == 'backbone_kan':
            self.kan = KANvolver(layers_hidden=[1024, 32, num_classes], use_feature_extractor=False, flat_features=backbone_output_shape)
        else:
            raise ValueError("Invalid network_type. Supported values are 'mlp', 'kanvolver_kan', and 'backbone_kan'.")
    def forward(self, x):
        if self.network_type == 'mlp':
            x = self.backbone(x)
            x = self.classifier(x)
        elif self.network_type == 'kanvolver_kan':
            x = self.kan(x)
        elif self.network_type == 'backbone_kan':
            x = self.backbone(x)
            x = self.kan(x)
        return x
        
# images.shape torch.Size([32, 3, 224, 224])
#  x's shape after backbone : torch.Size([32, 1000])
# above is x's shape after coming out from classifier
# torch.Size([32, 1000])
# Input shape: torch.Size([32, 1000])
# Instantiate the models
model_names = [
'vit_base_patch16_224',
'resnet50.a1_in1k',
'beit_base_patch16_224',
'vit_tiny_patch16_224',
'mobilenetv3_large_100.ra_in1k'
]

# Create a dictionary to hold the models
# models = {}
# for model_name in model_names:
#    print(model_name)
#    models[model_name] = ImageClassifier(model_name, num_classes=13, pretrained=True)
#    print(f"Instantiated mode : {model_name}")

# # Print the models to verify
# for name, model in models.items():
#    print(f"Model: {name}")
#    print(model)
#    print('-' * 80)

# # Instantiate the model with desired hidden layers configuration
# model = KANvolver(layers_hidden=[128, 64, 13])
# # Create a random tensor with shape (32, 3, 224, 224)
# input_tensor = torch.randn(32, 3, 224, 224)
# # Pass the input tensor through the model and print the shapes
# output = model(input_tensor)
# print(f'Output shape: {output.shape}')
