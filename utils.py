import torch
# Create a random tensor A of size [32, 13]
A = torch.randn(32, 13)
# Apply torch.max along dimension 1
max_values, max_indices = torch.max(A.data, 1)
# Print the outputs and their shapes
print("A:\n", A)
print("A shape:", A.shape)
print("\nmax_values:\n", max_values)
print("max_values shape:", max_values.shape)
print("\nmax_indices:\n", max_indices)
print("max_indices shape:", max_indices.shape)