import subprocess

# List of model arguments
model_arguments = [
    'vit_base_patch16_224',
    'resnet50.a1_in1k',
    'beit_base_patch16_224',
    'vit_tiny_patch16_224',
    'mobilenetv3_large_100.ra_in1k'
]

# Iterate over each model argument
for model_arg in model_arguments:
    # Construct the command to run training_loop.py with the current model argument
    command = f"python training_loop.py --models {model_arg}"
    
    # Run the command
    subprocess.run(command, shell=True)
