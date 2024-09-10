import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from util_networks import ImageClassifier
from util_dataload import train_loader, validation_loader, train_dataset, validation_dataset

# Define evaluate function for validation and train loss
def evaluate(model, dataloader, loss_fn):
	num_correct = 0
	num_samples = 0
	total_loss = 0
	model.eval()

	with torch.no_grad():
		for images, labels in dataloader:
			if torch.cuda.is_available():
				images = images.cuda()
				labels = labels.cuda()

			outputs = model(images)
			loss = loss_fn(outputs, labels)
			total_loss += loss.item() * labels.size(0)  # Average loss per batch

			# Get predictions and update accuracy
			_, predicted = torch.max(outputs.data, 1) #explain this line of code in detail
			num_correct += (predicted == labels).sum().item()
			num_samples += labels.size(0)

   # Calculate average validation loss and accuracy
	val_loss = total_loss / num_samples
	accuracy = 100.0 * num_correct / num_samples

	return val_loss, accuracy

# Parse arguments
parser = argparse.ArgumentParser(description='Train models.')
parser.add_argument('--models', nargs='+', default=['vit_tiny_patch16_224'], help='Models to train (default: vit_base_patch16_224)')
parser.add_argument('--network', type=str, default='backbone_kan', help='Mode: mlp or kan (default: mlp)')
args = parser.parse_args()

# Define optimizer
base_learning_rate = 0.0001
# Define loss function
loss_fn = nn.CrossEntropyLoss()

# Define number of epochs
epochs = 10
print("fixed epochs")

# Define the directory to save models
save_dir = f"./saved_models/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

# Tensorboard Logging
from torch.utils.tensorboard import SummaryWriter
# change log_dir to save_dir/logs
log_dir = os.path.join(save_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


# Training loop for all models
for model_name in args.models:
	print(f'Training model: {model_name}')

	# Instantiate ImageClassifier model
	model = ImageClassifier(model_name, num_classes=13, pretrained=True, network_type=args.network)
	# Move model to GPU if available
	model.cuda()

	# Define optimizer for each model
	optimizer = optim.Adam(model.parameters(), lr=base_learning_rate)

	# Training loop
	for epoch in tqdm(range(epochs)):
		print(f'Epoch: {epoch + 1}')
		
	# Training
		model.train()
		for batch_idx, (images, labels) in tqdm(enumerate(train_loader)):
			if torch.cuda.is_available():
				images = images.cuda()
				labels = labels.cuda()

			optimizer.zero_grad()
			outputs = model(images)		
			loss = loss_fn(outputs, labels)
			loss.backward()
			optimizer.step()

			if (batch_idx + 1) % 100 == 0:
				print(f'Batch {batch_idx + 1}/{len(train_loader)}')
		# Validation
		val_loss, val_accuracy = evaluate(model, validation_loader, loss_fn)
		print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.2f}%')
		# Calculate training loss
		train_loss, train_accuracy = evaluate(model, train_loader, loss_fn)
		print(f'Training loss: {train_loss:.4f}, Training accuracy: {train_accuracy:.2f}%')

		# Save the model
		if (epoch + 1) % 10 == 0:
			model_filename = os.path.join(save_dir, f'{model_name}_epoch_{epoch + 1}_{args.network}.pth')
			torch.save(model.state_dict(), model_filename)
			print(f'Model saved as: {model_filename}')
		
		# Tensorboard logging
		writer.add_scalar(f'{model_name}/Validation_Loss', val_loss, epoch)
		writer.add_scalar(f'{model_name}/Validation_Accuracy', val_accuracy, epoch)
		writer.add_scalar(f'{model_name}/Training_Loss', train_loss, epoch)
		writer.add_scalar(f'{model_name}/Training_Accuracy', train_accuracy, epoch)
writer.close()

			