import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from util_networks import ImageClassifier  
import time
from torchvision import transforms, datasets
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


start_time = time.time()

def precision_macro(predictions, labels):
    num_classes = len(np.unique(labels))
    precision = 0.0
    for i in range(num_classes):
        tp = (labels == i) & (predictions == i)
        tp_sum = tp.sum()
        predicted_positive = (predictions == i).sum()
        precision += tp_sum / predicted_positive if predicted_positive > 0 else 0.0
    precision /= num_classes
    return precision

def recall_macro(predictions, labels):
    num_classes = len(np.unique(labels))
    recall = 0.0
    for i in range(num_classes):
        tp = (labels == i) & (predictions == i)
        tp_sum = tp.sum()
        actual_positive = (labels == i).sum()
        recall += tp_sum / actual_positive if actual_positive > 0 else 0.0
    recall /= num_classes
    return recall

def f1_macro(predictions, labels):
    num_classes = len(np.unique(labels))
    f1 = 0.0
    for i in range(num_classes):
        tp = (labels == i) & (predictions == i)
        tp_sum = tp.sum()
        predicted_positive = (predictions == i).sum()
        actual_positive = (labels == i).sum()
        precision = tp_sum / predicted_positive if predicted_positive > 0 else 0.0
        recall = tp_sum / actual_positive if actual_positive > 0 else 0.0
        f1 += 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f1 /= num_classes
    return f1

def evaluate(model, dataloader, loss_fn):
    num_samples = 0
    total_loss = 0
    all_predictions = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            if torch.cuda.is_available():
                images = images.cuda(1)
                labels = labels.cuda(1)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * labels.size(0)  # Average loss per batch, fix this 
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted)
            all_labels.append(labels)
            num_samples += labels.size(0)

    # Calculate average validation loss and accuracy
    test_loss = total_loss / num_samples
    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    cm = confusion_matrix(all_labels, all_predictions,)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('/home/sanjotst/img_cls/resnet_kan_cm.png')
    test_accuracy = (all_predictions == all_labels).mean()

    # Using simple functions for macro-averaging
    test_precision = precision_macro(all_predictions, all_labels)
    test_recall = recall_macro(all_predictions, all_labels)
    test_f1 = f1_macro(all_predictions, all_labels)

    return test_loss, test_accuracy, test_precision, test_recall, test_f1

# Define image directory
image_dir = '/home/sanjotst/img_cls/datasets/paddy-doctor-diseases-small-augmented-65k-split/test'

# Define transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset using ImageFolder
dataset = datasets.ImageFolder(root=image_dir, transform=transform)

# Create DataLoader
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load model
model_path ='/home/sanjotst/img_cls/saved_models/20240605-045051/resnet50.a1_in1k_epoch_10_backbone_kan.pth'
# i need to extract model name from model path
# model name is obtained as : goto the last "/" in path, and select entire string after it which is vit_base_patch16_224_epoch_4_mlp.pth
# read from the end and take the part before _epoch which is vit_base_patch16_224 here
# Extract model name from model path
model_name = model_path.split('/')[-1]  # Get the filename from the path
backbone_model_name = model_name.rsplit('_', 4)[0]  # Remove the part after the last '_'
model = ImageClassifier(backbone_model_name, num_classes=13, pretrained=False, network_type='backbone_kan')
model.load_state_dict(torch.load(model_path))
model.cuda(1)

# Evaluate model
test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, nn.CrossEntropyLoss())

# Print evaluation results
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}, Test precision: {test_precision}, Test recall: {test_recall}, Test F1 score: {test_f1}')

# Save evaluation results to a text file
results_dir = './test_results'
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f'{os.path.basename(model_path)}_results.txt')
with open(results_file, 'w') as f:
    f.write(f'Test loss: {test_loss}\n')
    f.write(f'Test accuracy: {test_accuracy}\n')
    f.write(f'Test precision: {test_precision}\n')
    f.write(f'Test recall: {test_recall}\n')
    f.write(f'Test F1 score: {test_f1}\n')



end_time = time.time()
total_time = end_time - start_time
print(f'Total time taken: {total_time} seconds')
