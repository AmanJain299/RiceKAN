import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from torchvision import transforms, datasets
from util_networks import ImageClassifier  
import torch.nn as nn
import time
from tqdm import tqdm
from scipy.stats import mode

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

def evaluate_ensemble(models, dataloader, loss_fn):
    num_samples = 0
    total_loss = 0
    all_predictions = {}
    for model in models:
        model.eval()
        model_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                if torch.cuda.is_available():
                    images = images.cuda(2)
                    labels = labels.cuda(2)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                # print(f" labels.shape is {labels.shape}") # given : torch.Size([32]), torch.Size([8]) at the last batch, doubt : what is the datatype of labels
                # print(f" labels.size(0) is {labels.size(0)}") # given : labels.size(0) is 32, same pattern for the last batch, same about as above for labels.size(0)
                # print(loss.item()) # given :  0.0006303921109065413, doubt : what is the datatype of loss.item(), what does .item() do
                # print(loss.dtype) # given : torch.float32
                # print(loss.shape) # given : torch.Size([])
                total_loss += loss.item() * labels.size(0)  # total loss per batch
                # print(total_loss) # given : 7872.097711163238
                _, predicted = torch.max(outputs.data, 1) # predicted is a tensor of indices of class, shape = torch.Size([32])
                print(predicted.shape)
                print(labels.shape)
                model_predictions.append(predicted.cpu().numpy())
                # print(labels.shape) # given : torch.Size([32])
                all_labels.append(labels.cpu().numpy())
                num_samples += labels.size(0)
                # print(num_samples) # given : 65000, this is the total number of samples
        
        # all_predictions.append(model_predictions) # doubt : will the number of model_predictions also be 65k, no it will be list where each item torch.size([32]) tensor, length = 65k//32
        # print(len(all_predictions)) # given : 5, the 2nd dimension will be lists which contain 32 tensors
        model_predictions = np.concatenate(model_predictions)
        all_predictions[model] = model_predictions
    
    # Calculate average validation loss
    test_loss = total_loss / num_samples

    # Combine predictions from all models
    ensemble_predictions = np.array(list(all_predictions.values()))
    print(ensemble_predictions.shape) # (5, 65000) (5, 13000)
    final_predictions, _ = mode(ensemble_predictions, axis=0)
    print(final_predictions.shape) # (1, 65000)
    all_labels = np.concatenate(all_labels) 

    # # Combine predictions from all models
    # ensemble_predictions = np.sum(all_predictions, axis=0)
    # print(ensemble_predictions.shape)
    # # Perform majority voting
    # final_predictions = np.argmax(ensemble_predictions, axis=1)
    # print(final_predictions.shape)
    # # Convert predictions and labels to numpy arrays
    # all_predictions = torch.concatenate(all_predictions)
    # print(all_predictions.shape)
    # all_labels = torch.cat(all_labels).cpu().numpy()

    # Calculate evaluation metrics
    test_accuracy = (final_predictions == all_labels).mean()
    test_precision = precision_macro(final_predictions, all_labels)
    test_recall = recall_macro(final_predictions, all_labels)
    test_f1 = f1_macro(final_predictions, all_labels)

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
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load multiple models for ensemble
ensemble_models = []
model_paths = [
    '/home/sanjotst/img_cls/saved_models/20240527-064844/vit_base_patch16_224_epoch_10_mlp.pth',
    '/home/sanjotst/img_cls/saved_models/20240527-084230/resnet50.a1_in1k_epoch_10_mlp.pth',
    '/home/sanjotst/img_cls/saved_models/20240527-092031/beit_base_patch16_224_epoch_10_mlp.pth',
    '/home/sanjotst/img_cls/saved_models/20240527-111200/vit_tiny_patch16_224_epoch_10_mlp.pth',
    '/home/sanjotst/img_cls/saved_models/20240527-113435/mobilenetv3_large_100.ra_in1k_epoch_10_mlp.pth',

]

for model_path in model_paths:
    # Extract model name from model path
    model_name = model_path.split('/')[-1]  # Get the filename from the path
    backbone_model_name = model_name.rsplit('_', 3)[0]  # Remove the part after the last '_'
    model = ImageClassifier(backbone_model_name, num_classes=13, pretrained=False, network_type='mlp')#
    model.load_state_dict(torch.load(model_path))
    model.cuda(2)
    ensemble_models.append(model)

# Evaluate ensemble model
test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_ensemble(ensemble_models, test_loader, nn.CrossEntropyLoss())

# Print evaluation results
print(f'Ensemble Test results:')
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}, Test precision: {test_precision}, Test recall: {test_recall}, Test F1 score: {test_f1}')

# Save evaluation results to a text file
results_dir = './test_results'
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f'ensemble_{time.strftime("%Y%m%d-%H%M%S")}_results.txt')
with open(results_file, 'w') as f:
    f.write(f'Test loss: {test_loss}\n')
    f.write(f'Test accuracy: {test_accuracy}\n')
    f.write(f'Test precision: {test_precision}\n')
    f.write(f'Test recall: {test_recall}\n')
    f.write(f'Test F1 score: {test_f1}\n')
    f.write(f'Ensemble Model: \n')
    for model_path in model_paths:
        f.write(f'{model_path}\n')
