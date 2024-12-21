import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import urllib.error

# Define a simple model for this experiment
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_data(batch_size=64, num_workers=4, data_path='./data'):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    try:
        trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return trainloader, testloader
    except urllib.error.URLError:
        print("Failed to download CIFAR-10 dataset automatically.")
        print("Please download the CIFAR-10 dataset manually from http://www.cs.toronto.edu/~kriz/cifar.html")
        print("Extract the dataset and place it in the './data' directory.")
        raise SystemExit("Dataset download failed. Exiting.")

# Function to calculate sensitivity scores using gradients
def calculate_sensitivity(model, trainloader, device='cpu', num_batches=10):
    model.train()
    sensitivity = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            sensitivity[name] = torch.zeros(module.weight.size(), device=device)
    
    for inputs, _ in trainloader:
        inputs, _ = inputs.to(device), _.to(device)
        outputs = model(inputs)
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                sensitivity[name] += module.weight.grad.abs().sum(dim=tuple(range(1, module.weight.ndim)))
        
        if len(sensitivity[name]) > num_batches:
            break
            
    for name in sensitivity:
        sensitivity[name] /= len(sensitivity[name])
    
    model.zero_grad()
    model.eval()
    
    return sensitivity

# Quantizes a model based on sensitivity scores
def quantize_model(model, sensitivity, quant_threshold=0.15, device='cpu'):
    quantized_model = SimpleCNN().to(device)
    quantized_model.load_state_dict(model.state_dict())
    quantized_layers = {}

    for name, module in quantized_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            scores = sensitivity[name]
            mask = scores >= quant_threshold  # Create a mask based on the threshold
            
            if mask.any():
                quantized_layers[name] = mask

            # Create a new weight tensor with the same shape
            new_weight = module.weight.clone()
            # Apply mask for quantization
            new_weight = torch.where(mask.unsqueeze(-1).expand_as(new_weight), 
                                     torch.quantize_per_tensor(module.weight, scale=0.05, zero_point=2, dtype=torch.qint4), 
                                     new_weight)

            module.weight.data = new_weight

    return quantized_model, quantized_layers

# Training loop with dynamic quantization
def train_model(model, quantized_model, trainloader, testloader, optimizer, criterion, epochs=5, device='cpu'):
    model.train()
    quantized_model.train()
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Dynamic quantization update every epoch
        sensitivity = calculate_sensitivity(model, trainloader, device=device)
        quantized_model, quantized_layers = quantize_model(model, sensitivity, device=device)
        
        test_loss = evaluate_model(quantized_model, testloader, criterion, device=device)
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
    
    return model, quantized_model, train_losses, test_losses, quantized_layers

# Evaluation function
def evaluate_model(model, testloader, criterion, device='cpu'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return running_loss / len(testloader)

# Main execution
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        trainloader, testloader = load_data()
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        initial_model = model.state_dict()
        quantized_model = SimpleCNN().to(device)
        quantized_model.load_state_dict(initial_model)

        trained_model, quantized_model, train_losses, test_losses, quantized_layers = train_model(model, quantized_model, trainloader, testloader, optimizer, criterion, epochs=5, device=device)

        # Plotting sensitivity scores and quantized layers
        sensitivity_scores = calculate_sensitivity(trained_model, trainloader, device=device)
        layer_names = list(sensitivity_scores.keys())
        sensitivity_values = [sensitivity_scores[name].mean().item() for name in layer_names]
        is_quantized = [(layer in quantized_layers) for layer in layer_names]

        plt.figure(figsize=(10, 6))
        plt.bar(layer_names, sensitivity_values, color=['blue' if not q else 'orange' for q in is_quantized])
        plt.xticks(rotation='vertical')
        plt.xlabel('Layer Name')
        plt.ylabel('Average Sensitivity Score')
        plt.title('Layer Sensitivity Scores and Quantization Decisions')
        plt.legend(['Full Precision', 'Quantized'])
        plt.show()
    except Exception as e:
        print(f"An error occurred: {str(e)}")