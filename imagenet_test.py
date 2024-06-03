import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import BabyNet
import os
print(f'cuda version:{torch.version.cuda}')
print(f'cudnn version:{torch.backends.cudnn.version()}')

# Hyperparameters
batch_size = 512
learning_rate = 0.001
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,persistent_workers=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,persistent_workers=True)

def train():
    # Define the model, loss function, and optimizer
    model = BabyNet()
    model_path = "./model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Evaluation on the test dataset
        model.eval()
        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                total += targets.size(0)

                # Top-1 accuracy
                _, predicted = outputs.max(1)
                top1_correct += predicted.eq(targets).sum().item()

                # Top-5 accuracy
                _, top5_predicted = outputs.topk(5, 1, True, True)
                top5_correct += sum(targets[j] in top5_predicted[j] for j in range(targets.size(0)))

        top1_accuracy = 1 - top1_correct / total
        top5_accuracy = 1 - top5_correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Top-1 Error: {top1_accuracy*100:.2f}%, Top-5 Error: {top5_accuracy*100:.2f}%')
    torch.save(model.state_dict(),model_path)
    print('Training finished')

if __name__ == '__main__':
    train()