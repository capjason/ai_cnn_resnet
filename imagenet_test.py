import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import BabyNet
import os
import wandb
import random


print(f'cuda version:{torch.version.cuda}')
print(f'cudnn version:{torch.backends.cudnn.version()}')

# Hyperparameters
batch_size = 128
learning_rate = 0.001
num_epochs = 200
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)



# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                          persistent_workers=True,pin_memory=True,pin_memory_device=device_str)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                         persistent_workers=True,pin_memory=True,pin_memory_device=device_str)

def get_random_batch(data_loader):
    idx = random.randint(0,len(data_loader) - 1)
    for i, (inputs, targets) in enumerate(data_loader):
        if i == idx:
            return inputs, targets
    return next(iter(data_loader))

def test():
    model = BabyNet()
    model_path = "./trained_model/model.pth"
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    import matplotlib.pyplot as plt

    # Select a batch from the test dataset
    batch_inputs, batch_targets = get_random_batch(test_loader)
    batch_inputs = batch_inputs.to(device)

    # Predict classes for the batch
    with torch.no_grad():
        model.eval()
        batch_outputs = model(batch_inputs)
        _, batch_predicted = batch_outputs.max(1)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Plot the result
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        image = batch_inputs[i].cpu().numpy().transpose(1, 2, 0)
        label = batch_predicted[i].item()
        ax.imshow(image)
        ax.set_title(f"{classes[label]}/{classes[batch_targets[i]]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    


def train():
    wandb.init(project="imagenet")
    wandb.config = {
        "learning_rate":learning_rate,
        "epochs": num_epochs,
        "batch_size":batch_size
    }
    # Define the model, loss function, and optimizer
    model = BabyNet()
    model_path = "./trained_model/model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    wandb.watch(model)


    criterion = nn.CrossEntropyLoss    ()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-3)
    last_top_1_accuracy = 0


    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        training_top_1_accuracy = 0.0
        total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total += targets.size(0)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            training_top_1_accuracy += predicted.eq(targets).sum().item()

        avg_loss = running_loss / len(train_loader)
        training_top_1_accuracy /= total

        # Evaluation on the test dataset
        model.eval()
        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                total += targets.size(0)

                # Top-1 accuracy
                _, predicted = outputs.max(1)
                top1_correct += predicted.eq(targets).sum().item()

                # Top-5 accuracy
                _, top5_predicted = outputs.topk(5, 1, True, True)
                top5_correct += sum(targets[j] in top5_predicted[j] for j in range(targets.size(0)))

        top1_accuracy = top1_correct / total
        top5_accuracy = top5_correct / total
        if top1_accuracy > last_top_1_accuracy:
            last_top_1_accuracy = top1_accuracy
            torch.save(model.state_dict(),model_path)
            print(f'save model at epoch {epoch} with accuracy {top1_accuracy * 100:.2f}%')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Training accuracy:{training_top_1_accuracy* 100:.2f}%, Top-1 accuracy: {top1_accuracy*100:.2f}%, Top-5 accurary: {top5_accuracy*100:.2f}%')
        wandb.log({
            "train_loss":avg_loss,
            "train_accuracy": training_top_1_accuracy,
            "top-1 accuracy": top1_accuracy,
            "top-5 accuracy": top5_accuracy
        },step=epoch + 1)

    wandb.finish()
    print('Training finished')

if __name__ == '__main__':
    # train()
    test()