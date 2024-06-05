from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.optim import SGD
import os

print(f'cuda version:{torch.version.cuda}')
print(f'cudnn version:{torch.backends.cudnn.version()}')
os.environ["https_proxy"] = "http://127.0.0.1:1080"


model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name,ignore_mismatched_sizes=True)
processor = ViTImageProcessor.from_pretrained(model_name)
model.classifier = torch.nn.Linear(in_features=model.config.hidden_size, out_features=10, bias=True)


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean,std=processor.image_std)
    ])
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 128
num_epochs = 10
learning_rate = 0.001


trainset = datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size,persistent_workers=True,
                                        shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='../data', train=False,
                                    download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train():
    optimizer = SGD(model.parameters(),lr=learning_rate,momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total = 0
        correct = 0
        for _,(inputs,labels) in enumerate(trainloader):
            inputs,labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss/len(trainloader)
        train_accuracy = correct/total


        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_accuracy = correct / total
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, train_accuracy: {train_accuracy*100:.2f}%, test_accuracy: {test_accuracy*100:.2f}%")

        model.save_pretrained("../trained_model/vit_cifar10.pth")
        model.config.save_pretrained("../trained_model/vit_cifar10_config.json")

if __name__ == '__main__':
    train()