import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch import nn
from model import BabyNet
import torch
import os

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test(net,test_dl,device):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    for i, data in enumerate(test_dl):
        inputs,labels = data[0].to(device),data[1].to(device)
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        print(f"loss:{loss}")
    net.train()


def main():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 128

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,persistent_workers=True,
                                            shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)
    # imshow(torchvision.utils.make_grid(images))    
    model_path = "./model.pth"
    net = BabyNet()
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    torch.save(net.state_dict(), model_path)
    test(net,testloader,device)

    print('Finished Training')


if __name__ == '__main__':
    main()