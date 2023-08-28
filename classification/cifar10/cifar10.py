import torch
from torch import nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import numpy as np

batch_size = 64
epochs = 20
lr = 1e-03

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(4 * 4 * 48, num_classes)


    def forward(self, x):
        x = self.layer1(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x

transform = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

model = CNN(10)
optimizer = torch.optim.Adam(model.parameters(), lr)
criterion = torch.nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, len(train_loader)*epochs)

for epoch in range(epochs):
    train_loss = []
    for i, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # pass data through network
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss.append(loss.item())
    test_loss = []
    test_accuracy = []
    for i, (data, labels) in enumerate(test_loader):
        # pass data through network
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        test_loss.append(loss.item())
        test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
    print(
        'epoch: {}, train loss: {}, test loss: {}, test accuracy: {}'.\
            format(epoch, np.mean(train_loss), np.mean(test_loss), np.mean(test_accuracy))
    )