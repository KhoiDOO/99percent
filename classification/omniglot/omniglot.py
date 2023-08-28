import torch
from torch import nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import Omniglot
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
            # 1 x 105 x 105
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 52 x 52
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 26 x 26
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 13 x 13
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  
            # 64 x 7 x 7
        )

        self.fc1 = nn.Linear(7 * 7 * 64, num_classes)


    def forward(self, x):
        x = self.layer1(x)        
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = Omniglot(root='./data', background=True, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

model = CNN(30)
optimizer = torch.optim.Adam(model.parameters(), lr)
criterion = torch.nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, len(train_loader)*epochs)

for epoch in range(epochs):
    train_loss = []
    train_accuracy = []
    for i, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        train_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
    print('epoch: {}, train loss: {}, train accuracy: {}'.format(epoch, np.mean(train_loss), np.mean(train_accuracy)))