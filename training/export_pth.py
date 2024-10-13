import torch
import numpy as np
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from PIL import Image

class DataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST("./data", train=True, download=True, transform=transform),
            batch_size=self.batch_size, shuffle=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST("./data", train=False, download=True, transform=transform),
            batch_size=self.batch_size, shuffle=False
        )
        return self.train_loader, self.val_loader

class BetterModel(nn.Module):
    def __init__(self, num_classes=10):
        super(BetterModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.01, num_epochs=20):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1}/{self.num_epochs} - Loss: {total_loss / (i + 1):.4f}")
            self.validate(epoch)

    def validate(self, epoch):
        self.model.eval()
        val_losses, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_losses += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch + 1} - Validation Loss: {val_losses / len(self.val_loader):.4f} - Accuracy: {correct / total:.4f}")

    def save_model(self, path='mnist_model.pth'):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

