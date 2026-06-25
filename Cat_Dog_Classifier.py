import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset =ImageFolder(
    root="animals",
    transform=transform
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)
train_loader = DataLoader(
    train_dataset,
    batch_size= 64,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle= False
)

class AnimalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
# Features
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
# decision blocker
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*32*32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# initializing Tools
model = AnimalClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training phase
epochs = 3
for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(images)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

    total_train_loss = train_loss/len(train_loader)
    accuracy = correct/len(test_loader.dataset)

    print(f"Epoch {epoch}/{epochs} -> loss: {total_train_loss:.4f} | accuracy : {accuracy*100:.2f}%")

# VISUALIZATION

images, labels = next(iter(test_loader))
images = images.to(device)

model.eval()
with torch.no_grad():
    outputs = model(images)
    preds = outputs.argmax(dim=1)

for i in range(6):
    plt.subplot(2, 3, i + 1)

    img = images[i].cpu().permute(1, 2, 0)
    plt.imshow(img)

    plt.title(f"Pred: {dataset.classes[preds[i]]}")
    plt.axis("off")

plt.show()


torch.save(model.state_dict(), 'Cat_Dog_Classifier.pth')
print("Model saved as animal_classifier.pth")