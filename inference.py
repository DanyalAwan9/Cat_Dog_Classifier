import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class AnimalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*32*32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnimalClassifier().to(device)
model.load_state_dict(torch.load('Cat_Dog_Classifier.pth', map_location=device))
model.eval()

def predict_image(image_path):
    """Predict if image is cat or dog"""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
    
    classes = ['Cat', 'Dog']
    return classes[pred]

if __name__ == "__main__":
    result = predict_image("test_image.jpg")
    print(f"Prediction: {result}")