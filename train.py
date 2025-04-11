import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Force CPU usage and optimize
device = torch.device('cpu')
torch.set_num_threads(4)  # Optimize for a 4-core CPU
print(f"Using device: {device} with {torch.get_num_threads()} threads")

# Define lightweight CNN model
class LightweightNet(nn.Module):
    def __init__(self):
        super(LightweightNet, self).__init__()
        # Reduced number of filters for CPU efficiency
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)  # Smaller fully connected layer
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Set up data loading for MNIST
def train():
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load data
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    
    # Use a smaller batch size for CPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2  # Parallel data loading
    )
    
    # Initialize model, optimizer, loss function
    model = LightweightNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    epochs = 3  # Reduced for faster training
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    
    # Evaluate on a test set
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    print(f'Test accuracy: {correct / len(test_loader.dataset):.4f}')
    
    # Save model to shared volume
    model_path = os.path.join('/models', 'mnist_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Create directory for model if it doesn't exist
    os.makedirs('/models', exist_ok=True)
    train()