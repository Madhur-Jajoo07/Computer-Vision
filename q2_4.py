import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_neurons, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_layer_neurons)
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_layer_neurons, output_size)

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

def initialize_weights(model, pretrained_weights):
    model.layer1.weight.data = torch.Tensor(pretrained_weights[0].T)
    model.layer1.bias.data = torch.Tensor(pretrained_weights[1])
    model.layer2.weight.data = torch.Tensor(pretrained_weights[2].T)
    model.layer2.bias.data = torch.Tensor(pretrained_weights[3])

def train(model, criterion, optimizer, train_loader, num_epochs=10):
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = Variable(inputs), Variable(targets)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

def evaluate(model, criterion, data_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

pretrained_weights = [
    torch.randn(128, 784),
    torch.randn(128),
    torch.randn(10, 128),
    torch.randn(10)
]

input_size = 784
hidden_layer_neurons = 128
output_size = 10
mlp_model = MLP(input_size, hidden_layer_neurons, output_size)

initialize_weights(mlp_model, pretrained_weights)

criterion = nn.MSELoss()
optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)

train(mlp_model, criterion, optimizer, train_loader, num_epochs=10)

train_loss, train_accuracy = evaluate(mlp_model, criterion, train_loader)
val_loss, val_accuracy = evaluate(mlp_model, criterion, test_loader)

print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
