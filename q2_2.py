#%%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
# Hyperparameters
input_size = 784  # MNIST images are 28x28 pixels
hidden_size = 128
batch_size = 64
learning_rate = 0.01
num_epochs = 10
losses = []
# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define autoencoder model
encoder = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.Tanh()
)

decoder = nn.Sequential(
    nn.Linear(hidden_size, input_size),
    nn.Sigmoid()  # Using Sigmoid activation for the output layer to scale values between 0 and 1
)

# Loss function and optimizer
criterion = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# Training the autoencoder
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = Variable(img)

        # Forward pass
        encoded = encoder(img)
        decoded = decoder(encoded)
        loss = criterion(decoded, img)
        losses.append(loss.detach().numpy())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss at the end of each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Testing the autoencoder on a few samples
test_samples, _ = next(iter(train_loader))
test_samples = Variable(test_samples[:5])
encoded_samples = encoder(test_samples)
reconstructed_samples = decoder(encoded_samples)

# Plotting original and reconstructed samples
import matplotlib.pyplot as plt
#%%
plt.plot(losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()





def to_img(x):
    x = x.view(1, 28, 28)
    return x

fig, axes = plt.subplots(nrows=2, ncols=len(test_samples), figsize=(10, 4))

# Iterate over the images and plot them
for images, title, row in zip([test_samples, reconstructed_samples], ['Original', 'Reconstructed'], axes):
    for img, ax in zip(images, row):
        img = to_img(img)
        ax.imshow(img.view(28, 28).detach().numpy(), cmap='gray')
        ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
# %%
