import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64*64*64, latent_dim)
        self.fc_var = nn.Linear(64*64*64, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 64*64*64)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # standard deviation
        eps = torch.randn_like(std) # draw a sample from standard normal distribution
        return mu + eps*std
    

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # flatten
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        z = self.decoder_input(z)
        z = z.view(z.size(0), 64, 64, 64)  # reshape
        return self.decoder(z), mu, logvar
    


class ImageFolderWithoutSubfolders(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_files = os.listdir(folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder, self.image_files[idx])
        image = Image.open(image_path).convert('L')  # convert image to grayscale
        if self.transform:
            image = self.transform(image)
        return image
    
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Create datasets
train_dataset = ImageFolderWithoutSubfolders('keras_png_slices_data\keras_png_slices_train', transform=transform)
valid_dataset = ImageFolderWithoutSubfolders('keras_png_slices_data\keras_png_slices_validate', transform=transform)
test_dataset = ImageFolderWithoutSubfolders('keras_png_slices_data\keras_png_slices_test', transform=transform)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def train(vae, dataloader, epochs=10):
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for i, x in enumerate(dataloader):
            x = x.view(-1, 1, 256, 256)
            x_reconstructed, mu, logvar = vae(x)
            loss = vae_loss(x, x_reconstructed, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")

def vae_loss(x, x_recon, mu, logvar):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

latent_dim = 2
vae = VAE(latent_dim)
train(vae, train_dataloader, epochs=20)

vae.eval()

latent_points = []
labels = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        x = data[0]
        x = x.view(-1, 1, 256, 256) 
        _, mu, _ = vae(x)
        latent_points.append(mu)
        labels.append(data[1])

# Now we can visualize the latent space
latent_points = torch.cat(latent_points, dim=0)
labels = torch.cat(labels, dim=0)
plt.figure(figsize=(10, 10))
plt.scatter(latent_points[:, 0], latent_points[:, 1], c=labels, cmap='tab10')
plt.colorbar()
plt.show()