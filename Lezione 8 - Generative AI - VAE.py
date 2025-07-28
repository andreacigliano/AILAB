!pip install torch torchvision matplotlib numpy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

"""
ESEMPIO 1: Variational Autoencoder (VAE)
Implementazione di un VAE per la generazione di immagini
"""

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 400),  # Per dataset MNIST
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )
        
        # Layer per calcolare media e varianza dello spazio latente
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_var = nn.Linear(200, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid()  # Output normalizzato tra 0 e 1
        )
        
    def encode(self, x):
        # Codifica l'input nello spazio latente
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        # Trick di reparametrizzazione per permettere il backpropagation
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        # Decodifica dallo spazio latente all'output
        return self.decoder(z)
    
    def forward(self, x):
        # Forward pass completo
        mu, log_var = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    """
    Funzione di loss per VAE:
    Combina errore di ricostruzione (BCE) e KL divergence
    """
    BCE = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, 28*28), reduction='sum'
    )
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

"""
ESEMPIO 2: Generative Adversarial Network (GAN)
Implementazione di una GAN base per generazione di immagini
"""

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: latent_dim
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28*28),  # Output: immagine 28x28
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        flattened = img.view(-1, 28*28)
        return self.model(flattened)

def train_gan(generator, discriminator, dataloader, num_epochs=100, device="cuda"):
    """
    Funzione per addestrare una GAN
    """
    # Ottimizzatori
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    
    # Criterio di loss
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels per loss
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Training Discriminator
            d_optimizer.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            
            # Genera immagini fake
            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            # Loss totale discriminator
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Training Generator
            g_optimizer.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

"""
ESEMPIO 3: Conditional GAN (CGAN)
Implementazione di una GAN condizionale
"""

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(ConditionalGenerator, self).__init__()
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
        
    def forward(self, z, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat([z, label_embedding], 1)
        return self.model(x).view(-1, 1, 28, 28)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(ConditionalDiscriminator, self).__init__()
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(28*28 + num_classes, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img, labels):
        img_flat = img.view(-1, 28*28)
        label_embedding = self.label_embedding(labels)
        x = torch.cat([img_flat, label_embedding], 1)
        return self.model(x)

def sample_images(generator, n_row=10, latent_dim=100):
    """
    Funzione per generare e visualizzare immagini campione
    """
    z = torch.randn(n_row**2, latent_dim).to(generator.device)
    generated_imgs = generator(z)
    generated_imgs = generated_imgs.detach().cpu()
    
    # Visualizzazione
    fig, axs = plt.subplots(n_row, n_row, figsize=(10, 10))
    for i in range(n_row):
        for j in range(n_row):
            axs[i, j].imshow(generated_imgs[i*n_row + j].squeeze(), cmap='gray')
            axs[i, j].axis('off')
    plt.show()

# Esempio di utilizzo
if __name__ == "__main__":
    # Configurazione device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Caricamento dataset MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    dataloader = DataLoader(mnist, batch_size=64, shuffle=True)
    
    # Esempio VAE
    print("Training VAE...")
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters())
    
    # Esempio GAN
    print("\nTraining GAN...")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    train_gan(generator, discriminator, dataloader)
    
    # Generazione di immagini di esempio
    print("\nGenerazione immagini...")
    sample_images(generator)