import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        """
        Inizializza il Variational Autoencoder
        
        Parametri:
        input_dim: dimensione dell'input (28*28=784 per MNIST)
        hidden_dim: dimensione dello strato nascosto
        latent_dim: dimensione dello spazio latente
        """
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Layer per media e log varianza dello spazio latente
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Per normalizzare l'output tra 0 e 1
        )
        
    def reparameterize(self, mu, logvar):
        """
        Implementa il trucco della reparametrizzazione per permettere il backpropagation
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
        
    def forward(self, x):
        """
        Forward pass attraverso il VAE
        """
        # Flatten input
        x = x.view(-1, 784)
        
        # Encode
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        # Reparametrizzazione
        z = self.reparameterize(mu, logvar)
        
        # Decode
        return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    Calcola la loss del VAE:
    Loss = BCE (ricostruzione) + KLD (regolarizzazione)
    """
    # Binary Cross Entropy per la ricostruzione
    BCE = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, 784), reduction='sum'
    )
    
    # Kullback-Leibler Divergence per la regolarizzazione
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def train_epoch(model, train_loader, optimizer, device):
    """
    Addestra il modello per un'epoca
    """
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Calcolo loss
        loss = loss_function(recon_batch, data, mu, logvar)
        
        # Backward pass
        loss.backward()
        train_loss += loss.item()
        
        # Ottimizzazione
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item()/len(data):.4f}')
            
    return train_loss / len(train_loader.dataset)

def generate_samples(model, device, num_samples=25):
    """
    Genera nuove immagini campionando dallo spazio latente
    """
    with torch.no_grad():
        # Campiona punti dallo spazio latente
        sample = torch.randn(num_samples, 20).to(device)
        
        # Decodifica i punti in immagini
        sample = model.decoder(sample).cpu()
        
        # Visualizza le immagini generate
        fig, axes = plt.subplots(5, 5, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(sample[i].view(28, 28), cmap='gray')
            ax.axis('off')
        plt.show()

def plot_reconstructions(model, data_loader, device):
    """
    Mostra le ricostruzioni di alcune immagini di test
    """
    model.eval()
    with torch.no_grad():
        data = next(iter(data_loader))[0][:8].to(device)
        recon, _, _ = model(data)
        
        # Visualizza originali e ricostruzioni
        fig, axes = plt.subplots(2, 8, figsize=(15, 4))
        for i in range(8):
            # Immagine originale
            axes[0, i].imshow(data[i].cpu().view(28, 28), cmap='gray')
            axes[0, i].axis('off')
            # Ricostruzione
            axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].axis('off')
        plt.show()

def main():
    # Parametri
    batch_size = 128
    epochs = 10
    hidden_dim = 400
    latent_dim = 20
    learning_rate = 1e-3
    
    # Configurazione device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    # Preparazione dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Creazione modello
    model = VAE(
        input_dim=784,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    print("Inizio training...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        print(f'Epoca {epoch}/{epochs} \tLoss media: {loss:.4f}')
        
        # Ogni 10 epoche mostra alcuni risultati
        if epoch % 10 == 0:
            generate_samples(model, device)
            plot_reconstructions(model, test_loader, device)
    
    print("Training completato!")
    
    # Salva il modello
    torch.save(model.state_dict(), 'vae_model.pth')
    print("Modello salvato come 'vae_model.pth'")

if __name__ == '__main__':
    main()