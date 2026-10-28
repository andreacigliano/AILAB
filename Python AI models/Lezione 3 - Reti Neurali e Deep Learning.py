# Corso di Laboratorio di Intelligenza Artificiale
# Prof. Ing. Andrea Cigliano
# Esempio di una semplice rete neurale feed-forward usando NumPy
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inizializzazione dei pesi
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        # Propagazione in avanti
        self.z1 = np.dot(X, self.W1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        self.output = self.sigmoid(self.z2)
        return self.output

# Esempio di utilizzo
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input di esempio
output = nn.forward(X)
print("Output della rete:", output)

**********************************************************************************************

import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
      self.W1 = np.random.randn(input_size, hidden_size)
      self.W2= np.random.randn(hidden_size, output_size)

    def sigmoid(self, x):
      return 1 / (1 + np.exp(-x))

    def forward(self, X):
      self.z1 = np.dot(X, self.W1)
      self.a1 = self.sigmoid(self.z1)
      self.z2 = np.dot(self.a1, self.W2)
      self.output = self.sigmoid(self.z2)
      return self.output

nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
output = nn.forward(X)
print("Output della rete:", output)

**********************************************************************************************


# Corso di Laboratorio di Intelligenza Artificiale
# Prof. Ing. Andrea Cigliano
# Esempio di una CNN usando PyTorch per classificazione immagini:

import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Creazione del modello
model = ConvNet()


# Corso di Laboratorio di Intelligenza Artificiale
# Prof. Ing. Andrea Cigliano
# Esempio di una RNN semplice per sequenze

import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model = SimpleRNN(input_size=10, hidden_size=20, output_size=2)

# Corso di Laboratorio di Intelligenza Artificiale
# Prof. Ing. Andrea Cigliano
# Esempio di utilizzo
# Esempio di implementazione di un autoencoder
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Creazione del modello
autoencoder = Autoencoder()