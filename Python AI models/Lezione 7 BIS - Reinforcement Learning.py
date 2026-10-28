import numpy as np
import gym
import random
from collections import defaultdict
import matplotlib.pyplot as plt

"""
ESEMPIO 1: Q-Learning base
Implementazione dell'algoritmo Q-Learning per risolvere l'ambiente FrozenLake di OpenAI Gym
"""

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        # Inizializzazione dei parametri dell'agente
        self.q_table = np.zeros((n_states, n_actions))  # Tabella Q dei valori stato-azione
        self.lr = learning_rate                         # Tasso di apprendimento
        self.gamma = discount_factor                    # Fattore di sconto per ricompense future
        self.epsilon = epsilon                          # Parametro per l'esplorazione ε-greedy
        self.n_actions = n_actions                      # Numero di azioni possibili

    def choose_action(self, state):
        # Implementazione della strategia ε-greedy
        if random.uniform(0, 1) < self.epsilon:
            # Esplorazione: scelta casuale
            return random.randint(0, self.n_actions - 1)
        else:
            # Sfruttamento: scelta della migliore azione nota
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        # Aggiornamento della Q-table usando la formula di Q-Learning
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Formula Q-Learning: Q(s,a) = Q(s,a) + α[R + γ*max(Q(s',a')) - Q(s,a)]
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

def train_q_learning(env, episodes=1000):
    """
    Funzione per addestrare l'agente Q-Learning
    """
    # Inizializzazione dell'agente
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Scelta dell'azione
            action = agent.choose_action(state)
            
            # Esecuzione dell'azione nell'ambiente
            next_state, reward, done, _ = env.step(action)
            
            # Apprendimento dall'esperienza
            agent.learn(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state

        rewards_per_episode.append(total_reward)

    return agent, rewards_per_episode
	
def plot_training_results(rewards, title="Training Results"):
    """
    Funzione per visualizzare i risultati dell'addestramento
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel('Episodio')
    plt.ylabel('Ricompensa totale')
    plt.grid(True)
    plt.show()

# Esempio di utilizzo dei vari agenti
if __name__ == "__main__":
    # Creazione dell'ambiente
    env = gym.make('FrozenLake-v1')
    
    # Addestramento Q-Learning
    print("Addestramento Q-Learning...")
    q_agent, q_rewards = train_q_learning(env)
    plot_training_results(q_rewards, "Risultati Q-Learning")
    
	# Chiusura dell'ambiente
    env.close()


"""
ESEMPIO 2: SARSA (State-Action-Reward-State-Action)
Implementazione dell'algoritmo SARSA
"""

class SARSAAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, next_action):
        # Aggiornamento della Q-table usando la formula SARSA
        old_value = self.q_table[state, action]
        next_value = self.q_table[next_state, next_action]
        
        # Formula SARSA: Q(s,a) = Q(s,a) + α[R + γ*Q(s',a') - Q(s,a)]
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_value)
        self.q_table[state, action] = new_value

def train_sarsa(env, episodes=1000):
    """
    Funzione per addestrare l'agente SARSA
    """
    agent = SARSAAgent(env.observation_space.n, env.action_space.n)
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False

        while not done:
            # Esecuzione dell'azione nell'ambiente
            next_state, reward, done, _ = env.step(action)
            
            # Scelta della prossima azione
            next_action = agent.choose_action(next_state)
            
            # Apprendimento dall'esperienza
            agent.learn(state, action, reward, next_state, next_action)
            
            total_reward += reward
            state = next_state
            action = next_action

        rewards_per_episode.append(total_reward)

    return agent, rewards_per_episode

"""
ESEMPIO 3: Deep Q-Network (DQN) semplificato
Implementazione di base di DQN usando PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reti neurali principale e target
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 0.1

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Campionamento casuale dalla memoria
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Conversione in tensori
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Calcolo dei valori Q correnti
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Calcolo dei valori Q target
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calcolo della loss e ottimizzazione
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def plot_training_results(rewards, title="Training Results"):
    """
    Funzione per visualizzare i risultati dell'addestramento
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel('Episodio')
    plt.ylabel('Ricompensa totale')
    plt.grid(True)
    plt.show()

# Esempio di utilizzo dei vari agenti
if __name__ == "__main__":
    # Creazione dell'ambiente
    env = gym.make('FrozenLake-v1')
    
    # Addestramento Q-Learning
    print("Addestramento Q-Learning...")
    q_agent, q_rewards = train_q_learning(env)
    plot_training_results(q_rewards, "Risultati Q-Learning")
    
    # Addestramento SARSA
    print("Addestramento SARSA...")
    sarsa_agent, sarsa_rewards = train_sarsa(env)
    plot_training_results(sarsa_rewards, "Risultati SARSA")
    
    # Chiusura dell'ambiente
    env.close()