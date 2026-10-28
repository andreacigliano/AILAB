import warnings
warnings.filterwarnings('ignore')

import numpy as np
import gym

# Creazione dell'ambiente FrozenLake
env = gym.make('FrozenLake-v1', is_slippery=False)  # Ambiente non scivoloso

# Inizializzazione della Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Parametri di apprendimento
learning_rate = 0.8
discount_factor = 0.95
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Numero di episodi di addestramento
num_episodes = 10000

# Ciclo di addestramento
for episode in range(num_episodes):
    # Reset dell'ambiente
    state = env.reset()

    # Inizializzazione di variabili per l'episodio
    done = False
    rewards_current_episode = 0

    # Ciclo di interazione con l'ambiente
    while not done:
        # Scelta dell'azione (ε-greedy)
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])  # Sfruttamento
        else:
            action = env.action_space.sample()  # Esplorazione

        # Esecuzione dell'azione e osservazione del nuovo stato e della ricompensa
        new_state, reward, done, info = env.step(action)

        # Aggiornamento della Q-table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]))

        # Aggiornamento dello stato e della ricompensa totale
        state = new_state
        rewards_current_episode += reward

    # Decadimento del tasso di esplorazione
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

# Stampa della Q-table finale
print("Q-table:")
print(q_table)

# Test dell'agente addestrato
state = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state, :])
    new_state, reward, done, info = env.step(action)
    state = new_state
    env.render()  # Visualizzazione dell'ambiente

print("Fine del test.")