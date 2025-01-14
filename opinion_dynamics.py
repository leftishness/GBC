import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_agents = 100  # Number of agents
T_steps = 200  # Number of time steps
interaction_prob = 0.05  # Probability of interaction between agents
memory_decay = 0.1  # Decay rate for memory effects
global_influence_strength = 0.1  # Strength of global environmental influence
heterogeneity_factor = 0.5  # Degree of variability in agent susceptibility to influence

# Initialize agents' opinions and properties
opinions = np.random.uniform(-1, 1, N_agents)  # Opinions initialized randomly between -1 and 1
memory = np.zeros((N_agents, 10))  # Memory of past 10 states for each agent
susceptibility = np.random.uniform(1 - heterogeneity_factor, 1 + heterogeneity_factor, N_agents)  # Agent heterogeneity


# Define global stimuli
def global_stimulus_1(t):
    return 0.5 * np.sin(2 * np.pi * t / 50)  # Periodic stimulus


def global_stimulus_2(t):
    return -0.3 * np.cos(2 * np.pi * t / 100)  # Conflicting periodic stimulus


# Interaction function
def interaction(opinion_a, opinion_b, memory_effect, feedback, susceptibility_a):
    # Influence function: weighted average with feedback, memory, and heterogeneity effects
    influence = susceptibility_a * (0.4 * (opinion_a + opinion_b) + 0.2 * memory_effect + 0.4 * feedback)
    return np.clip(influence, -1, 1)  # Ensure opinion stays in [-1, 1]


# Memory effect function
def memory_effect(past_states):
    return np.sum(past_states * np.exp(-memory_decay * np.arange(len(past_states))[::-1])) / (len(past_states) + 1e-6)


# Simulation loop
opinion_history = [opinions.copy()]  # Track opinions over time
for t in range(T_steps):
    new_opinions = opinions.copy()
    feedback = global_influence_strength * (global_stimulus_1(t) + global_stimulus_2(t))  # Combined global influence

    # Update each agent's opinion
    for i in range(N_agents):
        # Pairwise interactions
        for j in range(N_agents):
            if i != j and np.random.rand() < interaction_prob:
                mem_effect = memory_effect(memory[i])  # Memory effect for agent i
                new_opinions[i] = interaction(opinions[i], opinions[j], mem_effect, feedback, susceptibility[i])

        # Update memory
        memory[i] = np.roll(memory[i], -1)
        memory[i, -1] = opinions[i]

    opinions = new_opinions
    opinion_history.append(opinions.copy())

# Convert opinion history to numpy array
opinion_history = np.array(opinion_history)

# Visualization of opinion dynamics
plt.figure(figsize=(12, 6))
for i in range(N_agents):
    plt.plot(opinion_history[:, i], alpha=0.5, lw=0.5)
plt.title("Opinion Dynamics Over Time with Heterogeneity and Global Stimuli")
plt.xlabel("Time Steps")
plt.ylabel("Opinions")
plt.grid()
plt.show()

# Final opinion distribution
plt.figure(figsize=(6, 4))
plt.hist(opinions, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Opinion Distribution at Final Time Step")
plt.xlabel("Opinion")
plt.ylabel("Frequency")
plt.grid()
plt.show()
