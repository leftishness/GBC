import numpy as np
import matplotlib.pyplot as plt

# Define parameters
n_v, n_d, n_j = 5, 3, 4  # Number of segments for V, D, J
T_max, dt = 100, 0.1  # Total simulation time and time step
beta = 1.0  # Segment bias parameter
rag_energy_cost = 0.05  # Energy cost for RAG1/2 activity
tdt_energy_cost = 0.02  # Energy cost for TdT nucleotide addition
feedback_threshold = 0.8  # Threshold for functional inhibition

# Initial biases (non-uniform chromatin accessibility)
v_bias = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
d_bias = np.array([0.4, 0.4, 0.2])
j_bias = np.array([0.3, 0.3, 0.2, 0.2])

# Normalize biases
v_bias /= v_bias.sum()
d_bias /= d_bias.sum()
j_bias /= j_bias.sum()


# Cytokine signal (dynamic input)
def cytokine_signal(t):
    return np.sin(t / 10) * 0.5 + 0.5  # Oscillating cytokine levels


# Transition probabilities (with bias, cytokine modulation)
def transition_probabilities(state, bias, cytokine_input, beta):
    n_segments = len(state)
    probabilities = np.zeros((n_segments, n_segments))
    for i in range(n_segments):
        for j in range(n_segments):
            probabilities[i, j] = np.exp(-beta * abs(i - j)) * (1 + cytokine_input) * bias[j]
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


# Entropy calculation
def calculate_entropy(state):
    state = np.clip(state, 1e-10, None)  # Avoid log(0)
    return -np.sum(state * np.log(state))


# Simulate V(D)J recombination with perturbations
def vdj_recombination_simulation(n_v, n_d, n_j, T_max, dt, beta, rag_energy_cost, tdt_energy_cost, feedback_threshold,
                                 perturbation=None):
    time = np.arange(0, T_max, dt)
    v_state = v_bias.copy()  # Start with initial biases
    d_state = d_bias.copy()
    j_state = j_bias.copy()
    energy_used = []
    functional_output = 0  # Track functional recombination products
    functional_outputs = []  # Track over time

    for t in time:
        cytokine_input = cytokine_signal(t)

        # Update V segment probabilities
        v_prob = transition_probabilities(v_state, v_bias, cytokine_input, beta)
        v_state = np.dot(v_prob, v_state)

        # Update D segment probabilities
        d_prob = transition_probabilities(d_state, d_bias, cytokine_input, beta)
        d_state = np.dot(d_prob, d_state)

        # Update J segment probabilities
        j_prob = transition_probabilities(j_state, j_bias, cytokine_input, beta)
        j_state = np.dot(j_prob, j_state)

        # Compute functional output probability
        functional_output = (v_state.max() * d_state.max() * j_state.max())
        functional_outputs.append(functional_output)

        # Apply feedback inhibition if functional product exceeds threshold
        if functional_output > feedback_threshold:
            v_state *= 0.5
            d_state *= 0.5
            j_state *= 0.5

        # Apply perturbations
        if perturbation == "RAG1/2":
            v_state[:] = 0
            d_state[:] = 0
            j_state[:] = 0
        elif perturbation == "TdT":
            v_state[:] = np.ones_like(v_state) / len(v_state)
            d_state[:] = np.ones_like(d_state) / len(d_state)
            j_state[:] = np.ones_like(j_state) / len(j_state)

        # Compute energy usage
        energy = rag_energy_cost * (np.sum(v_state) + np.sum(d_state) + np.sum(j_state))
        if perturbation != "TdT":
            energy += tdt_energy_cost * functional_output  # TdT adds random nucleotides
        energy_used.append(energy)

    # Calculate final entropy
    diversity_entropy = calculate_entropy(v_state) + calculate_entropy(d_state) + calculate_entropy(j_state)
    return time, energy_used, functional_outputs, diversity_entropy


# Run simulations under different scenarios
scenarios = ["None", "RAG1/2", "TdT"]
results = {}
for scenario in scenarios:
    results[scenario] = vdj_recombination_simulation(
        n_v, n_d, n_j, T_max, dt, beta, rag_energy_cost, tdt_energy_cost, feedback_threshold, perturbation=scenario
    )

# Plot combined energy usage
fig, ax = plt.subplots(figsize=(10, 6))
for scenario, (time, energy_used, _, _) in results.items():
    ax.plot(time, energy_used, label=f"{scenario} Perturbation")
ax.set_title("Energy Usage Across Scenarios")
ax.set_xlabel("Time")
ax.set_ylabel("Energy Dissipated")
ax.legend()
plt.show()

# Plot functional output over time
fig, ax = plt.subplots(figsize=(10, 6))
for scenario, (time, _, functional_outputs, _) in results.items():
    ax.plot(time, functional_outputs, label=f"{scenario} Perturbation")
ax.set_title("Functional Output Across Scenarios")
ax.set_xlabel("Time")
ax.set_ylabel("Functional Output")
ax.legend()
plt.show()

# Summarize diversity and efficiency
efficiency_results = []
for scenario, (_, energy_used, _, diversity_entropy) in results.items():
    total_energy = np.sum(energy_used)
    efficiency = diversity_entropy / total_energy if total_energy > 0 else 0
    efficiency_results.append((scenario, diversity_entropy, total_energy, efficiency))

# Display summary as a bar chart
scenarios, diversities, energies, efficiencies = zip(*efficiency_results)
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2
x = np.arange(len(scenarios))
ax.bar(x - width, diversities, width, label="Diversity (Entropy)")
ax.bar(x, energies, width, label="Energy Used")
ax.bar(x + width, efficiencies, width, label="Efficiency")
ax.set_title("Diversity, Energy Usage, and Efficiency Across Scenarios")
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.set_ylabel("Metric Value")
ax.legend()
plt.show()
