import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
N_genes = 4  # Number of genes in the GRN (Pax6, Olig2, Nkx2.2, Irx3)
T_max = 100  # Total simulation time
dt = 0.05  # Reduced time step for finer resolution
positions = np.linspace(0, 10, 10)  # Increased spatial resolution for gradients

# Define morphogen gradients with cross-talk
def shh_gradient(t, x):
    return max(0, 1 - x / 10) * np.tanh(t / 20)

def bmp_gradient(t, x):
    return max(0, x / 10) * np.tanh(t / 20) * (1 - 0.1 * shh_gradient(t, x))  # Cross-talk effect

# Non-Markovian memory kernel
def memory_effect(past_states, decay_rate=0.1):
    return np.sum(past_states * np.exp(-decay_rate * np.arange(len(past_states))[::-1])) / (len(past_states) + 1e-6)

# Define GRN dynamics (Pax6, Olig2, Nkx2.2, Irx3)
def grn_dynamics(t, state, W, shh_input, bmp_input, past_states, decay_rate, forced_activation=None):
    gene_states = state[:-1]  # Last element is the energy used
    energy_used = state[-1]

    # Hill function for regulatory interactions (with overflow protection)
    hill = lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))

    # Add memory effect to gene states
    memory_contributions = np.array([memory_effect(past_states[i], decay_rate) for i in range(N_genes)])

    # Shh and BMP effects
    external_inputs = 0.1 * np.array([shh_input, -shh_input, bmp_input, -bmp_input])  # Balanced scaling

    # Update gene states based on regulatory interactions, external inputs, and memory
    d_gene_states = (
        np.dot(W, hill(gene_states)) + external_inputs + 0.1 * memory_contributions - 0.1 * gene_states
    )

    # Apply forced activation if specified
    if forced_activation is not None:
        for gene_name, activation_level in forced_activation.items():
            gene_index = {'Pax6': 0, 'Olig2': 1, 'Nkx2.2': 2, 'Irx3': 3}.get(gene_name)
            if gene_index is not None:
                d_gene_states[gene_index] = activation_level - gene_states[gene_index]  # Adjust dynamics

    # Normalize updates to prevent runaway changes
    d_gene_states /= (1 + np.abs(d_gene_states))

    # Add soft quadratic damping to stabilize dynamics
    gene_states_clipped = np.clip(gene_states, 0, 50)  # Limit to biologically plausible positive range
    d_gene_states -= 0.015 * gene_states_clipped**2  # Softer damping for better stability

    # Compute energy dissipation (scaled)
    energy_dissipated = np.sum(np.abs(d_gene_states)) / 20  # Adjusted scaling to improve efficiency

    # Ensure the output matches the shape of `state`
    return np.hstack((d_gene_states, energy_dissipated))

# Initialize gene states, energy consumption, and past states
initial_state = np.random.uniform(0.1, 1, N_genes + 1)  # Non-zero initial states for differentiation
past_states = [np.zeros(N_genes) for _ in range(10)]  # Keep track of the last 10 states

# Define GRN connectivity matrix (tweaked for sensitivity)
np.random.seed(42)
W = np.array([
    [-1, -0.6, 0.3, 0.7],  # Pax6 inhibits Olig2, Nkx2.2; activates Irx3
    [0.7, -1, 1.1, -0.1],  # Olig2 activates Nkx2.2; inhibits Pax6; reduced activation of Irx3
    [0.3, 1, -1, -0.6],    # Nkx2.2 inhibits Irx3 and Pax6; activated by Olig2
    [0.9, -0.1, 0.6, -0.5] # Irx3 activates Pax6 and Nkx2.2; inhibits itself
])

# Simulate for multiple dorsal-ventral positions with physiological and artificial scenarios
results = {}
knockouts = [None, 'Pax6', 'Olig2']  # Genes to knock out
forced_activations = [{'Shh': 50}, None]  # Artificially activate Shh in one scenario

for knockout in knockouts:
    for forced_activation in forced_activations:

        # Modify the GRN matrix for knockouts
        W_modified = W.copy()
        if knockout == 'Pax6':
            W_modified[0, :] = 0  # Remove Pax6 outgoing interactions
            W_modified[:, 0] = 0  # Remove Pax6 incoming interactions
        elif knockout == 'Olig2':
            W_modified[1, :] = 0  # Remove Olig2 outgoing interactions
            W_modified[:, 1] = 0  # Remove Olig2 incoming interactions

        for x in positions:
            time = np.arange(0, T_max, dt)
            gene_states = []
            energy_used = []
            current_state = initial_state

            for t in time:
                shh_input = shh_gradient(t, x)
                bmp_input = bmp_gradient(t, x)

                # Update GRN dynamics using solve_ivp for stability
                sol = solve_ivp(
                    grn_dynamics, [t, t + dt], current_state,
                    args=(W_modified, shh_input, bmp_input, past_states, 0.1, forced_activation),
                    method='RK45', vectorized=False, max_step=dt
                )
                next_state = sol.y[:, -1]

                # Save current gene states and energy dissipation
                gene_states.append(next_state[:-1])
                energy_used.append(next_state[-1])

                # Update past states for memory effects
                past_states.pop(0)
                past_states.append(next_state[:-1])

                # Update current state
                current_state = next_state

            gene_states = np.array(gene_states)
            energy_used = np.array(energy_used)
            results[(x, knockout, tuple(forced_activation.items()) if forced_activation else None)] = (gene_states, energy_used)

# Plot results for up to 5 scenarios for clarity
selected_scenarios = list(results.items())[:5]  # Limit to 5 scenarios
fig, axes = plt.subplots(len(selected_scenarios), 2, figsize=(16, 4 * len(selected_scenarios)))
axes = axes.ravel()  # Flatten the axes array for 1D indexing

for i, ((x, knockout, forced_activation), (gene_states, energy_used)) in enumerate(selected_scenarios):
    # Plot gene states
    ax1 = axes[2 * i]
    for j in range(N_genes):
        ax1.plot(time, gene_states[:, j], label=f"Gene {j+1}")
    title = f"Knockout: {knockout or 'None'}, Activation: {forced_activation or 'None'}, Pos: {x:.1f}"
    ax1.set_title(title)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Expression Level")
    ax1.legend()

    # Plot energy dissipation
    ax2 = axes[2 * i + 1]
    ax2.plot(time, energy_used, label="Energy Dissipation", color="red")
    ax2.set_title(title)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Cumulative Energy Dissipated")
    ax2.legend()

plt.tight_layout()
plt.show()

# Analyze attractor states by clustering gene expression at the final time
from sklearn.cluster import KMeans
all_final_states = []
for (x, knockout, forced_activation), (gene_states, _) in results.items():
    final_gene_states = gene_states[-int(T_max / 10):, :]  # Use last 10% of time for steady-state analysis
    all_final_states.append(final_gene_states)
all_final_states = np.vstack(all_final_states)

kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto").fit(all_final_states)

# Print identified attractor states
print("Identified Attractor States (Cluster Centers):")
print(kmeans.cluster_centers_)

# Calculate mutual information and efficiency
from sklearn.metrics import mutual_info_score

def calculate_mutual_information(gradient, gene_states):
    discrete_gradient = np.digitize(gradient, bins=np.linspace(-1, 1, 20))
    discrete_states = np.digitize(gene_states, bins=np.linspace(0, 10, 20))  # Adjusted range for positive values
    return mutual_info_score(discrete_gradient, discrete_states)

# Estimate mutual information between morphogen gradients and attractor states
mutual_infos = []
for (x, _, _), (gene_states, _) in results.items():
    morphogen_values = np.array([shh_gradient(t, x) for t in time])
    mutual_info = calculate_mutual_information(morphogen_values, gene_states[:, 0])  # Example with Gene 1
    mutual_infos.append(mutual_info)
    print(f"Mutual Information between Shh and Gene 1 at Position {x:.1f}: {mutual_info:.2f}")

# Print efficiency at each position
efficiencies = []
for (x, _, _), (_, energy_used) in results.items():
    efficiency = np.mean(mutual_infos) / energy_used[-1]
    efficiencies.append(efficiency)
    print(f"Efficiency (Information per Unit Energy) at Position {x:.1f}: {efficiency:.4f}")
