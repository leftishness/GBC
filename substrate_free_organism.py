import numpy as np
from tqdm.auto import tqdm
import time
import matplotlib.pyplot as plt

class RiemannianManifold:
    """Step 1: State Manifold with Ricci Flow and Stabilizing Terms"""

    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.metric = np.eye(dimensions)
        self.previous_metric = np.eye(dimensions)
        self.christoffel_symbols = np.zeros((dimensions, dimensions, dimensions))
        self.ricci_tensor = np.zeros((dimensions, dimensions))

    def compute_christoffel_symbols(self):
        """Fast approximation of Christoffel symbols"""
        metric_inv = np.linalg.inv(self.metric + 1e-6 * np.eye(self.dimensions))

        # Compute approximate derivatives using metric evolution
        dg = np.zeros((self.dimensions, self.dimensions, self.dimensions))
        metric_diff = self.metric - self.previous_metric
        for k in range(self.dimensions):
            dg[:, :, k] = 0.1 * metric_diff

        # Compute symbols
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                for k in range(self.dimensions):
                    self.christoffel_symbols[i, j, k] = 0.5 * np.sum(
                        metric_inv[i, :] * (
                                dg[:, j, k] + dg[:, k, j] - dg[j, k, :]
                        )
                    )

    def compute_ricci_tensor(self):
        """Fast approximation of Ricci tensor"""
        self.compute_christoffel_symbols()
        self.ricci_tensor = np.zeros((self.dimensions, self.dimensions))
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                self.ricci_tensor[i, j] = np.sum(
                    self.christoffel_symbols[:, i, :] *
                    self.christoffel_symbols[:, j, :]
                )

    def compute_metric_stability(self):
        """Compute a normalized measure of metric stability"""
        # Compute relative change using element-wise comparisons
        diff = np.abs(self.metric - self.previous_metric)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Scale the stability measure to be more sensitive
        stability = 5.0 * mean_diff + 10.0 * max_diff

        # Add eigenvalue variation component
        current_eigenvals = np.linalg.eigvals(self.metric).real
        prev_eigenvals = np.linalg.eigvals(self.previous_metric).real
        eigenval_change = np.mean(np.abs(current_eigenvals - prev_eigenvals))

        # Combine measures with scaling
        total_stability = (stability + 2.0 * eigenval_change) / 3.0

        return np.clip(total_stability, 0, 1)

    def evolve_metric(self, dt=0.01, stabilization_factor=0.1):
        """Evolve metric using simplified Ricci flow with more dynamic updates."""
        # Store current metric for stability computation
        self.previous_metric = self.metric.copy()

        # Compute Ricci tensor
        self.compute_ricci_tensor()

        # Evolution with stabilization
        ricci_flow = -self.ricci_tensor
        stabilization = -stabilization_factor * (self.metric - np.eye(self.dimensions))

        # Allow larger updates and include random perturbations
        update = dt * (ricci_flow + stabilization)
        update += np.random.normal(0, 0.01, self.metric.shape)  # Random perturbation
        update = np.clip(update, -0.2, 0.2)  # Allow larger changes

        self.metric += update

        # Ensure metric remains symmetric and well-conditioned
        self.metric = 0.5 * (self.metric + self.metric.T)
        min_eigenval = np.min(np.linalg.eigvals(self.metric).real)
        if min_eigenval < 1e-6:
            self.metric += (1e-6 - min_eigenval) * np.eye(self.dimensions)

        # Compute and return stability
        return self.compute_metric_stability()


class AdaptiveMemoryKernel:
    """Step 2: Generalized Memory Kernel with Diagnostics"""

    def __init__(self, memory_depth, dimensions):
        self.memory_depth = memory_depth
        self.dimensions = dimensions
        self.memory_buffer = []
        self.memory_weights = np.exp(-np.arange(memory_depth) / memory_depth)
        self.diagnostic_metrics = {
            'memory_influence_strength': [],
            'effective_memory_depth': []
        }

    def update(self, current_state):
        self.memory_buffer.append(current_state.copy())
        if len(self.memory_buffer) > self.memory_depth:
            self.memory_buffer.pop(0)

        if len(self.memory_buffer) > 0:
            # Compute influence
            influence = np.mean([
                np.exp(-0.7 * i / len(self.memory_buffer)) *
                np.mean(np.exp(-np.abs(current_state - past) ** 1.2))
                for i, past in enumerate(self.memory_buffer)
            ])
            # Append the computed influence for each timestep
            self.diagnostic_metrics['memory_influence_strength'].append(influence)
            self.diagnostic_metrics['effective_memory_depth'].append(len(self.memory_buffer))

    def compute_memory_effect(self, current_state):
        if not self.memory_buffer:
            return np.zeros_like(current_state)

        total_effect = np.zeros_like(current_state)
        effective_memory = 0.0

        # Dynamic weighting based on temporal distance and similarity
        for i, past_state in enumerate(self.memory_buffer):
            # Temporal weight decays faster to avoid long-term saturation
            temporal_weight = np.exp(-0.7 * i / len(self.memory_buffer))

            # Similarity metric with added variability to prevent stagnation
            diff = current_state - past_state
            similarity = np.exp(-np.abs(diff) ** 1.2)

            # Add contribution
            contribution = temporal_weight * similarity * past_state
            total_effect += contribution
            effective_memory += np.mean(similarity)

        # Normalize the effect
        normalized_effect = total_effect / (effective_memory + 1e-6)

        # Store diagnostic information with capped scaling
        self.diagnostic_metrics['memory_influence_strength'].append(
            np.clip(effective_memory / len(self.memory_buffer), 0, 1)
        )

        return normalized_effect


class HierarchicalFeedback:
    """Step 3: Hierarchical Feedback with Normalization"""

    def __init__(self, n_micro, n_meso, n_macro):
        self.n_micro = n_micro
        self.n_meso = n_meso
        self.n_macro = n_macro
        self.feedback_strengths = np.ones(3)
        self.feedback_thresholds = np.array([0.1, 0.2, 0.3])
        self.normalization_factors = np.ones(3)

    def compute_normalized_feedback(self, micro_state, meso_state, macro_state):
        micro_feedback = self._compute_microscale_feedback(micro_state)
        meso_feedback = self._compute_mesoscale_feedback(meso_state)
        macro_feedback = self._compute_macroscale_feedback(macro_state)

        # Normalize feedbacks with a minimum threshold to avoid over-normalization
        micro_norm = np.max(np.abs(micro_feedback)) + 1e-2
        meso_norm = np.max(np.abs(meso_feedback)) + 1e-2
        macro_norm = np.max(np.abs(macro_feedback)) + 1e-2

        total_feedback = (
                self.feedback_strengths[0] * micro_feedback / micro_norm +
                self.feedback_strengths[1] * meso_feedback / meso_norm +
                self.feedback_strengths[2] * macro_feedback / macro_norm
        )

        # Add slight perturbations to prevent feedback stagnation
        perturbation = np.random.normal(0, 0.01, total_feedback.shape)
        total_feedback += perturbation

        return total_feedback

    def _compute_microscale_feedback(self, state):
        return np.where(np.abs(state) > self.feedback_thresholds[0],
                        state, np.zeros_like(state))

    def _compute_mesoscale_feedback(self, state):
        feedback = np.zeros(self.n_micro)
        segment_size = self.n_micro // self.n_meso
        for i in range(self.n_meso):
            if np.abs(state[i]) > self.feedback_thresholds[1]:
                feedback[i * segment_size:(i + 1) * segment_size] = state[i]
        return feedback

    def _compute_macroscale_feedback(self, state):
        feedback = np.zeros(self.n_micro)
        segment_size = self.n_micro // self.n_macro
        for i in range(self.n_macro):
            if np.abs(state[i]) > self.feedback_thresholds[2]:
                feedback[i * segment_size:(i + 1) * segment_size] = state[i]
        return feedback


class FreeEnergyFunctional:
    """Step 4: Free Energy Functional with Validation"""

    def __init__(self, n_micro, n_meso, n_macro):
        self.n_micro = n_micro
        self.n_meso = n_meso
        self.n_macro = n_macro
        self.validation_metrics = {
            'entropy_components': [],
            'energy_components': [],
            'constraint_violations': []
        }

    def compute_free_energy(self, micro_state, meso_state, macro_state, memory_kernel):
        S_micro = self.compute_entropy(micro_state) / self.n_micro
        S_meso = self.compute_entropy(meso_state) / self.n_meso
        S_macro = self.compute_entropy(macro_state) / self.n_macro

        E_micro = np.sum(micro_state ** 2) / self.n_micro
        E_meso = np.sum(meso_state ** 2) / self.n_meso
        E_macro = np.sum(macro_state ** 2) / self.n_macro

        memory_effect = memory_kernel.compute_memory_effect(micro_state)
        memory_term = np.sum(memory_effect * micro_state) / self.n_micro

        self.validation_metrics['entropy_components'].append([S_micro, S_meso, S_macro])
        self.validation_metrics['energy_components'].append([E_micro, E_meso, E_macro])

        F = (S_micro + 0.1 * E_micro +
             2 * S_meso + 0.2 * E_meso +
             4 * S_macro + 0.4 * E_macro +
             0.3 * memory_term)

        return np.clip(F, -10, 10)

    def compute_entropy(self, state):
        p = np.abs(state) + 1e-10
        p = p / np.sum(p)
        return -np.sum(p * np.log(p))


class CompleteOrganism:
    """Step 5: Full Evolution Framework with Stabilization"""

    def __init__(self, n_micro=50, n_meso=5, n_macro=2, memory_depth=10):
        self.n_micro = n_micro
        self.n_meso = n_meso
        self.n_macro = n_macro

        self.manifold = RiemannianManifold(n_micro)
        self.memory_kernel = AdaptiveMemoryKernel(memory_depth, n_micro)
        self.feedback = HierarchicalFeedback(n_micro, n_meso, n_macro)
        self.free_energy = FreeEnergyFunctional(n_micro, n_meso, n_macro)

        self.micro_state = np.random.randn(n_micro)
        self.meso_state = np.zeros(n_meso)
        self.macro_state = np.zeros(n_macro)

        self.dt = 0.01
        self.stability_factor = 0.1

    def evolve_step(self, step):
        """Evolve the organism's state, incorporating memory and feedback."""
        # Apply localized macro-state shock
        if step == 70:  # Example: Shock applied at step 70
            self.macro_state[0] += 5  # Add a large perturbation to the first macro-state

        # Update manifold geometry
        metric_stability = self.manifold.evolve_metric(self.dt, self.stability_factor)

        # Update memory and compute effect
        self.memory_kernel.update(self.micro_state)
        memory_effect = self.memory_kernel.compute_memory_effect(self.micro_state)
        memory_effect = np.clip(memory_effect, -1, 1)

        # Compute feedback
        feedback = self.feedback.compute_normalized_feedback(
            self.micro_state, self.meso_state, self.macro_state
        )
        feedback = np.clip(feedback, -1, 1)

        # Update states with added noise for variability
        self.micro_state = np.dot(self.manifold.metric,
                                  self.micro_state + memory_effect + feedback)
        self.micro_state += np.random.normal(0, 0.01, self.micro_state.shape)

        # Loosen normalization constraints
        norm = np.max(np.abs(self.micro_state))
        if norm > 1e-6:
            self.micro_state /= (norm + 0.1)  # Add slack to allow state evolution

        self.update_meso_state()
        self.update_macro_state()

        # Compute free energy
        free_energy = self.free_energy.compute_free_energy(
            self.micro_state, self.meso_state, self.macro_state, self.memory_kernel
        )

        # Track metrics for visualization
        diagnostics = {
            'micro_state': self.micro_state,
            'meso_state': self.meso_state,
            'macro_state': self.macro_state,
            'free_energy': self.free_energy.compute_free_energy(
                self.micro_state, self.meso_state, self.macro_state, self.memory_kernel
            ),
            'metric': self.manifold.metric,
            'metric_stability': self.manifold.compute_metric_stability(),
            'memory_diagnostics': {
                'memory_influence_strength': self.memory_kernel.diagnostic_metrics['memory_influence_strength'].copy(),
                'effective_memory_depth': self.memory_kernel.diagnostic_metrics['effective_memory_depth'].copy()
            }
        }
        return diagnostics

    def update_meso_state(self):
        segments = np.array_split(self.micro_state, self.n_meso)
        self.meso_state = np.array([np.mean(np.abs(seg)) for seg in segments])
        self.meso_state = np.clip(self.meso_state, -1, 1)

    def update_macro_state(self):
        segments = np.array_split(self.meso_state, self.n_macro)
        self.macro_state = np.array([np.mean(seg) for seg in segments])
        self.macro_state = np.clip(self.macro_state, -1, 1)

class LearningSystem:
    """Step 6: Learning and Applications with Stabilization"""

    def __init__(self, organism):
        self.organism = organism
        self.learning_rate = 0.01
        self.stability_threshold = 0.1

    def train(self, n_steps, adapt_rate=True, adapt_feedback=True, plot_interval=100):
        history = []
        final_metrics = {}  # To store summary metrics

        pbar = tqdm(total=n_steps, desc="Training System")
        try:
            for step in range(n_steps):
                # Pass the current step to evolve_step
                state = self.organism.evolve_step(step)
                history.append(state)

                # Update learning rate based on stability
                if adapt_rate and step > 0:
                    energy_change = abs(state['free_energy'] - history[-2]['free_energy'])
                    if energy_change > self.stability_threshold:
                        self.learning_rate *= 0.95
                    else:
                        self.learning_rate *= 1.05

                # Extract key metrics for progress bar
                free_energy = state['free_energy']
                pattern_strength = np.mean(np.abs(state['macro_state']))
                metric_stability = state['metric_stability']
                memory_influence = state['memory_diagnostics']['memory_influence_strength'][-1] \
                    if state['memory_diagnostics']['memory_influence_strength'] else 0.0

                # Update progress bar
                postfix = (
                    f"FE: {free_energy:.2e} | "
                    f"Pattern: {pattern_strength:.2f} | "
                    f"Stability: {metric_stability:.2f} | "
                    f"Memory: {memory_influence:.2f}"
                )
                pbar.set_postfix_str(postfix)
                pbar.update(1)

                # Generate visualizations at regular intervals
                if step % plot_interval == 0 and step > 0:
                    self.plot_diagnostics(history)

            # Compute final metrics from the last state in history
            last_state = history[-1]
            final_metrics = {
                'final_free_energy': last_state['free_energy'],
                'final_micro_state': last_state['micro_state'],
                'final_pattern_strength': np.mean(np.abs(last_state['macro_state'])),
                'final_memory_influence': last_state['memory_diagnostics']['memory_influence_strength'][-1]
                if last_state['memory_diagnostics']['memory_influence_strength'] else 0.0
            }

        except Exception as e:
            print(f"\nError at step {step}: {str(e)}")
        finally:
            pbar.close()

        return history, final_metrics

    def plot_diagnostics(self, history):
        """Generate visualizations for key metrics."""
        self.plot_free_energy(history)
        self.plot_metric_stability(history)
        self.plot_memory_influence(history)
        self.plot_states_over_time(history)
        self.plot_pattern_strength(history)

    def plot_free_energy(self, history):
        free_energy = [state['free_energy'] for state in history]
        plt.figure(figsize=(8, 5))
        plt.plot(free_energy, label='Free Energy', color='blue')
        plt.axhline(y=np.min(free_energy), color='red', linestyle='--', label='Minimum Achieved')
        plt.title('Free Energy Evolution')
        plt.xlabel('Time Steps')
        plt.ylabel('Free Energy')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_metric_stability(self, history):
        stability = [state['metric_stability'] for state in history]
        plt.figure(figsize=(8, 5))
        plt.plot(stability, label='Metric Stability', color='green')
        plt.title('Metric Stability Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Stability')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_memory_influence(self, history):
        memory_strength = [
            state['memory_diagnostics']['memory_influence_strength'][-1]
            if state['memory_diagnostics']['memory_influence_strength'] else 0
            for state in history
        ]

        plt.figure(figsize=(8, 5))
        plt.plot(memory_strength, label='Memory Influence', color='orange')
        plt.title('Memory Influence Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Influence Strength')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_states_over_time(self, history):
        micro_states = np.array([state['micro_state'] for state in history])
        plt.figure(figsize=(10, 6))
        plt.imshow(micro_states.T, aspect='auto', cmap='viridis', interpolation='none')
        plt.colorbar(label='Micro State Values')
        plt.title('Micro States Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Micro States')
        plt.show()

    def plot_pattern_strength(self, history):
        pattern_strength = [np.mean(np.abs(state['macro_state'])) for state in history]
        plt.figure(figsize=(8, 5))
        plt.plot(pattern_strength, label='Pattern Strength', color='purple')
        plt.title('Pattern Strength Evolution')
        plt.xlabel('Time Steps')
        plt.ylabel('Pattern Strength')
        plt.legend()
        plt.grid()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize organism with smaller dimensions for faster testing
    organism = CompleteOrganism(
        n_micro=100,
        n_meso=10,
        n_macro=3,
        memory_depth=20
    )

    # Create learning system
    learner = LearningSystem(organism)

    # Run simulation with learning
    print("Starting simulation...")
    history, final_metrics = learner.train(n_steps=101, adapt_rate=True, adapt_feedback=True)

    # Print final metrics summary
    print("\nFinal Metrics:")
    for metric_name, value in final_metrics.items():
        if isinstance(value, np.ndarray):
            print(f"{metric_name}: {np.array2string(value, precision=6)}")
        else:
            print(f"{metric_name}: {value:.6f}")

    # Print summary statistics
    print("\nFinal state summary:")
    print(f"Feedback strengths: {organism.feedback.feedback_strengths}")
    print(f"Final learning rate: {learner.learning_rate}")
    print(f"Number of stable patterns: {np.sum(np.abs(history[-1]['macro_state']) > 0.1)}")
