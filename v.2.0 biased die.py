import numpy as np
from numpy import random as rng
from collections import Counter
import matplotlib.pyplot as plt

# Keeping original bias
ssp = [1, 2, 3, 4, 4, 4, 4, 4, 5, 6]

def epoch(n=600, decay_factor=0.9):
    """
    Runs a single epoch of predictions and tracks the success rate.

    Parameters:
    - n (int): Number of trials.
    - decay_factor (float): Factor for exponential decay in counts.

    Returns:
    - tr (int): Total correct predictions.
    - success_history (list): Success rates over time.
    """
    tr = 0
    asp = []  # History of observed numbers
    running_success = 0
    success_history = []

    # Initialize counts for EWMA
    counts = {i: 0.0 for i in range(1, 7)}

    # Warm-up period
    for _ in range(100):
        s = rng.choice(ssp)
        asp.append(s)
        counts[s] += 1

    # Main prediction loop
    for i in range(n):
        # Exponential decay on counts
        for num in counts:
            counts[num] *= decay_factor

        # Observe new number
        s = rng.choice(ssp)
        counts[s] += 1  # Update count with new observation

        # Calculate probabilities
        total = sum(counts.values())
        probs = {num: counts[num] / total for num in counts}

        # Predict the number with the highest probability
        a = max(probs.items(), key=lambda x: x[1])[0]

        # Check prediction
        if a == s:
            tr += 1
            running_success += 1

        asp.append(s)

        # Record running success rate every 10 iterations
        if (i + 1) % 10 == 0:
            success_history.append(running_success / (i + 1))

    return tr, success_history

# Run multiple experiments
n_experiments = 250
all_results = []
all_histories = []

for _ in range(n_experiments):
    result, history = epoch()
    all_results.append(result)
    all_histories.append(history)

rl = np.array(all_results)
print(f"Average success rate: {rl.mean()/600:.2%}")
print(f"Best run success rate: {rl.max()/600:.2%}")
print(f"Worst run success rate: {rl.min()/600:.2%}")
print(f"Standard deviation: {rl.std()/600:.2%}")

# Plotting
plt.figure(figsize=(12, 6))

# Plot all experiment histories
histories_array = np.array(all_histories)
mean_history = histories_array.mean(axis=0)
std_history = histories_array.std(axis=0)

x = np.arange(10, 601, 10)  # Every 10 steps
plt.plot(x, mean_history, 'b-', label='Average Success Rate', linewidth=2)
plt.fill_between(
    x,
    mean_history - std_history,
    mean_history + std_history,
    alpha=0.2,
    color='b',
    label='±1 Standard Deviation'
)

plt.axhline(y=0.5, color='r', linestyle='--', label='50% Success Rate')
plt.grid(True, alpha=0.3)
plt.title(f'Success Rate Over Time (Averaged over {n_experiments} Experiments)')
plt.xlabel('Number of Trials')
plt.ylabel('Success Rate')
plt.legend()
plt.tight_layout()
plt.show()
