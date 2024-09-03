# Re-plotting the Acceptance Probability vs. Distance to Goal

import numpy as np
import matplotlib.pyplot as plt

# Define the distance range
distances = np.linspace(0, 20, 100)

# Calculate the acceptance probabilities
acceptance_probabilities = np.minimum(1, np.exp(-0.1 * distances))

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(distances, acceptance_probabilities, label='Acceptance Probability', color='blue')
plt.title('Acceptance Probability vs. Distance')
plt.xlabel('Distance to Goal')
plt.ylabel('Acceptance Probability')
plt.grid(True)
plt.legend()
plt.show()
