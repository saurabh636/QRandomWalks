#Copyright Saurabh Srivastava, 2024
#License LGPL
#Email: saurabh636@yahoo.com

import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
import os

# Maze generation using DFS
def generate_maze(size):
    maze = np.ones((size, size))
    stack = [(0, 0)]
    maze[0, 0] = 0
    
    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and maze[nx, ny] == 1:
                neighbors.append((nx, ny))
        
        if neighbors:
            nx, ny = random.choice(neighbors)
            maze[(x + nx) // 2, (y + ny) // 2] = 0
            maze[nx, ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()
    
    return maze

# Function to handle plotting
def setup_plot(maze, start, end):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(maze, cmap='binary')
    ax.scatter(start[1], start[0], color='green', s=100, label='Start')
    ax.scatter(end[1], end[0], color='blue', s=100, label='End')
    path_line, = ax.plot([], [], color='red', linewidth=2)
    plt.legend(loc="upper right")
    return fig, ax, path_line

# Function to save and create GIF
def save_and_create_gif(filenames, gif_name):
    if filenames:
        with imageio.get_writer(gif_name, mode='I', duration=0.1) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Remove files
        for filename in filenames:
            os.remove(filename)

        print(f"Animated GIF saved as '{gif_name}'.")
    else:
        print("No images were saved, no GIF created.")


from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit.library import QFT

# Quantum Walk stepping function
def quantum_walk_step(current_pos, possible_moves, end, temperature=None, num_qubits=4):
    # Ensure num_qubits is an integer
    num_qubits = int(num_qubits)
    
    # Create a quantum circuit with enough qubits for the walker
    qc = QuantumCircuit(num_qubits)
    
    # Initialize walker state in superposition
    qc.h(range(num_qubits))
    
    # Apply a phase oracle to amplify the correct position (simplified as an example)
    if current_pos == end:
        return current_pos, True

    # Use QFT as a placeholder for walk evolution
    qc.append(QFT(num_qubits), range(num_qubits))
    qc.barrier()
    qc.save_statevector()
    #simulator = AerSimulator()
    simulator = Aer.get_backend('aer_simulator_statevector')

    qct = transpile(qc, simulator)
    #print(qct)

    # Use AerSimulator for simulation
    result = simulator.run(qct).result()
    try:
        state = result.get_statevector(qc)
        print("Statevector:", state)
    except Exception as e:
        print("Error:", e)
        return None, 0.
    
    # Convert statevector to probabilities
    probs = np.abs(state) ** 2
    
    # Ensure the length of move_probs matches the length of possible_moves
    if len(probs) < len(possible_moves):
        raise ValueError("The length of the statevector probabilities is less than the number of possible moves.")
    
    move_probs = [probs[i] for i in range(len(possible_moves))]
    
    # Choose the next move based on the highest probability
    next_move_index = np.argmax(move_probs)
    next_pos = possible_moves[next_move_index]
    
    print(move_probs)
    return next_pos, move_probs[next_move_index]


# Metropolis-Hastings stepping function
def mh_step(current_pos, possible_moves, end, temperature=None):
    next_pos = random.choice(possible_moves)
    distance_current = np.linalg.norm(np.array(end) - np.array(current_pos))
    distance_next = np.linalg.norm(np.array(end) - np.array(next_pos))

    # Calculate acceptance probability using Metropolis-Hastings criteria
    acceptance_prob = min(1, (np.exp(-distance_next) / np.exp(-distance_current)))
    return next_pos, acceptance_prob
    #return current_pos, False

# MCMC stepping function
def mcmc_step(current_pos, possible_moves, end, temperature=1.0):
    next_pos = random.choice(possible_moves)
    acceptance_prob = min(1, np.exp(-0.1 * np.linalg.norm(np.array(end) - np.array(next_pos))))
    return next_pos, acceptance_prob
    #return current_pos, False

# Simulated Annealing stepping function
def sa_step(current_pos, possible_moves, end, temperature):
    next_pos = random.choice(possible_moves)
    distance_current = np.linalg.norm(np.array(end) - np.array(current_pos))
    distance_next = np.linalg.norm(np.array(end) - np.array(next_pos))
    acceptance_prob = min(1, np.exp((distance_current - distance_next) / temperature))
    
    return next_pos, acceptance_prob
    #return current_pos, False

# Generic maze solver using a given stepping function
def maze_solver(maze, start, end, nsteps, step_func, gif_name, method='RW', initial_temp=1.0):
    current_pos = start
    path = [current_pos]
    filenames = []
    fig, ax, path_line = setup_plot(maze, start, end)

    temperature = initial_temp
    cooling_rate = 1-1.0/nsteps

    for step in range(nsteps):
        if current_pos == end:
            break

        x, y = current_pos
        possible_moves = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                possible_moves.append((nx, ny))

        if not possible_moves:
            continue

        next_pos, acceptance_prob = step_func(current_pos, possible_moves, end, temperature)
        print(f"Step: {step}, possibleMoves: {possible_moves}, Acceptance: {acceptance_prob}")
        
        if random.random() < acceptance_prob:
            print(f"Next pos: {next_pos}")
            current_pos = next_pos
            path.append(current_pos)
            path_np = np.array(path)
            path_line.set_data(path_np[:, 1], path_np[:, 0])

            filename = f'{gif_name.split(".")[0]}_step_{step}.png'
            filenames.append(filename)
            plt.title(f"{method}, Step: {step}")
            plt.savefig(filename)
            temperature *= cooling_rate  # Cooling for simulated annealing
            print(f"path: {path}")

    plt.close()
    save_and_create_gif(filenames, gif_name)
    return path

# Main
size = 21  # Maze size
maze = generate_maze(size)
start = (0, 0)
end = (size - 1, size - 1)

#Classical random walks...
# Run MCMC solver
print("Running MCMC Maze Solver...")
#path_mcmc = maze_solver(maze, start, end, nsteps=50000, step_func=mcmc_step, gif_name='mcmc_maze_solution.gif', method='MCMC')

# Run Simulated Annealing solver
print("Running Simulated Annealing Maze Solver...")
path_sa = maze_solver(maze, start, end, nsteps=50000, step_func=sa_step, gif_name='sa_maze_solution.gif', method='SA', initial_temp=1.0)

# Run Metropolis solver
print("Running Metropolis-Hastings Maze Solver...")
#path_mh = maze_solver(maze, start, end, nsteps=50000, step_func=mh_step, gif_name='mh_maze_solution.gif', method='MH')

#Quantum walks...
# Run quantum walk solver
print("Running Quantum-Walk Maze Solver...")
#path_qw = maze_solver(maze, start, end, nsteps=50000, step_func=quantum_walk_step, gif_name='qw_maze_solution.gif', method='QW')
