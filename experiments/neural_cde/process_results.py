import numpy as np

if __name__ == "__main__":
    method = "rev"
    solvers = ["midpoint", "rk4", "dopri5"]
    for solver in solvers:
        print(f"Solver: {solver}")
        data = np.loadtxt(f"results/{method}/{solver}.txt", delimiter=",", skiprows=1)
        memory_usages = data[:, 0] / 1024
        losses = data[:, 1]
        print(f"Memory Usage: {np.mean(memory_usages)} +/- {np.std(memory_usages)}")
        print(f"Accuracy: {np.mean(losses)} +/- {np.std(losses)}")
