import numpy as np
import matplotlib.pyplot as plt

# constants
bound = 1
gamma = 0.33
mass = 1
steps = 10000

t_start = 0
t_end = 5
dt = (t_end - t_start) / steps
timestamps = np.arange(t_start, t_end, dt)


def run_simulation():
    X = 0
    V = 0
    t = 0
    positions = [X]
    for i in range(0, steps - 1):
        V = V + (-gamma * V * dt + np.random.normal(loc=0.0, scale=np.sqrt(dt))) / mass
        X = X + V * dt
        t += dt
        positions.append(X)

        if abs(X) > bound:
            positions.extend([None] * (steps - i - 2))
            plt.plot(timestamps, positions)
            break

    return t


def job(simulations):
    results = []
    for _ in range(simulations):
        results.append(run_simulation())
    return sum(results) / len(results)


results = job(25)
print(results)
plt.show()
