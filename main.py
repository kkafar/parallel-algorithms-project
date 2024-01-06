import numpy as np

# constants
bound = 1
gamma = 0.33
mass = 1
dt = 0.001


def run_simulation():
    X = 0
    V = 0
    t = 0
    while True:
        V = V + (-gamma * V * dt + np.random.normal(loc=0.0, scale=np.sqrt(dt))) / mass
        X = X + V * dt
        t += dt
        if abs(X) > bound:
            return t


def job(simulations):
    results = []
    for _ in range(simulations):
        results.append(run_simulation())
    return sum(results) / len(results)


results = job(25)
print(results)
