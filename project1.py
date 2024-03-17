import numpy as np
from scipy.optimize import differential_evolution

def objective_function(x, true_labels):
    return 100 * np.sum(x != true_labels) / len(x)

def bounds(L, U):
    return [(L[i], U[i]) for i in range(len(L))]

def CenDE_DOBL(MaxNFC, NP, Jr, N, L, U, F, CR, j_rand, true_labels):
    U = np.maximum(U, L)
    bounds_list = bounds(L, U)

    def cost_function(x):
        return objective_function(x, true_labels)

    result = differential_evolution(cost_function, bounds=bounds_list, maxiter=MaxNFC // NP, popsize=NP, strategy='randtobest1bin',
                                    tol=1e-5, recombination=CR, updating='immediate', disp=False)

    best_individual = result.x
    best_value = result.fun

    print("Final Objective Function Value:", best_value)
    print("Number of Function Evaluations:", result.nfev)

    return best_individual

D = 4
MaxNFC = 10000
NP = 10
Jr = 0.9
N = 3
L = np.random.uniform(0, 1, D)
U = np.random.uniform(0, 1, D)
F = 0.5
CR = 0.9
j_rand = 0.1
true_labels = np.array([1, 0, 1, 0])

best_solution = CenDE_DOBL(MaxNFC, NP, Jr, N, L, U, F, CR, j_rand, true_labels)
print("Best Solution:", best_solution)
