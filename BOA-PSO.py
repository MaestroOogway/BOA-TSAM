import math, random, os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# --- Definición del problema ---
class Problem:
    def __init__(self):
        self.n_clients = 12
        self.n_hubs = 5

        # Distancias cliente-hub (12 x 5)
        self.distancias = [
            [6, 9, 12, 7, 8],
            [5, 8, 6, 10, 9],
            [8, 7, 9, 6, 11],
            [7, 5, 6, 9, 10],
            [9, 6, 8, 12, 7],
            [6, 5, 9, 11, 8],
            [8, 7, 5, 6, 10],
            [7, 6, 9, 8, 6],
            [5, 9, 10, 7, 11],
            [9, 6, 7, 8, 9],
            [6, 10, 8, 7, 5],
            [8, 6, 9, 10, 6]
        ]

        # Costo fijo por abrir cada hub
        self.costs = [20, 25, 18, 22, 19]

        # Capacidad máxima por hub (total debe cubrir 12 clientes)
        self.capacidad = [3, 3, 3, 3, 3]  # total 15 ≥ 12

        # Distancia máxima tolerada
        self.D_max = 8




    def check(self, x):
        hubs = [0]*self.n_hubs
        for h in x: hubs[h] = 1
        conteo = [0]*self.n_hubs
        for c in range(self.n_clients):
            h = x[c]
            if self.distancias[c][h] > self.D_max:
                return False
            conteo[h] += 1
            if conteo[h] > self.capacidad[h]:
                return False
        return True

    def fit(self, x):
        hubs = [0]*self.n_hubs
        for h in x: hubs[h] = 1
        total = sum(self.distancias[c][x[c]] for c in range(self.n_clients))
        total += sum(self.costs[j] for j, active in enumerate(hubs) if active)
        return total

# --- BOA implementation ---
class BOA_Particle:
    def __init__(self, problem):
        self.p = problem
        self.dim = self.p.n_clients
        # sol factible inicial
        self.position = [random.randrange(self.p.n_hubs) for _ in range(self.dim)]
        while not self.p.check(self.position):
            self.position = [random.randrange(self.p.n_hubs) for _ in range(self.dim)]
        self.p_best = self.position.copy()

    def fitness(self, pos=None):
        return self.p.fit(pos or self.position)

    def update_p_best(self):
        if self.fitness() < self.p.fit(self.p_best):
            self.p_best = self.position.copy()

    def tracking_and_move(self, swarm):
        Fi = self.fitness()
        CP = [p for p in swarm if p.fitness() < Fi and p is not self]
        if not CP: return
        SP = random.choice(CP).position
        new_pos = []
        for k in range(self.dim):
            r = random.random(); I = random.choice([1,2])
            val = self.position[k] + (1-2*r)*(SP[k] - I*self.position[k])
            new_pos.append(max(0, min(self.p.n_hubs-1, round(val))))
        if self.p.check(new_pos) and self.fitness(new_pos) <= Fi:
            self.position = new_pos
            self.update_p_best()

    def chasing_to_catch(self, t):
        Fi = self.fitness()
        new_pos = []
        for k in range(self.dim):
            r = random.random()
            val = self.position[k] + ((1-2*r)/(1+t))*self.position[k]
            new_pos.append(max(0, min(self.p.n_hubs-1, round(val))))
        if self.p.check(new_pos) and self.fitness(new_pos) <= Fi:
            self.position = new_pos
            self.update_p_best()

class BOA:
    def __init__(self, n_particles, max_iter, problem):
        self.N, self.T, self.p = n_particles, max_iter, problem
        self.swarm = []
        self.global_best = None
        self.convergence = []

    def initialize(self):
        while len(self.swarm) < self.N:
            self.swarm.append(BOA_Particle(self.p))
        self.global_best = min(self.swarm, key=lambda p: p.fitness())
        self.convergence = [self.global_best.fitness()]

    def evolve(self):
        for t in range(1, self.T+1):
            for p in self.swarm: p.tracking_and_move(self.swarm)
            for p in self.swarm: p.chasing_to_catch(t)
            candidate = min(self.swarm, key=lambda p: p.fitness())
            if candidate.fitness() < self.global_best.fitness():
                self.global_best = candidate
            self.convergence.append(self.global_best.fitness())

    def solve(self):
        self.initialize()
        self.evolve()
        return self.global_best

# --- PSO implementation ---
class PSO_Particle:
    def __init__(self, problem):
        self.p = problem
        self.dim = self.p.n_clients
        self.position = [random.randrange(self.p.n_hubs) for _ in range(self.dim)]
        self.velocity = [0]*self.dim
        while not self.p.check(self.position):
            self.position = [random.randrange(self.p.n_hubs) for _ in range(self.dim)]
        self.p_best = self.position.copy()

    def fitness(self, pos=None):
        return self.p.fit(pos or self.position)

    def update_p_best(self):
        if self.fitness() < self.p.fit(self.p_best):
            self.p_best = self.position.copy()

    def move(self, g_best, theta, alpha, beta):
        for j in range(self.dim):
            v = (self.velocity[j]*theta +
                 alpha*random.random()*(g_best[j]-self.position[j]) +
                 beta*random.random()*(self.p_best[j]-self.position[j]))
            self.velocity[j] = v
            # mapa a dominio discreto
            idx = max(0, min(self.p.n_hubs-1, round(v)))
            self.position[j] = idx

class PSO:
    def __init__(self, n_particles, max_iter, problem, theta=0.7, alpha=2, beta=2):
        self.n_particles, self.max_iter = n_particles, max_iter
        self.theta, self.alpha, self.beta = theta, alpha, beta
        self.p = problem
        self.swarm = []
        self.g_best = None

    def initialize(self):
        while len(self.swarm) < self.n_particles:
            p = PSO_Particle(self.p)
            self.swarm.append(p)
        self.g_best = min(self.swarm, key=lambda p: p.fitness()).position.copy()

    def evolve(self):
        for _ in range(self.max_iter):
            for p in self.swarm:
                p.move(self.g_best, self.theta, self.alpha, self.beta)
                if not self.p.check(p.position):
                    continue
                p.update_p_best()
                if p.fitness(p.position) < self.p.fit(self.g_best):
                    self.g_best = p.position.copy()

    def solve(self):
        self.initialize()
        self.evolve()
        return self.g_best, self.p.fit(self.g_best)

# --- Funciones de ejecución y comparación ---
def run_boa(n_runs, n_particles, max_iter):
    prob = Problem()
    return [BOA(n_particles, max_iter, prob).solve().fitness() for _ in range(n_runs)]

def run_pso(n_runs, n_particles, max_iter):
    prob = Problem()
    return [PSO(n_particles, max_iter, prob).solve()[1] for _ in range(n_runs)]

if __name__ == '__main__':
    n_runs, n_particles, n_iter = 30, 20, 500
    res_boa = run_boa(n_runs, n_particles, n_iter)
    res_pso = run_pso(n_runs, n_particles, n_iter)

    print("Resultados BOA:", res_boa)
    print("Resultados PSO:", res_pso)

    # Test estadístico
    stat, p = mannwhitneyu(res_boa, res_pso, alternative='two-sided')
    print(f"Mann-Whitney U = {stat:.2f}, p-value = {p:.4f}")
    if p < 0.05:
        print("Diferencia significativa (α=0.05)")
    else:
        print("No hay diferencia significativa (α=0.05)")
