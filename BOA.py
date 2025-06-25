import math, random, os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 89 151 271  10 y 100 - 20 y 500 - 30 y 1000
class Problem:
    def __init__(self):
        self.n_clients = 24
        self.n_hubs = 8

        # Matriz de distancias cliente-hub (24 x 8)
        self.distancias = [
            [6, 8, 12, 7, 9, 6, 10, 11],
            [5, 9, 6, 10, 8, 7, 9, 12],
            [8, 7, 10, 6, 11, 9, 8, 10],
            [7, 6, 8, 9, 10, 6, 7, 9],
            [9, 6, 9, 12, 7, 10, 11, 8],
            [6, 5, 10, 11, 9, 8, 6, 7],
            [8, 7, 6, 6, 10, 8, 9, 9],
            [7, 6, 10, 8, 6, 9, 11, 10],
            [5, 9, 11, 7, 11, 8, 10, 12],
            [9, 6, 7, 8, 9, 6, 10, 8],
            [6, 10, 8, 7, 5, 9, 6, 8],
            [8, 6, 9, 10, 6, 8, 10, 7],
            [7, 9, 10, 8, 11, 9, 7, 8],
            [6, 7, 8, 9, 10, 8, 9, 11],
            [9, 8, 6, 10, 7, 6, 9, 10],
            [7, 6, 9, 8, 10, 7, 6, 9],
            [8, 9, 10, 6, 11, 10, 8, 7],
            [9, 7, 6, 9, 10, 8, 7, 6],
            [6, 8, 7, 10, 9, 7, 8, 6],
            [8, 9, 10, 8, 9, 10, 9, 7],
            [7, 6, 8, 9, 7, 9, 10, 8],
            [5, 9, 11, 10, 9, 7, 8, 6],
            [6, 8, 7, 9, 6, 10, 11, 9],
            [8, 7, 9, 8, 10, 8, 7, 6],
        ]

        # Costo fijo por abrir cada hub
        self.costs = [22, 25, 18, 20, 19, 24, 21, 23]

        # Capacidades: total 30 ≥ 24
        self.capacidad = [4, 4, 4, 4, 4, 3, 4, 3]

        # Distancia máxima tolerada
        self.D_max = 9


    def check(self, x):
        # Derivar hubs desde asignaciones
        hubs = []
        for _ in range(self.n_hubs):
            hubs.append(0)
        for h in x:
            hubs[h] = 1

        # Verificar capacidad y distancia máxima
        conteo = [0] * self.n_hubs
        for c in range(self.n_clients):
            h = x[c]

            # Verificar distancia máxima
            if self.distancias[c][h] > self.D_max:
                return False

            # Hub debe estar activo (ya se cumple por construcción)
            conteo[h] += 1
            if conteo[h] > self.capacidad[h]:
                return False

        return True

    def fit(self, x):
        # Derivar hubs desde asignaciones
        hubs = []
        for _ in range(self.n_hubs):
            hubs.append(0)

        for h in x:
            hubs[h] = 1

        # Calcular distancia total de asignación
        total = 0
        for c in range(self.n_clients):
            total += self.distancias[c][x[c]]

        # Agregar costo fijo de hubs activados
        for j in range(self.n_hubs):
            if hubs[j] == 1:
                total += self.costs[j]

        return total

    def keep_domain(self, v):
        probs = []
        for _ in range(self.n_hubs):
            probs.append(1 / (1 + math.exp(-(v + random.gauss(0, 0.5)))))

        total = sum(probs)
        normalized = [p / total for p in probs]

        # Muestreo proporcional
        r = random.random()
        acc = 0
        for i, p in enumerate(normalized):
            acc += p
            if r <= acc:
                return i
        return self.n_hubs - 1

class Particle:
    def __init__(self):
        self.p = Problem()
        self.dimension = self.p.n_clients
        self.position = []
        self.p_best = []
        # Inicialización BOA
        for _ in range(self.dimension):
            lb, ub = 0, self.p.n_hubs - 1
            r = random.random()
            val = math.floor(lb + r * (ub - lb + 1))
            self.position.append(val)
        self.update_p_best()

    def fitness(self, pos=None):
        return self.p.fit(pos or self.position)

    def is_feasible(self, pos=None):
        return self.p.check(pos or self.position)

    def update_p_best(self):
        self.p_best = self.position.copy()

    def fitness_p_best(self):
        return self.p.fit(self.p_best)

    # Fase de exploración
    def tracking_and_move(self, swarm):
        # Determinar las presas candidatas para cada bobcat
        CP = []
        Fi = self.fitness(self.position)
        for p in swarm:
            if ((p.fitness() < Fi) and (p != self)):
                CP.append(p)

        if (len(CP) == 0):
            return None

        SP = random.choice(CP).position
        new_position = []

        for k in range(self.dimension):
            I = random.choice([1, 2])
            r = random.random()
            val = self.position[k] + \
                ((1 - 2*r) * (SP[k] - I * self.position[k]))
            # Limitar al dominio válido (0 a n_hubs - 1)
            val = round(val)
            # val = self.p.keep_domain(val)
            val = max(0, min(self.p.n_hubs - 1, val))
            new_position.append(val)
        # Es factible y mejora
        if self.is_feasible(new_position):
            new_fit = self.fitness(new_position)
            if new_fit <= Fi:
                self.position = new_position
                self.update_p_best()

    def chasing_to_catch(self, t):
        new_position = []
        Fi = self.fitness(self.position)

        for k in range(self.dimension):
            r = random.random()
            val = self.position[k] + (((1-2*r)/(1+t)) * self.position[k])
            val = round(val)
            # val = self.p.keep_domain(val)
            val = max(0, min(self.p.n_hubs - 1, val))
            new_position.append(val)
        # Es factible y mejora
        if self.is_feasible(new_position):
            new_fit = self.fitness(new_position)
            if new_fit <= Fi:
                self.position = new_position
                self.update_p_best()

    def __str__(self):
        return f"p_best: {self.p_best}, fitness {self.fitness_p_best()}"

class BOA:
    def __init__(self, n_particles, max_iter):
        self.N = n_particles
        self.T = max_iter
        self.swarm = []
        self.global_best = None
        self.convergence = []

    def initialize(self):
        # Población inicial factible
        while len(self.swarm) < self.N:
            p = Particle()
            feasible = p.is_feasible()
            if feasible:
                self.swarm.append(p)
        # Mejor global
        self.global_best = min(self.swarm, key=lambda p: p.fitness())
        self.convergence.append(self.global_best.fitness())
        #self.show_results(0)

    def evolve(self):
        for t in range(1, self.T + 1):
            for idx, p in enumerate(self.swarm):
                feasible = False
                while not feasible:
                    p.tracking_and_move(self.swarm)
                    feasible = p.is_feasible()
            for idx, p in enumerate(self.swarm):
                feasible = False
                while not feasible:
                    p.chasing_to_catch(t)
                    feasible = p.is_feasible()
            # Actualizar global best
            candidate = min(self.swarm, key=lambda p: p.fitness())
            if candidate.fitness() < self.global_best.fitness():
                self.global_best = candidate
            self.convergence.append(self.global_best.fitness())
            #self.show_results(t)

    def solve(self):
        self.initialize()
        self.evolve()
        return self.global_best

    #def show_results(self, t):
        #print(f"t: {t}, g_best: {self.global_best}")


folder_path = 'C:/Users/fabia/Desktop/BOA'

part = 20
iter = 1000
n_runs = 30
# Ejecutar 40 veces
resultados = []
convergencia = None
mayor_mejora = -float('inf')

for i in range(n_runs):
    boa = BOA(part, iter)
    best = boa.solve()
    resultados.append(best.fitness())

    # Evaluar mejora
    mejora = boa.convergence[0] - boa.convergence[-1]
    if mejora > mayor_mejora:
        mayor_mejora = mejora
        convergencia = boa.convergence

# Resumen descriptivo
serie = pd.Series(resultados, name='fitness')
summary = serie.describe(percentiles=[0.25, 0.5, 0.75]).rename({
    '25%': 'Q1',
    '50%': 'Mediana',
    '75%': 'Q3'
})
summary_df = summary.to_frame().T

# Mostrar resultados al usuario
print("Resultados de cada corrida:", resultados)
print("\nTabla de resumen:\n", summary_df)

# Convertir a DataFrame a tabla descriptiva
resumen_final = summary_df
resumen_final

plt.figure(figsize=(10, 5))
plt.plot(convergencia, marker='o', linestyle='-')
plt.title('Gráfico de convergencia del BOA')
plt.xlabel('Iteración')
plt.ylabel('Mejor Valor (fitness)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(folder_path + '/D_Convergencia.png'))


# Cálculo de QMetric por iteración de la mejor corrida
f_optimo = 271
digitos_significativos = 4
f_max = max(convergencia)
qmetric_iter = []

for f_ach in convergencia:
    if f_max == f_optimo:
        q = 0
    else:
        q = (f_max - f_ach) / (f_max - f_optimo)
    qm = q 
    qm = max(0, min(1, qm))
    qmetric_iter.append(qm)

# boxplot

plt.figure(figsize=(6, 5))
sns.boxplot(data=resultados, orient='v', color='skyblue', width=0.4)
plt.title('Boxplot del mejor fitness en 40 ejecuciones')
plt.ylabel('Fitness')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(folder_path + '/D_Boxpot.png')


plt.figure(figsize=(10, 5))
plt.plot(qmetric_iter, marker='s', linestyle='-', color='green')
plt.title('QMetric por iteración (mejor ejecución)')
plt.xlabel('Iteración')
plt.ylabel('QMetric')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_path + '/D_QMetric.png')
