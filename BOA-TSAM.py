import random, math

class Problem:
    def __init__(self):
        self.n_clients = 6
        self.n_hubs = 3

        # Matriz de distancias cliente-hub
        self.distancias = [
            [5, 8, 11],
            [7, 4, 10],
            [6, 6, 6],
            [9, 5, 7],
            [8, 9, 5],
            [4, 7, 12]
        ]

        # Costo fijo por abrir cada hub
        self.costs = [20, 25, 15]

        # Capacidad máxima de atención por hub
        self.capacidad = [2, 3, 2]

        # Distancia máxima tolerada cliente–hub
        self.D_max = 8

    def check(self, x):
        x = [int(round(i)) for i in x]  # ← Esto es lo importante
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
        x = [int(round(i)) for i in x]  # ← Esto es lo importante
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
        #Inicialización BOA
        for _ in range(self.dimension):
            lb, ub = 0, self.p.n_hubs - 1
            r = random.random()
            val = math.floor(lb + r * (ub - lb + 1))
            self.position.append(val)
        self.update_p_best()

    def update_p_best(self):
        self.p_best = self.position.copy()

    def fitness(self, pos=None):
        return self.p.fit(pos or self.position)

    def explore(self, population):
        Fi = self.fitness()
        CP = [p for p in population if p.fitness() < Fi and p is not self]
        if not CP:
            return
        SP = random.choice(CP).position
        new_pos = self.position.copy()

        for j in range(self.dimension):
            r = random.random()
            I = random.choice([1, 2])
            delta = (1 - 2*r) * (SP[j] - I * self.position[j])
            new_pos[j] = self._clip(round(self.position[j] + delta))
            print(new_pos[j])

        # Sólo acepto si es factible y mejora
        if self.p.check(new_pos):
            new_fit = self.fitness(new_pos)
            if new_fit <= Fi:
                self.position = new_pos
                self.update_p_best()

    def exploit(self, t):
        Fi = self.fitness()
        new_pos = self.position.copy()

        for j in range(self.dimension):
            r = random.random()
            step = self.position[j] * (1 + 2*r) / (1 + t)
            new_pos[j] = self._clip(round(self.position[j] + step))

        # **Rechequear factibilidad antes de aceptar**
        if self.p.check(new_pos):
            new_fit = self.fitness(new_pos)
            if new_fit <= Fi:
                self.position = new_pos
                self.update_p_best()

    def _clip(self, val):
        return max(0, min(self.p.n_hubs - 1, val))

    def __str__(self):
        return f"X={self.position}  f={self.fitness()}"

class BOA:
    def __init__(self, n_particles=10, max_iter=25):
        self.N = n_particles
        self.T = max_iter
        self.swarm = []
        self.global_best = None

    def initialize(self):
        # 1) Población inicial factible
        while len(self.swarm) < self.N:
            p = Particle()
            if p.p.check(p.position):
                self.swarm.append(p)
        # 2) Mejor global
        self.global_best = min(self.swarm, key=lambda p: p.fitness())

    def evolve(self):
        for t in range(1, self.T + 1):
            # Phase 1: Exploration
            for p in self.swarm:
                p.explore(self.swarm)
            # Phase 2: Exploitation
            for p in self.swarm:
                p.exploit(t)
            # Actualizar global best
            candidate = min(self.swarm, key=lambda p: p.fitness())
            if candidate.fitness() < self.global_best.fitness():
                self.global_best = candidate
            print(f"Iter {t:2d}, best = {self.global_best}")

    def solve(self):
        self.initialize()
        self.evolve()
        return self.global_best

# Ejecutar BOA
best = BOA(n_particles=20, max_iter=50).solve()
print("Solución final:", best)