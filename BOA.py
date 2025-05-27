import math, random, sys
# 89 151 271
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
        #Inicialización BOA
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

    #Fase de exploración
    def tracking_and_move(self, swarm):
        #Determinar las presas candidatas para cada bobcat
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
            I = random.choice([1,2])
            r = random.random()
            val = self.position[k] + ((1 - 2*r) * (SP[k] - I * self.position[k]))
            # Limitar al dominio válido (0 a n_hubs - 1)
            val = round(val)
            val = max(0, min(self.p.n_hubs - 1, val)) #val = self.p.keep_domain(val)
            new_position.append(val)
        # Es factible y mejora
        if self.is_feasible(new_position):
            new_fit = self.fitness(new_position)
            if new_fit <= Fi:
                self.position = new_position
                self.update_p_best()
        
    def chasing_to_catch(self,t):
        new_position = []
        Fi = self.fitness(self.position)

        for k in range(self.dimension):
            r = random.random()
            val =  self.position[k] + (((1-2*r)/(1+t)) * self.position[k])
            val = round(val)
            val = max(0, min(self.p.n_hubs - 1, val)) #val = self.p.keep_domain(val)
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

    def initialize(self):
        #Población inicial factible
        while len(self.swarm) < self.N:
            p = Particle()
            feasible = p.is_feasible()
            if feasible:
                self.swarm.append(p)
                print(f"Bobcat {len(self.swarm)}: posición = {p.position}, fitness = {p.fitness()}")
        #Mejor global
        self.global_best = min(self.swarm, key=lambda p: p.fitness())
        self.show_results(0)

    def evolve(self):
        for t in range(1, self.T + 1):
            for idx, p in enumerate(self.swarm):
                prev_position = p.position.copy()
                feasible = False
                while not feasible:
                    p.tracking_and_move(self.swarm)
                    feasible = p.is_feasible()
                if p.position != prev_position:
                    print(f"t: {t}, Bobcat {idx+1} movió en exploracion de {prev_position} a {p.position}, fitness = {p.fitness()}")
            for idx, p in enumerate(self.swarm):
                prev_position = p.position.copy()
                feasible = False
                while not feasible:
                    p.chasing_to_catch(t) 
                    feasible = p.is_feasible()           
                if p.position != prev_position:
                    print(f"t: {t}, Bobcat {idx+1} movió en explotacion de {prev_position} a {p.position}, fitness = {p.fitness()}")

            # Actualizar global best
            candidate = min(self.swarm, key=lambda p: p.fitness())
            if candidate.fitness() < self.global_best.fitness():
                self.global_best = candidate

            self.show_results(t)

    def solve(self):
        self.initialize()
        self.evolve()
        return self.global_best
    
    def show_results(self, t):
        print(f"t: {t}, g_best: {self.global_best}")

if len(sys.argv) != 3:
    print("Uso: python BOA.py <iteraciones> <particulas>")
    sys.exit(1)

try:
    iteraciones = int(sys.argv[1])
    particulas = int(sys.argv[2])
    #Restricciones: ambos deben ser mayores que 0
    if iteraciones <= 0 or particulas <= 0:
        raise ValueError("Ambos valores deben ser enteros positivos mayores que cero.")

except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

part = sys.argv[1]
iter = sys.argv[2]

boa = BOA(int(part), int(iter))
best = boa.solve()
print("Solución final (posición):", best.position)
print("Fitness:", best.fitness())