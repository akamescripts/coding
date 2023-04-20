#Linear Search
def linear_search(arr, x):
    """
    Searches for the given element in the given array using Linear Search algorithm
    :param arr: list, array of elements to search in
    :param x: int, element to search for
    :return: int, index of the element if found, else -1
    """
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1

#Binary Search
def binary_search(arr, x):
    """
    Searches for the given element in the given array using Binary Search algorithm
    :param arr: list, array of elements to search in (Assumed to be sorted in non-decreasing order)
    :param x: int, element to search for
    :return: int, index of the element if found, else -1
    """
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1

#Jump Search
import math

def jump_search(arr, x):
    """
    Searches for the given element in the given array using Jump Search algorithm
    :param arr: list, array of elements to search in (Assumed to be sorted in non-decreasing order)
    :param x: int, element to search for
    :return: int, index of the element if found, else -1
    """
    n = len(arr)
    jump = int(math.sqrt(n))
    left, right = 0, 0
    while left < n and arr[left] <= x:
        right = min(n - 1, left + jump)
        if arr[left] <= x and arr[right] >= x:
            break
        left += jump
    if left >= n or arr[left] > x:
        return -1
    right = min(n - 1, right)
    i = left
    while i <= right and arr[i] <= x:
        if arr[i] == x:
            return i
        i += 1
    return -1

#Interpolation Search
def interpolation_search(arr, x):
    """
    Searches for the given element in the given array using Interpolation Search algorithm
    :param arr: list, array of elements to search in (Assumed to be sorted in non-decreasing order)
    :param x: int, element to search for
    :return: int, index of the element if found, else -1
    """
    n = len(arr)
    low, high = 0, n - 1
    while low <= high and arr[low] <= x <= arr[high]:
        pos = low + int(((x - arr[low]) * (high - low)) / (arr[high] - arr[low]))
        if arr[pos] == x:
            return pos
        elif arr[pos] < x:
            low = pos + 1
        else:
            high = pos - 1
    return -1

#Exponential Search
def exponential_search(arr, x):
    """
    Searches for the given element in the given array using Exponential Search algorithm
    :param arr: list, array of elements to search in (Assumed to be sorted in non-decreasing order)
    :param x: int, element to search for
    :return: int, index of the element if found, else -1
    """
    n = len(arr)
    if arr[0] == x:
        return 0
    i = 1
    while i < n and arr[i] <= x:
        i = i * 2
    return binary_search(arr, i//2, min(i, n-1), x)

def binary_search(arr, low, high, x):
    """
    Searches for the given element in the given sub-array of the given array using Binary Search algorithm
    :param arr: list, array of elements to search in
    :param low: int, start index of the sub-array
    :param high: int, end index of the sub-array
    :param x: int, element to search for
    :return: int, index of the element if found, else -1
    """
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1

#Fibonacci Search
def fibonacci_search(arr, x):
    """
    Perform Fibonacci search on a sorted array `arr` for the element `x`.
    Return the index of `x` if found, otherwise return -1.
    """
    n = len(arr)
    
    # Initialize the Fibonacci numbers
    fib_2 = 0  # Fib(n-2)
    fib_1 = 1  # Fib(n-1)
    fib = fib_2 + fib_1  # Fib(n)
    
    # Find the smallest Fibonacci number that is greater than or equal to `n`
    while fib < n:
        fib_2, fib_1 = fib_1, fib
        fib = fib_2 + fib_1
    
    # `offset` is the index from which to start the search
    offset = -1
    while fib > 1:
        i = min(offset + fib_2, n - 1)
        if arr[i] < x:
            # If the element to be searched is greater than the current element,
            # move the two Fibonacci numbers two steps down and set `offset` accordingly
            fib, fib_1, fib_2 = fib_1, fib_2, fib - fib_1
            offset = i
        elif arr[i] > x:
            # If the element to be searched is less than the current element,
            # move the two Fibonacci numbers one step down
            fib, fib_1, fib_2 = fib_2, fib_1 - fib_2, fib_2
        else:
            # Element found
            return i
    
    # If the last element to be compared is equal to x, return its index
    if fib_1 and arr[offset + 1] == x:
        return offset + 1
    
    # Element not found
    return -1

#Ternary Search
def ternary_search(array, target):
    left = 0
    right = len(array) - 1
    
    while left <= right:
        # dividing the search space into three parts
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        if array[mid1] == target:
            return mid1
        elif array[mid2] == target:
            return mid2
        elif target < array[mid1]:
            right = mid1 - 1
        elif target > array[mid2]:
            left = mid2 + 1
        else:
            left = mid1 + 1
            right = mid2 - 1
    
    return -1

#Depth-First Search (DFS)
def dfs(graph, start):
    visited = set()

    def dfs_helper(node):
        visited.add(node)
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs_helper(neighbor)

    dfs_helper(start)

#Breadth-First Search (BFS)
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            print(node)
            for neighbor in graph[node]:
                queue.append(neighbor)

#A* Search Algorithm
from heapq import heappop, heappush
from math import sqrt

def astar_search(start, goal, graph):
    # Initialize data structures
    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    # Define heuristic function
    def heuristic(a, b):
        # Euclidean distance
        return sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    # Search loop
    while frontier:
        current_cost, current_node = heappop(frontier)

        if current_node == goal:
            # Reconstruct path
            path = [current_node]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path

        for neighbor in graph[current_node]:
            # Calculate new cost
            new_cost = cost_so_far[current_node] + neighbor[1]
            if neighbor[0] not in cost_so_far or new_cost < cost_so_far[neighbor[0]]:
                # Update cost and priority queue
                cost_so_far[neighbor[0]] = new_cost
                priority = new_cost + heuristic(goal, neighbor[0])
                heappush(frontier, (priority, neighbor[0]))
                # Update came_from dictionary
                came_from[neighbor[0]] = current_node

    # If goal not found
    return None

#Best-First Search Algorithm
from queue import PriorityQueue

def best_first_search(graph, start, goal, h):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = h(next, goal)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far

#Hill Climbing Search Algorithm
def hill_climbing(initial_state, objective_function, get_neighbors, max_iterations=1000):
    current_state = initial_state
    current_score = objective_function(current_state)

    for i in range(max_iterations):
        neighbors = get_neighbors(current_state)
        best_neighbor = max(neighbors, key=lambda x: objective_function(x))
        best_score = objective_function(best_neighbor)

        if best_score > current_score:
            current_state = best_neighbor
            current_score = best_score
        else:
            break

    return current_state, current_score

#Simulated Annealing Search Algorithm
import math
import random

def simulated_annealing(initial_state, get_neighbors, objective_function, temperature, cooling_factor):
    current_state = initial_state
    current_cost = objective_function(current_state)
    best_state = current_state
    best_cost = current_cost
    while temperature > 0:
        neighbor = random.choice(get_neighbors(current_state))
        neighbor_cost = objective_function(neighbor)
        cost_diff = neighbor_cost - current_cost
        if cost_diff < 0 or math.exp(-cost_diff / temperature) > random.uniform(0, 1):
            current_state = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_state = current_state
                best_cost = current_cost
        temperature *= cooling_factor
    return best_state

#Tabu Search Algorithm
def tabu_search(initial_solution, objective_function, candidate_neighborhood, tabu_list_length, max_iterations):
    current_solution = initial_solution
    best_solution = initial_solution
    tabu_list = []

    for i in range(max_iterations):
        neighborhood = candidate_neighborhood(current_solution)

        # Select the best candidate solution that is not in the tabu list
        best_candidate = None
        for candidate in neighborhood:
            if candidate not in tabu_list:
                if best_candidate is None or objective_function(candidate) < objective_function(best_candidate):
                    best_candidate = candidate

        # Update the current solution
        current_solution = best_candidate

        # Update the tabu list
        tabu_list.append(best_candidate)
        if len(tabu_list) > tabu_list_length:
            tabu_list.pop(0)

        # Update the best solution
        if objective_function(current_solution) < objective_function(best_solution):
            best_solution = current_solution

    return best_solution

#Genetic  algorithm
import random

# define fitness function
def fitness_function(solution):
    # calculate fitness score for the solution
    return score

# initialize population
def initialize_population(population_size, gene_length):
    population = []
    for i in range(population_size):
        solution = [random.randint(0, 1) for j in range(gene_length)]
        population.append(solution)
    return population

# evaluate fitness of population
def evaluate_population(population):
    fitness_scores = []
    for solution in population:
        fitness_scores.append(fitness_function(solution))
    return fitness_scores

# select parents for crossover
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [fitness_score / total_fitness for fitness_score in fitness_scores]
    parent_indices = random.choices(range(len(population)), weights=probabilities, k=2)
    return [population[parent_index] for parent_index in parent_indices]

# perform crossover
def crossover(parents):
    crossover_point = random.randint(1, len(parents[0]) - 1)
    child1 = parents[0][:crossover_point] + parents[1][crossover_point:]
    child2 = parents[1][:crossover_point] + parents[0][crossover_point:]
    return [child1, child2]

# perform mutation
def mutate(solution, mutation_rate):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = 1 - solution[i]
    return solution

# evolve population
def evolve_population(population, fitness_scores, mutation_rate):
    new_population = []
    for i in range(len(population)):
        parents = select_parents(population, fitness_scores)
        child = crossover(parents)
        child = mutate(child, mutation_rate)
        new_population.append(child)
    return new_population

# main function to run genetic algorithm
def run_genetic_algorithm(population_size, gene_length, mutation_rate, num_generations):
    population = initialize_population(population_size, gene_length)
    for i in range(num_generations):
        fitness_scores = evaluate_population(population)
        population = evolve_population(population, fitness_scores, mutation_rate)
    best_solution = population[0]
    best_fitness = fitness_function(best_solution)
    for solution in population:
        fitness = fitness_function(solution)
        if fitness > best_fitness:
            best_solution = solution
            best_fitness = fitness
    return best_solution, best_fitness
#Ant Colony Optimization
import numpy as np

def ant_colony_optimization(distances, num_ants, num_iterations, alpha, beta, evaporation_rate, Q):
    num_cities = distances.shape[0]
    pheromones = np.ones((num_cities, num_cities))
    best_tour = None
    best_tour_length = np.inf
    
    for iteration in range(num_iterations):
        ant_tours = np.zeros((num_ants, num_cities), dtype=np.int32)
        ant_lengths = np.zeros(num_ants)
        
        for ant in range(num_ants):
            visited = np.zeros(num_cities, dtype=np.bool)
            current_city = np.random.randint(num_cities)
            ant_tours[ant, 0] = current_city
            visited[current_city] = True
            
            for i in range(1, num_cities):
                unvisited = ~visited
                pheromone_scores = pheromones[current_city, unvisited]
                heuristic_scores = 1.0 / distances[current_city, unvisited]
                probabilities = pheromone_scores**alpha * heuristic_scores**beta
                probabilities /= probabilities.sum()
                next_city = np.random.choice(np.arange(num_cities)[unvisited], p=probabilities)
                ant_tours[ant, i] = next_city
                visited[next_city] = True
                ant_lengths[ant] += distances[current_city, next_city]
                current_city = next_city
            
            ant_lengths[ant] += distances[ant_tours[ant, -1], ant_tours[ant, 0]]
        
        # Update pheromones
        pheromones *= (1.0 - evaporation_rate)
        
        for ant in range(num_ants):
            tour = ant_tours[ant]
            tour_length = ant_lengths[ant]
            delta_pheromones = Q / tour_length
            for i in range(num_cities):
                city1, city2 = tour[i], tour[(i + 1) % num_cities]
                pheromones[city1, city2] += delta_pheromones
        
        # Update best tour
        if ant_lengths.min() < best_tour_length:
            best_tour_length = ant_lengths.min()
            best_tour = ant_tours[ant_lengths.argmin()]
    
    return best_tour, best_tour_length


#Artificial Bee Colony Algorithm
import random

class Particle:
    def __init__(self, num_dimensions, min_x, max_x):
        self.position = [random.uniform(min_x, max_x) for _ in range(num_dimensions)]
        self.velocity = [0] * num_dimensions
        self.best_position = self.position[:]
        self.best_fitness = float('inf')

    def update_velocity(self, global_best_position, c1, c2):
        for i in range(len(self.velocity)):
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = self.velocity[i] + cognitive + social

    def update_position(self, min_x, max_x):
        for i in range(len(self.position)):
            self.position[i] = self.position[i] + self.velocity[i]
            if self.position[i] < min_x:
                self.position[i] = min_x
            elif self.position[i] > max_x:
                self.position[i] = max_x

    def evaluate_fitness(self, fitness_func):
        fitness = fitness_func(self.position)
        if fitness < self.best_fitness:
            self.best_position = self.position[:]
            self.best_fitness = fitness

def particle_swarm_optimization(num_particles, num_dimensions, fitness_func, min_x, max_x, num_iterations, c1, c2):
    swarm = [Particle(num_dimensions, min_x, max_x) for _ in range(num_particles)]
    global_best_position = swarm[0].position[:]
    global_best_fitness = float('inf')

    for _ in range(num_iterations):
        for particle in swarm:
            particle.evaluate_fitness(fitness_func)
            if particle.best_fitness < global_best_fitness:
                global_best_position = particle.best_position[:]
                global_best_fitness = particle.best_fitness
            particle.update_velocity(global_best_position, c1, c2)
            particle.update_position(min_x, max_x)

    return global_best_position, global_best_fitness

#Harmony Search Algorithm
import random

def harmony_search(cost_func, num_vars, bounds, max_iters, num_harmonies=20, 
                   harmony_memory_rate=0.95, pitch_adjusting_rate=0.7):
    """
    Harmony search algorithm for optimization problems.
    
    Args:
    - cost_func: function that takes a list of input variables and returns a scalar cost.
    - num_vars: integer number of input variables.
    - bounds: list of tuples, where each tuple contains the minimum and maximum bounds for the corresponding variable.
    - max_iters: integer number of iterations to run the algorithm for.
    - num_harmonies: integer number of harmonies in the harmony memory.
    - harmony_memory_rate: float rate between 0 and 1, determines how much of the memory is filled with existing harmonies.
    - pitch_adjusting_rate: float rate between 0 and 1, determines the probability of adjusting each pitch in a new harmony.
    
    Returns:
    - tuple containing the best cost found and the corresponding input variable values.
    """
    
    # Initialize the harmony memory with random harmonies within the specified bounds
    memory = [tuple(random.uniform(bound[0], bound[1]) for bound in bounds) for _ in range(num_harmonies)]
    
    # Loop over iterations
    for i in range(max_iters):
        # Generate a new harmony by adjusting pitches of existing harmonies and/or adding new pitches
        if random.random() < harmony_memory_rate:
            # Choose an existing harmony from the memory
            harmony = random.choice(memory)
            # Adjust some of the pitches in the harmony
            new_harmony = list(harmony)
            for j in range(num_vars):
                if random.random() < pitch_adjusting_rate:
                    new_harmony[j] = random.uniform(bounds[j][0], bounds[j][1])
            new_harmony = tuple(new_harmony)
        else:
            # Generate a completely new harmony
            new_harmony = tuple(random.uniform(bound[0], bound[1]) for bound in bounds)
        
        # Evaluate the cost of the new harmony
        new_cost = cost_func(new_harmony)
        
        # Update the memory with the new harmony if it's better than the worst one in the memory
        if new_cost < cost_func(max(memory, key=cost_func)):
            memory[memory.index(max(memory, key=cost_func))] = new_harmony
    
    # Return the best harmony found in the memory
    best_harmony = min(memory, key=cost_func)
    return cost_func(best_harmony), best_harmony

#Firefly Algorithm
import random
import math

# Firefly class
class Firefly:
    def __init__(self, n):
        self.n = n
        self.position = [random.uniform(0, 1) for _ in range(n)]
        self.brightness = 0
    
    def __str__(self):
        return f"Position: {self.position}, Brightness: {self.brightness}"
    

# Firefly algorithm
def firefly_algorithm(n, f, max_iterations, alpha=0.5, beta=1.0, gamma=1.0):
    # Initialize fireflies
    fireflies = [Firefly(n) for _ in range(f)]
    
    # Main loop
    for t in range(max_iterations):
        # Evaluate brightness
        for i in range(f):
            fireflies[i].brightness = fitness_function(fireflies[i].position)
        
        # Sort fireflies by brightness
        fireflies.sort(key=lambda x: x.brightness, reverse=True)
        
        # Move fireflies
        for i in range(f):
            for j in range(f):
                if fireflies[i].brightness < fireflies[j].brightness:
                    distance = math.sqrt(sum([(fireflies[i].position[k] - fireflies[j].position[k])**2 for k in range(n)]))
                    attractiveness = beta*math.exp(-gamma*distance**2)
                    for k in range(n):
                        fireflies[i].position[k] += alpha*(fireflies[j].position[k] - fireflies[i].position[k])*attractiveness + random.gauss(0, 1)
    
    # Return best solution
    return fireflies[0].position
    
    
# Example usage
def fitness_function(x):
    return sum([xi**2 for xi in x])

n = 10  # dimension of search space
f = 20  # number of fireflies
max_iterations = 1000  # maximum number of iterations
solution = firefly_algorithm(n, f, max_iterations)

print("Solution:", solution)
print("Fitness:", fitness_function(solution))

#Grey Wolf Optimization Algorithm
import numpy as np

def grey_wolf_optimization(obj_func, lb, ub, num_dimensions, num_wolves=30, max_iter=100):
    # Grey Wolf Optimization Algorithm
    # obj_func: objective function to be optimized
    # lb: lower bounds of the decision variables
    # ub: upper bounds of the decision variables
    # num_dimensions: number of decision variables
    # num_wolves: number of search agents (default = 30)
    # max_iter: maximum number of iterations (default = 100)
    
    # initialize the positions of the grey wolves
    positions = np.random.uniform(lb, ub, size=(num_wolves, num_dimensions))
    
    # initialize the best positions and scores
    best_positions = positions.copy()
    best_scores = np.array([obj_func(p) for p in positions])
    
    # initialize the alpha, beta, and delta positions and scores
    alpha_pos, beta_pos, delta_pos = None, None, None
    alpha_score, beta_score, delta_score = float("inf"), float("inf"), float("inf")
    
    # main loop
    for i in range(max_iter):
        # update the positions of the grey wolves
        a = 2.0 - i * (2.0 / max_iter)  # parameter for controlling the search intensity
        for j in range(num_wolves):
            r1, r2 = np.random.random(), np.random.random()
            A1, A2 = 2.0 * a * r1 - a, 2.0 * a * r2 - a
            C1, C2 = 2.0 * np.random.random(), 2.0 * np.random.random()
            D_alpha = np.abs(C1 * alpha_pos - positions[j])
            D_beta = np.abs(C2 * beta_pos - positions[j])
            D_delta = np.abs(C2 * delta_pos - positions[j])
            X1 = alpha_pos - A1 * D_alpha
            X2 = beta_pos - A2 * D_beta
            X3 = delta_pos - a * D_delta
            positions[j] = (X1 + X2 + X3) / 3.0
            
            # apply bounds
            positions[j] = np.maximum(positions[j], lb)
            positions[j] = np.minimum(positions[j], ub)
        
        # evaluate the objective function
        scores = np.array([obj_func(p) for p in positions])
        
        # update the best positions and scores
        mask = scores < best_scores
        best_positions[mask] = positions[mask]
        best_scores[mask] = scores[mask]
        
        # update the alpha, beta, and delta positions and scores
        sorted_indices = np.argsort(scores)
        alpha_pos, alpha_score = positions[sorted_indices[0]], scores[sorted_indices[0]]
        beta_pos, beta_score = positions[sorted_indices[1]], scores[sorted_indices[1]]
        delta_pos, delta_score = positions[sorted_indices[2]], scores[sorted_indices[2]]
        
        # print the progress
        print(f"Iteration {i+1}/{max_iter}: Best Score = {best_scores.min():.6f}")
    
    # return the best positions and scores
    return best_positions, best_scores

#Krill Herd Algorithm
import numpy as np

def krill_herd_algorithm(cost_function, dim, nkr, iterations):
    # Define the boundaries of the search space
    lb = -100 * np.ones(dim)
    ub = 100 * np.ones(dim)

    # Initialize krill population and their weights
    krill = np.random.uniform(lb, ub, (nkr, dim))
    weights = np.zeros(nkr)

    # Initialize the best krill position and its cost
    best_position = np.zeros(dim)
    best_cost = float('inf')

    for t in range(iterations):
        # Calculate the weight of each krill
        for i in range(nkr):
            weights[i] = 1 / (1 + cost_function(krill[i]))

        # Sort the krill by weight
        sorted_indices = np.argsort(weights)
        sorted_krill = krill[sorted_indices]

        # Update the position of the krill
        for i in range(nkr):
            # Calculate the distances to the other krill
            distances = np.linalg.norm(sorted_krill - sorted_krill[i], axis=1)

            # Find the nearest and the farthest krill
            nearest_index = np.argmin(distances)
            farthest_index = np.argmax(distances)

            # Calculate the average position of the nearest and the farthest krill
            average_position = (sorted_krill[nearest_index] + sorted_krill[farthest_index]) / 2

            # Update the position of the krill
            krill[i] += np.random.uniform(-1, 1) * (best_position - krill[i]) + np.random.uniform(-1, 1) * (average_position - krill[i])

            # Enforce the boundaries of the search space
            krill[i] = np.maximum(krill[i], lb)
            krill[i] = np.minimum(krill[i], ub)

        # Update the best krill position and its cost
        for i in range(nkr):
            cost = cost_function(krill[i])
            if cost < best_cost:
                best_position = krill[i]
                best_cost = cost

    return best_position, best_cost

#Beam Search Algorithm
def beam_search(initial_state, beam_width, evaluate, successors):
    """
    Beam Search algorithm implementation.
    :param initial_state: The starting state of the search problem.
    :param beam_width: The maximum number of states to be considered at each iteration.
    :param evaluate: A function that takes a state as input and returns a cost.
    :param successors: A function that takes a state as input and returns a list of successor states.
    :return: The best path from the initial state to a goal state.
    """
    # Create the initial beam
    beam = [(initial_state, [initial_state], 0)]
    # Loop until a goal state is found
    while beam:
        # Expand the current beam
        new_beam = []
        for state, path, cost in beam:
            for succ in successors(state):
                new_path = path + [succ]
                new_cost = cost + evaluate(succ)
                new_beam.append((succ, new_path, new_cost))
        # Select the best successors to become the new beam
        beam = sorted(new_beam, key=lambda x: x[2])[:beam_width]
        # Check if any of the new states is a goal state
        for state, path, cost in beam:
            if is_goal(state):
                return path
    # No goal state found
    return None

#Bidirectional Search Algorithm
from queue import Queue
 
def bidirectional_search(start, end, graph):
    visited_forward = set()
    visited_backward = set()
    queue_forward = Queue()
    queue_backward = Queue()
    queue_forward.put(start)
    queue_backward.put(end)
    visited_forward.add(start)
    visited_backward.add(end)
    
    while not queue_forward.empty() and not queue_backward.empty():
        # search from forward direction
        curr_forward = queue_forward.get()
        if curr_forward == end or curr_forward in visited_backward:
            return True
        for neighbor in graph[curr_forward]:
            if neighbor not in visited_forward:
                visited_forward.add(neighbor)
                queue_forward.put(neighbor)
        
        # search from backward direction
        curr_backward = queue_backward.get()
        if curr_backward == start or curr_backward in visited_forward:
            return True
        for neighbor in graph[curr_backward]:
            if neighbor not in visited_backward:
                visited_backward.add(neighbor)
                queue_backward.put(neighbor)
    
    return False

#Depth-Limited Search Algorithm
def depth_limited_search(node, goal, depth_limit):
    """
    Performs a depth-limited search on a tree/graph starting from a given node.
    Stops searching when the goal is found or when the depth limit is reached.

    Args:
    - node: The starting node of the search.
    - goal: The goal node we are searching for.
    - depth_limit: The maximum depth we are allowed to search.

    Returns:
    - The path from the starting node to the goal node, if it exists.
    - None otherwise.
    """
    # Check if we have reached the goal node
    if node == goal:
        return [node]
    
    # Check if we have reached the maximum depth
    if depth_limit == 0:
        return None
    
    # Recursively search each child node
    for child in node.children:
        result = depth_limited_search(child, goal, depth_limit - 1)
        if result is not None:
            return [node] + result
    
    # If we reach here, the goal was not found at this depth
    return None

#Iterative Deepening Depth-First Search Algorithm
def iddfs(start_node, goal_test, actions, max_depth):
    for depth in range(max_depth):
        result = dfs(start_node, goal_test, actions, depth)
        if result is not None:
            return result

def dfs(node, goal_test, actions, depth):
    if depth == 0:
        if goal_test(node):
            return node
        else:
            return None
    else:
        for action in actions(node):
            child = action(node)
            result = dfs(child, goal_test, actions, depth-1)
            if result is not None:
                return result
        return None

#Uniform Cost Search Algorithm
from queue import PriorityQueue

def uniform_cost_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put((0, start))
    explored = set()
    while not frontier.empty():
        cost, node = frontier.get()
        if node == goal:
            return cost
        explored.add(node)
        for child, child_cost in graph[node].items():
            if child not in explored:
                frontier.put((cost + child_cost, child))
    return None

#Dijkstra's Algorithm
import heapq

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

#Bellman-Ford Algorithm
def bellman_ford(graph, start):
    # Initialization
    dist = {v: float('inf') for v in graph}
    dist[start] = 0
    
    # Relax edges
    for i in range(len(graph) - 1):
        for u, neighbors in graph.items():
            for v, weight in neighbors.items():
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
    
    # Check for negative cycles
    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            if dist[u] + weight < dist[v]:
                raise Exception("Negative cycle detected!")
    
    return dist

#Floyd-Warshall Algorithm
def floyd_warshall(graph):
    """
    Find the shortest path between all pairs of vertices in a weighted graph using the Floyd-Warshall Algorithm.
    Returns a 2D matrix where matrix[i][j] represents the shortest path between vertices i and j.
    If there is no path between two vertices, the value is set to infinity.
    """
    n = len(graph)
    dist = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = graph[i][j]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

#Johnson's Algorithm
import heapq
from math import inf

def johnsons_algorithm(graph):
    # add super-source and edges to all other nodes with zero weight
    super_source = 'super_source'
    graph[super_source] = {node: 0 for node in graph.keys()}
    
    # find shortest paths from super-source to every other node
    distances = bellman_ford_algorithm(graph, super_source)
    
    # adjust edge weights to remove negative weights
    for source in graph.keys():
        for dest, weight in graph[source].items():
            graph[source][dest] += distances[source] - distances[dest]
            
    # run Dijkstra's algorithm for each node to find shortest paths
    shortest_paths = {}
    for node in graph.keys():
        shortest_paths[node] = dijkstras_algorithm(graph, node)
    
    # remove super-source node from results
    del shortest_paths[super_source]
    
    return shortest_paths

def bellman_ford_algorithm(graph, start):
    distances = {node: inf for node in graph.keys()}
    distances[start] = 0
    
    for _ in range(len(graph.keys()) - 1):
        for source, edges in graph.items():
            for dest, weight in edges.items():
                if distances[source] + weight < distances[dest]:
                    distances[dest] = distances[source] + weight
                    
    # check for negative cycles
    for source, edges in graph.items():
        for dest, weight in edges.items():
            if distances[source] + weight < distances[dest]:
                raise ValueError('Graph contains negative cycle')
                
    return distances

def dijkstras_algorithm(graph, start):
    distances = {node: inf for node in graph.keys()}
    distances[start] = 0
    heap = [(0, start)]
    
    while heap:
        current_dist, current_node = heapq.heappop(heap)
        if current_dist > distances[current_node]:
            continue
            
        for neighbor, weight in graph[current_node].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
                
    return distances

#Kruskal's Algorithm
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, src, dest, weight):
        self.graph.append([src, dest, weight])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        x_root = self.find(parent, x)
        y_root = self.find(parent, y)

        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
        elif rank[x_root] > rank[y_root]:
            parent[y_root] = x_root
        else:
            parent[y_root] = x_root
            rank[x_root] += 1

    def kruskal_mst(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:
            src, dest, weight = self.graph[i]
            i += 1
            x = self.find(parent, src)
            y = self.find(parent, dest)

            if x != y:
                e += 1
                result.append([src, dest, weight])
                self.union(parent, rank, x, y)

        return result

#Prim's Algorithm
import heapq

def prim(graph):
    # Initialize a list to keep track of visited nodes and a priority queue for edges
    visited = set()
    edges = []
    start_node = next(iter(graph))
    # Add the start node to visited and all its edges to the queue
    visited.add(start_node)
    for neighbor, weight in graph[start_node].items():
        heapq.heappush(edges, (weight, start_node, neighbor))
    # Initialize an empty dictionary to store the edges of the minimum spanning tree
    mst = {}
    while edges:
        # Get the edge with the smallest weight from the queue
        weight, u, v = heapq.heappop(edges)
        if v not in visited:
            # Add the edge to the minimum spanning tree and mark the endpoint as visited
            mst[(u, v)] = weight
            visited.add(v)
            # Add all unvisited neighbors of the endpoint to the queue
            for neighbor, weight in graph[v].items():
                if neighbor not in visited:
                    heapq.heappush(edges, (weight, v, neighbor))
    return mst

#Depth-First Branch and Bound Algorithm
import heapq

class State:
    def __init__(self, cost, path):
        self.cost = cost
        self.path = path
        
    def __lt__(self, other):
        return self.cost < other.cost
        
def depth_first_branch_and_bound(graph, start, goal, heuristic):
    stack = [(start, 0, [])]
    visited = set()
    while stack:
        node, cost, path = stack.pop()
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            children = [(child, cost+graph[node][child]) for child in graph[node]]
            children.sort(key=lambda x: heuristic[x[0]])
            for child, child_cost in children:
                if child not in visited:
                    heapq.heappush(stack, (child, child_cost, path + [child]))
    return None

#Best-First Branch and Bound Algorithm
import heapq

def bfbb(root, heuristic, is_feasible, cost, successor_fn):
    frontier = [(heuristic(root), root)]
    best = None
    while frontier:
        _, node = heapq.heappop(frontier)
        if is_feasible(node):
            node_cost = cost(node)
            if best is None or node_cost < cost(best):
                best = node
            for successor in successor_fn(node):
                f = node_cost + heuristic(successor)
                heapq.heappush(frontier, (f, successor))
    return best

#Iterative Deepening A* Algorithm
def ida_star(root, heuristic, goal_test, successor_fn):
    threshold = heuristic(root)
    while True:
        result, new_threshold = search(root, 0, threshold, heuristic, goal_test, successor_fn)
        if result is not None:
            return result
        if new_threshold == float('inf'):
            return None
        threshold = new_threshold

def search(node, g, threshold, heuristic, goal_test, successor_fn):
    f = g + heuristic(node)
    if f > threshold:
        return None, f
    if goal_test(node):
        return node, f
    min_cost = float('inf')
    for successor in successor_fn(node):
        result, new_threshold = search(successor, g + 1, threshold, heuristic, goal_test, successor_fn)
        if result is not None:
            return result, f
        if new_threshold < min_cost:
            min_cost = new_threshold
    return None, min_cost

#USE 
start_node = ...
heuristic = lambda node: ...
goal_test = lambda node: ...
successor_fn = lambda node: ...

result = ida_star(start_node, heuristic, goal_test, successor_fn)
if result is None:
    print("No solution found")
else:
    print("Solution found:", result)


#Recursive Best-First Search Algorithm
import sys

def rbfs(start_node, heuristic, is_goal, successor_fn):
    def search(node, f_limit):
        if heuristic(node) >= f_limit:
            return None, heuristic(node)
        if is_goal(node):
            return node, 0
        successors = [(heuristic(successor), successor) for successor in successor_fn(node)]
        while successors:
            _, best_successor = min(successors)
            alternative, alternative_cost = search(best_successor, min(f_limit, heuristic(best_successor)))
            if alternative is not None:
                return alternative, alternative_cost
            successors.remove((heuristic(best_successor), best_successor))
            successors.append((alternative_cost + heuristic(best_successor), best_successor))
        return None, float('inf')
    
    sys.setrecursionlimit(10000) # Set the recursion limit to 10000 or a higher value
    return search(start_node, float('inf'))[0]

#Limited Discrepancy Search Algorithm
def lds(current, heuristic, is_goal, successor_fn, max_discrepancies):
    if is_goal(current):
        return current
    if max_discrepancies == 0:
        return None
    
    for successor in successor_fn(current):
        if heuristic(successor) < heuristic(current):
            continue
        discrepancy = 0
        for i in range(len(current)):
            if current[i] != successor[i]:
                discrepancy += 1
            if discrepancy > max_discrepancies:
                break
        if discrepancy <= max_discrepancies:
            result = lds(successor, heuristic, is_goal, successor_fn, max_discrepancies - discrepancy)
            if result is not None:
                return result
    
    return None


#Alpha-Beta Pruning Algorithm
def alphabeta(node, depth, alpha, beta, maximizing_player, evaluation_fn, successor_fn):
    if depth == 0 or is_terminal(node):
        return evaluation_fn(node)
    
    if maximizing_player:
        value = -float('inf')
        for successor in successor_fn(node):
            value = max(value, alphabeta(successor, depth-1, alpha, beta, False, evaluation_fn, successor_fn))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for successor in successor_fn(node):
            value = min(value, alphabeta(successor, depth-1, alpha, beta, True, evaluation_fn, successor_fn))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

#Monte Carlo Tree Search Algorithm
import random
import math

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0
        
    def select(self, c):
        scores = [(child.wins / child.visits) + c * math.sqrt(2 * math.log(self.visits) / child.visits) for child in self.children]
        return self.children[scores.index(max(scores))]
        
    def expand(self, actions):
        for action in actions:
            state = self.state.apply(action)
            child = Node(state, self, action)
            self.children.append(child)
            
    def simulate(self):
        state = self.state.clone()
        while not state.is_terminal():
            action = random.choice(state.get_legal_actions())
            state.apply(action)
        return state.get_result(self.state.get_player_id())
    
    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)
            
def mcts(root, iterations, c):
    for i in range(iterations):
        node = root
        while node.children:
            node = node.select(c)
        if node.visits == 0:
            result = node.simulate()
        else:
            node.expand(node.state.get_legal_actions())
            child = random.choice(node.children)
            result = child.simulate()
        node.backpropagate(result)
    return max(root.children, key=lambda child: child.visits).action

#Red-Black Tree Search Algorithm
class Node:
    def __init__(self, key):
        self.key = key
        self.parent = None
        self.left = None
        self.right = None
        self.color = 'red'

class RedBlackTree:
    def __init__(self):
        self.root = None
    
    def insert(self, key):
        node = Node(key)
        if self.root is None:
            self.root = node
        else:
            parent = self.root
            while True:
                if key < parent.key:
                    if parent.left is None:
                        parent.left = node
                        node.parent = parent
                        break
                    else:
                        parent = parent.left
                else:
                    if parent.right is None:
                        parent.right = node
                        node.parent = parent
                        break
                    else:
                        parent = parent.right
            self.__fix_insert(node)
    
    def delete(self, key):
        node = self.search(key)
        if node is None:
            return
        if node.left is None or node.right is None:
            child = node.left or node.right
            if node.parent is None:
                self.root = child
            elif node.parent.left == node:
                node.parent.left = child
            else:
                node.parent.right = child
            if child is not None:
                child.parent = node.parent
            if node.color == 'black':
                self.__fix_delete(child, node.parent)
        else:
            successor = self.__find_successor(node)
            node.key = successor.key
            self.delete(successor.key)
    
    def search(self, key):
        node = self.root
        while node is not None:
            if node.key == key:
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None
    
    def __fix_insert(self, node):
        while node.parent is not None and node.parent.color == 'red':
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle is not None and uncle.color == 'red':
                    node.parent.color = 'black'
                    uncle.color = 'black'
                    node.parent.parent.color = 'red'
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self.__rotate_left(node)
                    node.parent.color = 'black'
                    node.parent.parent.color = 'red'
                    self.__rotate_right(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle is not None and uncle.color == 'red':
                    node.parent.color = 'black'
                    uncle.color = 'black'
                    node.parent.parent.color = 'red'
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self.__rotate_right(node)
                    node.parent.color = 'black'
                    node.parent.parent.color = 'red'
                    self.__rotate_left(node.parent.parent)
        self.root.color = 'black'
    
    def __fix_delete(self, node, parent):
        while node != self.root and (node is None or node.color == 'black'):
            if node == parent.left:
                sibling = parent.right
                if sibling.color == 'red':
                    sibling.color = 'black'
                    parent.color = 'red'
                    self.__rotate_left(parent)
                    sibling = parent.right
                if (sibling.left is None or sibling.left.color == 'black') and (sibling.right is None or sibling.right.color == 'black'):
                sibling.color = 'red'
                node = parent
                parent = node.parent
            else:
                if sibling.right is None or sibling.right.color == 'black':
                    sibling.left.color = 'black'
                    sibling.color = 'red'
                    self.__rotate_right(sibling)
                    sibling = parent.right
                sibling.color = parent.color
                parent.color = 'black'
                sibling.right.color = 'black'
                self.__rotate_left(parent)
                node = self.root
        else:
            sibling = parent.left
            if sibling.color == 'red':
                sibling.color = 'black'
                parent.color = 'red'
                self.__rotate_right(parent)
                sibling = parent.left
            if (sibling.right is None or sibling.right.color == 'black') and (sibling.left is None or sibling.left.color == 'black'):
                sibling.color = 'red'
                node = parent
                parent = node.parent
            else:
                if sibling.left is None or sibling.left.color == 'black':
                    sibling.right.color = 'black'
                    sibling.color = 'red'
                    self.__rotate_left(sibling)
                    sibling = parent.left
                sibling.color = parent.color
                parent.color = 'black'
                sibling.left.color = 'black'
                self.__rotate_right(parent)
                node = self.root
    if node is not None:
        node.color = 'black'

def __rotate_left(self, node):
    right_child = node.right
    node.right = right_child.left
    if right_child.left is not None:
        right_child.left.parent = node
    right_child.parent = node.parent
    if node.parent is None:
        self.root = right_child
    elif node == node.parent.left:
        node.parent.left = right_child
    else:
        node.parent.right = right_child
    right_child.left = node
    node.parent = right_child

def __rotate_right(self, node):
    left_child = node.left
    node.left = left_child.right
    if left_child.right is not None:
        left_child.right.parent = node
    left_child.parent = node.parent
    if node.parent is None:
        self.root = left_child
    elif node == node.parent.right:
        node.parent.right = left_child
    else:
        node.parent.left = left_child
    left_child.right = node
    node.parent = left_child

def __find_successor(self, node):
    if node.right is not None:
        successor = node.right
        while successor.left is not None:
            successor = successor.left
        return successor
    else:
        successor = node.parent
        while successor is not None and successor.right == node:
            node = successor
            successor = successor.parent
        return successor

def inorder_traversal(self):
    if self.root is not None:
        self.__inorder_traversal_helper(self.root)

def __inorder_traversal_helper(self, node):
    if node.left is not None:
        self.__inorder_traversal_helper(node.left)
    print(node.key)
    if node.right is not None:
        self.__inorder_traversal_helper(node.right)


#AVL Tree Search Algorithm
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        def _insert(node, key):
            if not node:
                return AVLNode(key)
            elif key < node.key:
                node.left = _insert(node.left, key)
            else:
                node.right = _insert(node.right, key)

            node.height = 1 + max(self._get_height(node.left),
                                  self._get_height(node.right))
            balance = self._get_balance(node)

            if balance > 1 and key < node.left.key:
                return self._right_rotate(node)

            if balance < -1 and key > node.right.key:
                return self._left_rotate(node)

            if balance > 1 and key > node.left.key:
                node.left = self._left_rotate(node.left)
                return self._right_rotate(node)

            if balance < -1 and key < node.right.key:
                node.right = self._right_rotate(node.right)
                return self._left_rotate(node)

            return node

        self.root = _insert(self.root, key)

    def search(self, key):
        def _search(node, key):
            if not node:
                return None
            elif node.key == key:
                return node
            elif key < node.key:
                return _search(node.left, key)
            else:
                return _search(node.right, key)

        return _search(self.root, key)

    def _left_rotate(self, node):
        right_child = node.right
        left_grandchild = right_child.left

        right_child.left = node
        node.right = left_grandchild

        node.height = 1 + max(self._get_height(node.left),
                              self._get_height(node.right))
        right_child.height = 1 + max(self._get_height(right_child.left),
                                     self._get_height(right_child.right))

        return right_child

    def _right_rotate(self, node):
        left_child = node.left
        right_grandchild = left_child.right

        left_child.right = node
        node.left = right_grandchild

        node.height = 1 + max(self._get_height(node.left),
                              self._get_height(node.right))
        left_child.height = 1 + max(self._get_height(left_child.left),
                                    self._get_height(left_child.right))

        return left_child

    def _get_height(self, node):
        if not node:
            return 0
        else:
            return node.height

    def _get_balance(self, node):
        if not node:
            return 0
        else:
            return self._get_height(node.left) - self._get_height(node.right)

def avl_tree_search(root, key):
    if not root:
        return None
    elif root.key == key:
        return root
    elif key < root.key:
        return avl_tree_search(root.left, key)
    else:
        return avl_tree_search(root.right, key)


#B-Tree Search Algorithm
class BTreeNode:
    def __init__(self, t, leaf):
        self.t = t
        self.keys = []
        self.children = []
        self.leaf = leaf

    def search(self, key):
        i = 0
        while i < len(self.keys) and key > self.keys[i]:
            i += 1

        if i < len(self.keys) and key == self.keys[i]:
            return self

        if self.leaf:
            return None

        return self.children[i].search(key)

class BTree:
    def __init__(self, t):
        self.root = None
        self.t = t

    def search(self, key):
        if self.root is None:
            return None
        else:
            return self.root.search(key)

#B+ Tree Search Algorithm
class BPlusTreeNode:
    def __init__(self, order, leaf):
        self.order = order
        self.keys = []
        self.values = []
        self.children = []
        self.leaf = leaf

    def search(self, key):
        i = 0
        while i < len(self.keys) and key > self.keys[i]:
            i += 1

        if i < len(self.keys) and key == self.keys[i]:
            return self

        if self.leaf:
            return None

        return self.children[i].search(key)

class BPlusTree:
    def __init__(self, order):
        self.root = None
        self.order = order

    def search(self, key):
        if self.root is None:
            return None
        else:
            return self.root.search(key)

#2-3 Tree Search Algorithm
class TwoThreeTreeNode:
    def __init__(self, keys=None, values=None, children=None):
        self.keys = keys or []
        self.values = values or []
        self.children = children or []

    def search(self, key):
        i = 0
        while i < len(self.keys) and key > self.keys[i]:
            i += 1

        if i < len(self.keys) and key == self.keys[i]:
            return self

        if not self.children:
            return None

        return self.children[i].search(key)

class TwoThreeTree:
    def __init__(self):
        self.root = None

    def search(self, key):
        if self.root is None:
            return None
        else:
            return self.root.search(key)

#Trie Search Algorithm
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current_node = self.root
        for char in word:
            if char not in current_node.children:
                current_node.children[char] = TrieNode()
            current_node = current_node.children[char]
        current_node.is_word = True

    def search(self, word):
        current_node = self.root
        for char in word:
            if char not in current_node.children:
                return False
            current_node = current_node.children[char]
        return current_node.is_word

#Suffix Tree Search Algorithm
class SuffixTreeNode:
    def __init__(self, start, end, suffix_link=None):
        self.start = start
        self.end = end
        self.suffix_link = suffix_link
        self.children = {}

    def edge_length(self, current_idx):
        return min(self.end, current_idx + 1) - self.start

class SuffixTree:
    def __init__(self, text):
        self.text = text
        self.root = SuffixTreeNode(-1, -1)
        self.root.suffix_link = self.root
        self.active_node = self.root
        self.active_edge = -1
        self.active_length = 0
        self.leaf_end = -1
        self.remainder = 0
        self.text_length = len(text)
        self.active_edge_index = 0

        for i in range(self.text_length):
            self.extend_suffix_tree(i)

    def extend_suffix_tree(self, idx):
        self.leaf_end = idx
        self.remainder += 1
        last_created_node = None

        while self.remainder > 0:
            if self.active_length == 0:
                self.active_edge = idx

            if self.active_edge not in self.active_node.children:
                leaf_node = SuffixTreeNode(idx, self.text_length - 1)
                self.active_node.children[self.active_edge] = leaf_node

                if last_created_node is not None:
                    last_created_node.suffix_link = self.active_node

                last_created_node = None
            else:
                next_node = self.active_node.children[self.active_edge]
                edge_length = next_node.edge_length(idx)

                if self.active_length >= edge_length:
                    self.active_edge += edge_length
                    self.active_length -= edge_length
                    self.active_node = next_node
                    continue

                if self.text[next_node.start + self.active_length] == self.text[idx]:
                    self.active_length += 1

                    if last_created_node is not None and self.active_node != self.root:
                        last_created_node.suffix_link = self.active_node

                    break

                split_node = SuffixTreeNode(next_node.start, next_node.start + self.active_length - 1)
                next_node.start += self.active_length
                split_node.children[self.text[next_node.start]] = next_node

                leaf_node = SuffixTreeNode(idx, self.text_length - 1)
                split_node.children[self.text[idx]] = leaf_node

                if last_created_node is not None:
                    last_created_node.suffix_link = split_node

                last_created_node = split_node

                if self.active_node == self.root:
                    self.active_length -= 1
                    self.active_edge = idx - self.remainder + 1
                else:
                    self.active_node = self.active_node.suffix_link

            self.remainder -= 1

    def search(self, pattern):
        current_node = self.root

        for i in range(len(pattern)):
            char = pattern[i]

            if char not in current_node.children:
                return False

            next_node = current_node.children[char]

            if i == len(pattern) - 1:
                return True

            j = next_node.start
            while j <= next_node.end and i < len(pattern):
                if self.text[j] != pattern[i]:
                    return False
                i += 1
                j += 1

            if i < len(pattern):
                current_node = next_node

        return False

#Splay Tree Search Algorithm
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class SplayTree:
    def __init__(self):
        self.root = None

    def search(self, key):
        node = self._search(key)

        if node is not None:
            self._splay(node)

        return node is not None

    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
            return

        node = self._search(key)

        if node.key == key:
            self._splay(node)
            return

        new_node = Node(key)

        if key < node.key:
            new_node.left = node.left
            new_node.right = node
            node.left = None
        else:
            new_node.right = node.right
            new_node.left = node
            node.right = None

        self.root = new_node

    def _search(self, key):
        node = self.root

        while node is not None:
            if key < node.key:
                if node.left is None:
                    break
                node = node.left
            elif key > node.key:
                if node.right is None:
                    break
                node = node.right
            else:
                break

        return node

    def _splay(self, node):
        while node != self.root:
            parent = node.parent
            grandparent = parent.parent

            if grandparent is None:
                if node == parent.left:
                    self._rotate_right(parent)
                else:
                    self._rotate_left(parent)
            elif node == parent.left and parent == grandparent.left:
                self._rotate_right(grandparent)
                self._rotate_right(parent)
            elif node == parent.right and parent == grandparent.right:
                self._rotate_left(grandparent)
                self._rotate_left(parent)
            elif node == parent.right and parent == grandparent.left:
                self._rotate_left(parent)
                self._rotate_right(grandparent)
            else:
                self._rotate_right(parent)
                self._rotate_left(grandparent)

    def _rotate_right(self, node):
        parent = node.parent
        grandparent = parent.parent
        new_parent = node
        new_right = node.right

        new_parent.right = parent
        parent.parent = new_parent
        parent.left = new_right
        if new_right is not None:
            new_right.parent = parent

        new_parent.parent = grandparent
        if grandparent is not None:
            if grandparent.left == parent:
                grandparent.left = new_parent
            else:
                grandparent.right = new_parent

        if parent == self.root:
            self.root = new_parent

    def _rotate_left(self, node):
        parent = node.parent
        grandparent = parent.parent
        new_parent = node
        new_left = node.left

        new_parent.left = parent
        parent.parent = new_parent
        parent.right = new_left
        if new_left is not None:
            new_left.parent = parent

        new_parent.parent = grandparent
        if grandparent is not None:
            if grandparent.left == parent:
                grandparent.left = new_parent
            else:
                grandparent.right = new_parent

        if parent == self.root:
            self.root = new_parent

#Skip List Search Algorithm
import random

class Node:
    def __init__(self, key, height):
        self.key = key
        self.next = [None] * height

class SkipList:
    def __init__(self):
        self.head = Node(float('-inf'), 1)
        self.tail = Node(float('inf'), 1)
        self.head.next[0] = self.tail
        self.length = 0
        self.max_height = 1

    def search(self, key):
        current = self.head

        for i in reversed(range(self.max_height)):
            while current.next[i] is not None and current.next[i].key < key:
                current = current.next[i]

        current = current.next[0]

        if current is not None and current.key == key:
            return True
        else:
            return False

    def insert(self, key):
        node = Node(key, self._random_height())

        while len(self.head.next) < len(node.next):
            self.head.next.append(self.tail)

        current = self.head

        for i in reversed(range(len(node.next))):
            while current.next[i] is not None and current.next[i].key < node.key:
                current = current.next[i]
            if i < len(node.next):
                node.next[i] = current.next[i]
                current.next[i] = node

        self.length += 1

    def delete(self, key):
        current = self.head

        for i in reversed(range(self.max_height)):
            while current.next[i] is not None and current.next[i].key < key:
                current = current.next[i]

        current = current.next[0]

        if current is not None and current.key == key:
            for i in reversed(range(len(current.next))):
                if current.next[i] is not None:
                    current.next[i] = current.next[i].next[i]

            self.length -= 1

    def _random_height(self):
        height = 1

        while random.random() < 0.5 and height < self.max_height + 1:
            height += 1

        if height > self.max_height:
            self.max_height = height

        return height

#Hash Table Search Algorithm
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_func(self, key):
        return hash(key) % self.size

    def search(self, key):
        index = self.hash_func(key)

        for pair in self.table[index]:
            if pair[0] == key:
                return pair[1]

        raise KeyError(key)

    def insert(self, key, value):
        index = self.hash_func(key)

        for pair in self.table[index]:
            if pair[0] == key:
                pair[1] = value
                return

        self.table[index].append([key, value])

    def delete(self, key):
        index = self.hash_func(key)

        for i, pair in enumerate(self.table[index]):
            if pair[0] == key:
                del self.table[index][i]
                return

        raise KeyError(key)

#Binary Heap Search Algorithm
class BinaryHeap:
    def __init__(self):
        self.heap = []

    def search(self, key):
        for i in range(len(self.heap)):
            if self.heap[i] == key:
                return True

        return False

    def insert(self, key):
        self.heap.append(key)
        self._sift_up(len(self.heap) - 1)

    def delete_min(self):
        if len(self.heap) == 0:
            raise IndexError('Heap is empty')

        min_val = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self._sift_down(0)
        return min_val

    def _sift_up(self, index):
        parent = (index - 1) // 2

        while index > 0 and self.heap[index] < self.heap[parent]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            index = parent
            parent = (index - 1) // 2

    def _sift_down(self, index):
        left = 2 * index + 1
        right = 2 * index + 2
        smallest = index

        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left

        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self._sift_down(smallest)

#Boyer-Moore Algorithm
def boyer_moore(pattern, text):
    pattern_len = len(pattern)
    text_len = len(text)

    if pattern_len > text_len:
        return -1

    # Preprocessing step: build the bad character table
    bad_char = {}
    for i in range(pattern_len):
        bad_char[pattern[i]] = i

    # Search step
    i = 0
    while i <= text_len - pattern_len:
        j = pattern_len - 1

        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1

        if j == -1:
            return i

        # If the bad character is not in the pattern, shift the pattern by the length of the pattern
        # Otherwise, shift the pattern to align the bad character with its last occurrence in the pattern
        if text[i + j] not in bad_char:
            shift = pattern_len
        else:
            shift = j - bad_char[text[i + j]]

        i += shift

    return -1

#Rabin-Karp Algorithm
def rabin_karp(pattern, text):
    pattern_len = len(pattern)
    text_len = len(text)
    pattern_hash = hash(pattern)

    # Calculate the hash of the first substring of the text
    text_hash = hash(text[:pattern_len])

    # Iterate through the text, checking for matches
    for i in range(text_len - pattern_len + 1):
        if text_hash == pattern_hash and text[i:i + pattern_len] == pattern:
            return i

        # Update the hash of the substring starting at i+1
        if i < text_len - pattern_len:
            text_hash = hash(text[i + 1:i + pattern_len + 1])

    return -1

#Knuth-Morris-Pratt Algorithm
def kmp(pattern, text):
    pattern_len = len(pattern)
    text_len = len(text)

    # Preprocessing step: build the failure function
    failure = [0] * pattern_len
    j = 0
    for i in range(1, pattern_len):
        while j > 0 and pattern[j] != pattern[i]:
            j = failure[j-1]
        if pattern[j] == pattern[i]:
            j += 1
        failure[i] = j

    # Search step
    j = 0
    for i in range(text_len):
        while j > 0 and pattern[j] != text[i]:
            j = failure[j-1]
        if pattern[j] == text[i]:
            j += 1
        if j == pattern_len:
            return i - pattern_len + 1

    return -1

#Z-Algorithm
def z_algorithm(pattern, text):
    concatenated_string = pattern + '#' + text
    z_values = [0] * len(concatenated_string)
    left = 0
    right = 0
    for i in range(1, len(concatenated_string)):
        if i <= right:
            z_values[i] = min(right - i + 1, z_values[i - left])
        while i + z_values[i] < len(concatenated_string) and concatenated_string[z_values[i]] == concatenated_string[i + z_values[i]]:
            z_values[i] += 1
        if i + z_values[i] - 1 > right:
            left = i
            right = i + z_values[i] - 1

    for i in range(len(z_values)):
        if z_values[i] == len(pattern):
            return i - len(pattern) - 1

    return -1

#String Matching with Finite Automata Algorithm
def finite_automaton(pattern, text):
    # Preprocessing step: build the transition function
    alphabet = set(text)
    transition = {}
    for q in range(len(pattern) + 1):
        for a in alphabet:
            k = min(len(pattern) + 1, q + 2)
            while k > 0 and pattern[:k-1] + a not in pattern[:q] + a:
                k -= 1
            transition[(q, a)] = k

    # Search step
    q = 0
    for i in range(len(text)):
        q = transition.get((q, text[i]), 0)
        if q == len(pattern):
            return i - len(pattern) + 1

    return -1

#Aho-Corasick Algorithm
class AhoCorasick:
    def __init__(self, patterns):
        self.patterns = patterns
        self.goto = [{}]
        self.fail = []
        self.out = []
        self.create_automaton()

    def create_automaton(self):
        # Create the trie and the goto function
        for pattern in self.patterns:
            state = 0
            for symbol in pattern:
                if symbol not in self.goto[state]:
                    self.goto[state][symbol] = len(self.goto)
                    self.goto.append({})
                state = self.goto[state][symbol]
            self.out.append([])

        # Create the fail function using BFS
        queue = []
        for symbol, state in self.goto[0].items():
            queue.append(state)
            self.fail.append(0)

        while queue:
            r = queue.pop(0)
            for symbol, s in self.goto[r].items():
                queue.append(s)
                state = self.fail[r]
                while symbol not in self.goto[state] and state != 0:
                    state = self.fail[state]
                self.fail[s] = self.goto[state].get(symbol, 0)
                self.out[s] = self.out[s] + self.out[self.fail[s]]

    def search(self, text):
        # Use the automaton to find all matches in the text
        state = 0
        matches = []
        for i, symbol in enumerate(text):
            while symbol not in self.goto[state] and state != 0:
                state = self.fail[state]
            state = self.goto[state].get(symbol, 0)
            matches.extend([(i-len(pattern)+1, pattern) for pattern in self.out[state]])
        return matches

#Quick Search Algorithm
def quick_search(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0:
        return 0
    if m > n:
        return -1

    # Preprocess the pattern
    skip = [m] * 256
    for i in range(m-1):
        skip[ord(pattern[i])] = m - i - 1

    # Search for the pattern in the text
    i = m - 1
    while i < n:
        k = i
        j = m - 1
        while j >= 0 and text[k] == pattern[j]:
            j -= 1
            k -= 1
        if j == -1:
            return k + 1
        i += skip[ord(text[i])]
    return -1

#Baeza-Yates-Gonnet Algorithm
def baeza_yates_gonnet(query, target, match_score=2, mismatch_score=-1, gap_score=-1):
    # Initialize the score matrix and the traceback matrix
    n = len(query)
    m = len(target)
    score = [[0 for j in range(m+1)] for i in range(n+1)]
    traceback = [[0 for j in range(m+1)] for i in range(n+1)]

    # Fill in the score matrix
    max_score = -1
    max_i, max_j = -1, -1
    for i in range(1, n+1):
        for j in range(1, m+1):
            match = score[i-1][j-1] + (match_score if query[i-1] == target[j-1] else mismatch_score)
            delete = score[i-1][j] + gap_score
            insert = score[i][j-1] + gap_score
            score[i][j] = max(0, match, delete, insert)
            if score[i][j] > max_score:
                max_score = score[i][j]
                max_i, max_j = i, j

    # Trace back the best alignment
    alignment = ''
    i, j = max_i, max_j
    while i > 0 and j > 0:
        if traceback[i][j] == 0:
            alignment = '-' + alignment
            i -= 1
        elif traceback[i][j] == 1:
            alignment = query[i-1] + alignment
            i -= 1
            j -= 1
        else:
            alignment = target[j-1] + alignment
            j -= 1

    return max_score, alignment

#Bitap Algorithm
def bitap_search(text, pattern):
    # Preprocess the pattern
    m = len(pattern)
    bitmask = [1 << i for i in range(m)]
    hash_table = {}
    for i, c in enumerate(pattern):
        hash_table[c] = hash_table.get(c, 0) | bitmask[i]

    # Search for the pattern
    n = len(text)
    state = 0
    for i in range(n):
        state |= hash_table.get(text[i], 0)
        state <<= 1
        if state & bitmask[-1]:
            return i - m + 1

    return -1

#Interpolating Subdivision Search Algorithm
def interpolation_search(arr, target):
    low, high = 0, len(arr) - 1

    while low <= high and arr[low] <= target <= arr[high]:
        pos = low + (target - arr[low]) * (high - low) // (arr[high] - arr[low])
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1

    return -1
