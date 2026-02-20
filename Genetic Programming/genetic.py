import pandas as pd
from tqdm import tqdm
from tree import *
from math import inf
import random

class Genetic:

    def __init__(self, x_label, y_label, is_two_dimension, population_size, num_generations):
        self.x_label = x_label
        self.y_label = y_label
        self.is_two_dimension = is_two_dimension
        self.population_size = population_size
        self.num_generations = num_generations

    def generate_initial_population(self, max_depth):
        trees = []
        for i in range(self.population_size):
            t = Tree(max_depth, self.is_two_dimension)
            t.create_tree(t.max_depth)
            trees.append(t)
        return trees

    def fitness_function(self, chromosome: Tree):
        mse = chromosome.mse(self.x_label, self.y_label)
        if mse == 0:
            return inf
        else:
            return 1 / mse

    def selection(self, fitness_array):
        if fitness_array[len(fitness_array) - 1][0] < 0.4:
            i = random.randint(0, len(fitness_array) - 1)
            return fitness_array[i][1]
        rand = random.random()
        min = 0
        for p in fitness_array:
            if rand <= min + p[0]:
                return p[1]
            else:
                min = p[0]
        return fitness_array[len(fitness_array) - 1][1]

    def crossover(self, chromosome1: Tree, chromosome2: Tree, prob: float = 1):
        rand = random.random()
        if rand > prob:
            return chromosome1, chromosome2
        else:
            node1 = chromosome1.generate_random_node(chromosome1.root)
            node2 = chromosome2.generate_random_node(chromosome2.root)
            parent1 = node1.parent
            node1.parent = node2.parent
            node2.parent = parent1
            if node1.parent == None:
                chromosome1.root = node1
            else:
                index = node1.parent.children.index(node2)
                node1.parent.children[index] = node1
            if node2.parent == None:
                chromosome2.root = node2
            else:
                index = node2.parent.children.index(node1)
                node2.parent.children[index] = node2
            return chromosome1, chromosome2

    def mutation(self, chromosome: Tree, prob: float = 0.3):
        rand = random.random()
        if rand > prob:
            return chromosome
        else:
            node = chromosome.generate_random_node(chromosome.root)
            if node.is_leaf:
                if self.is_two_dimension:
                    node.value = leaf_two_dimension()
                else:
                    node.value = leaf_one_dimension()
            else:
                if single_operation(node.value):
                    if node.value == 'sin':
                        node.value = 'cos'
                    else:
                        node.value = 'sin'
                else:
                    all_op = list(set(['+', '-', '*', '/', '^']) - set([node.value]))
                    node.value = random.choice(all_op)
            return chromosome

    def run_algorithm(self):
        best_cost = -1
        best_solution = None
        records = []

        trees = self.generate_initial_population(3)

        for i in tqdm(range(self.num_generations)):
            fitness_array = []
            iter_best_cost = -1
            iter_best_solution = None
            for chromosome in trees:
                fit = self.fitness_function(chromosome)
                if fit > iter_best_cost:
                    iter_best_cost = fit
                    iter_best_solution = chromosome
                fitness_array.append([fit, chromosome])
            records.append({'iteration': i + 1, 'best_cost': iter_best_cost,
                           'best_solution': iter_best_solution})
            if iter_best_cost > best_cost:
                best_cost = iter_best_cost
                best_solution = iter_best_solution
            if best_cost == math.inf:
                print(best_solution.__str__(best_solution.root))
                break
            trees = []
            sorted_fitness = sorted(fitness_array, key=lambda x: x[0])
            sum_fitness = 0
            for f in sorted_fitness:
                sum_fitness += f[0]
            for f in range(len(sorted_fitness)):
                sorted_fitness[f][0] = sorted_fitness[f][0] / sum_fitness
            for j in range(self.population_size // 2):
                chromosome1 = self.selection(sorted_fitness)
                chromosome2 = self.selection(sorted_fitness)
                chromosome1, chromosome2 = self.crossover(chromosome1, chromosome2)
                trees.append(chromosome1)
                trees.append(chromosome2)
            

        records = pd.DataFrame(records)
        return best_cost, best_solution, records