import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 200
X_BOUND = [0, 5]         # x upper and lower bounds

pop = np.random.randint(0, 2, (1, DNA_SIZE)).repeat(POP_SIZE, axis=0)  # initialize the pop DNA
pop1 = np.random.randint(0, 2, (1, DNA_SIZE)).repeat(POP_SIZE, axis=0)  # initialize the pop DNA

plt.ion()       # something about plotting
x = np.linspace(*X_BOUND, 200)

def F(x): return np.sin(10*x)*x + np.cos(2*x)*x

def translateDNA(pop): return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / (2**DNA_SIZE-1) * X_BOUND[1]

def get_fitness(pred): return pred + 1e-3 - np.min(pred)

def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


result=F(pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / (2**DNA_SIZE-1) * X_BOUND[1])
print((2**DNA_SIZE-1))
print(pop)
print(2 ** np.arange(DNA_SIZE)[::-1])
print(pop.dot(2 ** np.arange(DNA_SIZE)[::-1]))
print(result)

plt.plot(x,F(x))

F_values = F(translateDNA(pop))    # compute function value by extracting DNA
F_values1 = F(translateDNA(pop1))
print(F_values)
# something about plotting
if 'sca' in globals(): sca.remove()

plt.pause(0.05)
fitness = get_fitness(F_values)
fitness1 = get_fitness(F_values1)
print(fitness)
print(pop[np.argmax(fitness), :])
pop = select(pop, fitness)
pop1 = select(pop1, fitness1)
print(pop)

# for parent in pop:
#     dna=[1,0,1,0,0,1,1,1,1,1]
#     for point in range(DNA_SIZE):
#         parent[point]=dna[point]
#
# print(F(translateDNA(pop)))
sca = plt.scatter(translateDNA(pop), F(translateDNA(pop)), s=200, lw=0, c='red', alpha=0.5)

i_ = np.random.randint(0, POP_SIZE, size=1)[0]
cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
print(i_)
print(cross_points)
print(pop[i_])
print(pop1[i_])
print(pop[i_][cross_points])
print(pop1[i_][cross_points])
print(pop1[i_, cross_points])
pop[i_][cross_points] = pop1[i_, cross_points]
print(pop[:])
plt.ioff()
plt.show()