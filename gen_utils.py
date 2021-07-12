import numpy as np
import random
import tensorflow as tf


def init_pop(pop_size, n_genes):
    pop = np.random.uniform(0.0, 1.0, (pop_size, n_genes))
    pop = (pop > 0.9).astype(float)
    print(pop)
    return pop


def update_pop(pop, genes):
    new_pop = np.append(pop, genes, axis=0)
    return new_pop


def uniform_crossover(x_genes, y_genes, gene_length):
    mask = np.random.uniform(0.0, 1.0, gene_length)
    mask = (mask > 0.5).astype(int)
    # print('mask: ', mask)
    x_child = np.squeeze(np.zeros((1, len(mask)))).astype(int)
    y_child = np.squeeze(np.zeros((1, len(mask)))).astype(int)

    for i in range(len(mask)):
        if int(mask[i]) == 0:
            x_child[i] = y_genes[i]
            y_child[i] = x_genes[i]
        elif int(mask[i]) == 1:
            x_child[i] = x_genes[i]
            y_child[i] = y_genes[i]
    # print('x_child: ', x_child, '\ny_child: ', y_child)

    return x_child, y_child


def fitness(pop, k, truth):
    pop_mse = []
    for member in pop:
        mse = tf.keras.losses.MeanSquaredError()
        error = -(mse(truth, member).numpy())
        pop_mse.append(error)

    # print("error", pop_mse)
    result = tf.math.top_k(pop_mse, k)
    values = result.values.numpy()
    indices = result.indices.numpy()
    # print(values, indices)
    return values, indices


def point_mutation(pop, rate):
    for i in range(int(rate)):
        pop_idx = random.randint(0, (len(pop)-1))
        genebit_idx = random.randint(0, (len(pop[pop_idx])-1))
        if pop[pop_idx][genebit_idx] == 0:
            pop[pop_idx][genebit_idx] = 1
        elif pop[pop_idx][genebit_idx] == 1:
            pop[pop_idx][genebit_idx] = 0

    return pop


# test based on uniform crossover example given on page 254 of Machine Learning by Tom M. Mitchel
test1 = [1,1,1,0,1,0,0,1,0,0,0]
test2 = [0,0,0,0,1,0,1,0,1,0,1]
# uniform_crossover(test1, test2)


