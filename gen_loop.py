from gen_utils import *


def run_ga(pop_size,
           g_len,
           truth,
           r_fraction,
           m_rate,
           fitness_func,
           fitness_threshold):

    top_val = 1
    loop_count = 1
    pop = init_pop(pop_size, g_len)  # initialise population

    while top_val > fitness_threshold:
        p = pop_size
        r = int(((1 - r_fraction) * p) * 1)  # calculate r number of genes
        r_pairs = int((r_fraction * p) / 2)  # calculate number of pairs for crossover
        new_pop = []  # create new generation

        values, indices = fitness_func(pop,  # evaluate fitness of population
                                       r,
                                       truth)
        for indice in indices:  # add fittest genes to new generation
            new_gene = pop[indice]
            new_gene = np.array(new_gene)
            new_pop.append(new_gene)

        cross_genes = []
        top_pairs = tf.math.top_k(values, r_pairs*2)  # select top k fittest pairs from population
        cross_indices = top_pairs.indices.numpy()
        for indice in cross_indices:
            cross_genes.append(new_pop[indice])

        xgenes = cross_genes[:len(cross_genes)//2]
        ygenes = cross_genes[len(cross_genes)//2:]
        for i in range(r_pairs):
            xchild, ychild = uniform_crossover(xgenes[i], ygenes[i], g_len)  # create replacement crossed genes
            new_pop.append(xchild)   # add crossed genes to new generation
            new_pop.append(ychild)
        pop = np.array(new_pop)

        pop = point_mutation(pop, m_rate*10)  # randomly single-point mutate members of pop

        top_val = max(values)  # set top_value to max mse
        top_val = -(top_val)
        print('pop', pop)
        print('Loop Count: ', loop_count, 'Top Val: ', top_val)

        loop_count += 1
    print('Final Val: ', top_val)


if __name__ == '__main__':
    POP_SIZE = 20
    TRUTH = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1]
    G_LEN = 12
    R = 0.2
    M = 0.2
    f_threshold = 0
    f_func = fitness

    run_ga(POP_SIZE, G_LEN, TRUTH, R, M, f_func, f_threshold)
