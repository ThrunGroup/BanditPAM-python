'''
Convenience code to automatically generate a list of experiments to run.
Default output is to auto_exp_config.py.
'''

import itertools

def write_exp(algo, k, N, seed, dataset, metric):
    '''
    Takes the experiment variables and outputs a string description
    to go into a config file.
    '''
    if algo == 'naive_v1' and (N > 10000 or k > 100):
        return None
    return "\t['" + algo + "', 'BS', 0, " + str(k) + ", " + str(N) + \
        ", " + str(seed) + ", '" + dataset + "', '" + metric + "', ''],\n"

def main():
    # TODO(@Adarsh321123): change comments throughout
    # TODO(@Adarsh321123): remove unnecessary things
    # Possible algos are ['ucb', 'naive_v1', 'em_style', 'csh', and 'clarans']
    # algos = ['naive_v1']
    # seeds = range(10)

    ####### MNIST, L2 distance, k = 5 and k = 10
    # dataset = 'MNIST'
    # Ns = [1000, 10000, 20000, 40000, 70000]
    # ks = [5, 10]
    # metric = 'L2'

    ######## MNIST, Cosine distance, k = 5
    # dataset = 'MNIST'
    # Ns = [3000, 10000, 20000, 40000, 70000]
    # ks = [5]
    # metric = 'COSINE'

    ######## SCRNA, L1 distance, k = 5
    # dataset = 'SCRNA'
    # Ns = [3000, 10000, 20000, 30000, 40000]
    # ks = [5]
    # metric = 'L1'

    ######## SCRNAPCA, L2 distance, k = 5 and k = 10
    # dataset = 'SCRNAPCA'
    # Ns = [3000, 10000, 20000, 30000, 40000]
    # ks = [5, 10]
    # metric = 'L2'

    ######## HOC4, Tree edit distance (precomputed), k = 2 and k = 3
    algos = ['ucb']
    dataset = 'HOC4'
    metric = 'PRECOMP'
    Ns = [1000, 2000, 3000, 3360]
    ks = [2, 3]
    seeds = range(10)

    ######## For loss plots
    # algos = ['ucb']
    # seeds = range(10)
    # Ns = [500, 1000, 1500, 2000, 2500, 3000]
    # ks = [5]

    with open('auto_exp_config.py', 'w+') as fout:
        fout.write("experiments = [\n")
        for k in ks:
            for seed in seeds:
                for N in Ns:
                    for algo in algos:
                        # Adding 42 to seed for comparison with earlier experiments
                        exp = write_exp(algo, k, N, 42 + seed, dataset, metric)
                        if exp is not None:
                            fout.write(exp)
        fout.write("]")

if __name__ == "__main__":
    main()
