import itertools

def write_exp(algo, k, N, seed, dataset, metric):
    if algo == 'naive_v1' and (N > 10000 or k > 5):
        return None
    return "\t['" + algo + "', 'BS', 0, " + str(k) + ", " + str(N) + \
        ", " + str(seed) + ", '" + dataset + "', '" + metric + "', ''],\n"

def main():
    # algos = ['ucb']#, 'naive_v1']
    dataset = 'MNIST'
    metric = 'L2'

    # Ns = [3000, 10000, 20000, 40000, 70000] # For MNIST
    # Ns = [3000, 10000, 20000, 30000, 40000] # for SCRNA and SCRNA-PCA and
    # ks = [5]
    # seeds = range(10)


    # Ns = [1000, 2000, 3000, 3360] # for HOC4
    # ks = [2] $ for HOC4

    ##### For loss plots
    algos = ['clarans', 'em_style']#, 'ucb']#, 'naive_v1']
    seeds = range(10)
    Ns = [10000, 30000, 70000]
    ks = [5]

    with open('auto_exp_config.py', 'w+') as fout:
        fout.write("experiments = [\n")
        for seed in seeds:
            for N in Ns: # NOTE: Switched this loop
                for algo in algos: # NOTE: with this loop
                    for k in ks:
                        exp = write_exp(algo, k, N, 42 + seed, dataset, metric) #NOTE: Adding 42 for comparisons with earlier implementations
                        if exp is not None:
                            fout.write(exp)
        fout.write("]")

if __name__ == "__main__":
    main()
