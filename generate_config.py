import itertools

def write_exp(algo, k, N, seed, dataset, metric):
    if algo == 'naive_v1' and (N > 10000 or k > 5):
        return None
    return "\t['" + algo + "', 'BS', 0, " + str(k) + ", " + str(N) + \
        ", " + str(seed) + ", '" + dataset + "', '" + metric + "', ''],\n"

def main():
    algos = ['ucb', 'naive_v1']
    dataset = 'MNIST'
    metric = 'L2'

    Ns = [1000, 3000, 10000, 30000, 70000]
    ks = [2, 3, 4, 5, 10, 20, 30]
    seeds = range(10)

    with open('auto_exp_config.py', 'w+') as fout:
        fout.write("experiments = [\n")
        for algo in algos:
            for seed in seeds:
                for N in Ns:
                    for k in ks:
                        exp = write_exp(algo, k, N, seed, dataset, metric)
                        if exp is not None:
                            fout.write(exp)
        fout.write("]")

if __name__ == "__main__":
    main()
