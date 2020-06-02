import itertools

def write_exp(algo, k, N, seed, dataset, metric):
    if algo == 'naive_v1' and (N > 10000 or k > 5):
        return None
    return "\t['" + algo + "', 'BS', 0, " + str(k) + ", " + str(N) + \
        ", " + str(seed) + ", '" + dataset + "', '" + metric + "', ''],\n"

def main():
    algos = ['ucb']#, 'naive_v1']
    dataset = 'SCRNAPCA'
    metric = 'L2'

    Ns = [10000, 15000, 20000, 25000, 30000, 35000, 40000]
    ks = [10]
    seeds = range(10)

    with open('auto_exp_config.py', 'w+') as fout:
        fout.write("experiments = [\n")
        for seed in seeds:
            for algo in algos:
                for N in Ns:
                    for k in ks:
                        exp = write_exp(algo, k, N, 42 + seed, dataset, metric) #NOTE: Adding 42 for comparisons with earlier implementations
                        if exp is not None:
                            fout.write(exp)
        fout.write("]")

if __name__ == "__main__":
    main()
