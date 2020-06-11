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
    if algo == 'naive_v1' and (N > 10000 or k > 5):

        return None
    return "\t['" + algo + "', 'BS', 0, " + str(k) + ", " + str(N) + \
        ", " + str(seed) + ", '" + dataset + "', '" + metric + "', ''],\n"

def main():
    # Possible algos are ['ucb', 'naive_v1', 'em_style', 'csh', and 'clarans']
    algos = ['ucb']
    dataset = 'SCRNA'
    metric = 'L1'

    # Ns = [3000, 10000, 20000, 40000, 70000] # For MNIST
    Ns = [3000, 10000, 20000, 30000, 40000] # for SCRNA and SCRNA-PCA and

    ks = [5]
    seeds = range(6, 8)


    # Ns = [1000, 2000, 3000, 3360] # for HOC4
    # ks = [2] $ for HOC4

    ##### For loss plots
    # algos = ['clarans', 'em_style']#, 'ucb']#, 'naive_v1']
    # seeds = range(10)
    # Ns = [10000, 30000, 70000]
    # ks = [5]

    with open('auto_exp_config.py', 'w+') as fout:
        fout.write("experiments = [\n")
        for seed in seeds:
            for N in Ns:
                for algo in algos:
                    for k in ks:
                        # Adding 42 to seed for comparison with earlier experiments
                        exp = write_exp(algo, k, N, 42 + seed, dataset, metric)
                        if exp is not None:
                            fout.write(exp)
        fout.write("]")

if __name__ == "__main__":
    main()
