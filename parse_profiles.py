import sys
import os
import pstats
import numpy as np
import matplotlib.pyplot as plt

import snakevizcode

from generate_config import write_exp

FN_NAME = 'data_utils.py:104(empty_counter)'

def verify_logfiles():
    ucb_logfiles = [os.path.join('profiles', x) for x in os.listdir('profiles') if os.path.isfile(os.path.join('profiles', x)) and x != '.DS_Store' and x[:5] == 'L-ucb']
    for u_lfile in ucb_logfiles:
        n_lfile = u_lfile.replace('ucb', 'naive_v1')
        if not os.path.exists(n_lfile):
            print("Warning: no naive experiment", n_lfile)
        else:
            with open(u_lfile, 'r') as fin1:
                with open(n_lfile, 'r') as fin2:
                    for i in range(2): # Verify that the top two lines, built medoids and swap medoids, are equal
                        if fin1.readline() != fin2.readline():
                            print("ERROR: Results for", u_lfile, "disagree!!")
                    print("OK: Results for", u_lfile, "agree")

def main():
    algos = ['ucb', 'naive_v1']
    dataset = 'MNIST'
    metric = 'L2'

    Ns = [1000, 3000, 10000]#, 30000, 70000]
    ks = [2]#, 3, 4, 5, 10, 20, 30]
    seeds = range(1)

    build_profiles = [os.path.join('profiles', x) for x in os.listdir('profiles') if os.path.isfile(os.path.join('profiles', x)) and x != '.DS_Store' and x[:2] == 'p-B']
    swap_profiles = [os.path.join('profiles', x) for x in os.listdir('profiles') if os.path.isfile(os.path.join('profiles', x)) and x != '.DS_Store' and x[:2] == 'p-S']

    ucb_build = np.zeros((len(ks), len(Ns), len(seeds)))
    ucb_swap = np.zeros((len(ks), len(Ns), len(seeds)))

    naive_build = np.zeros((len(ks), len(Ns), len(seeds)))
    naive_swap = np.zeros((len(ks), len(Ns), len(seeds)))

    for algo in algos:
        for N_idx, N in enumerate(Ns):
            for k_idx, k in enumerate(ks):
                for seed_idx, seed in enumerate(seeds):
                    # WARNING: Brittle
                    # WARNING: Remove the +42!!!!
                    build_profile = 'profiles/p-B-' + algo + '-True-BS-v-0-k-' + str(k) + \
                        '-N-' + str(N) + '-s-' + str(seed + 42) + '-d-' + dataset + '-m-' + metric + '-w-'
                    if os.path.exists(build_profile):
                        p = pstats.Stats(build_profile)
                        for row in snakevizcode.table_rows(p):
                            if FN_NAME in row:
                                d_calls = row[0][1]
                                if algo == 'ucb':
                                    ucb_build[k_idx][N_idx][seed_idx] = d_calls
                                elif algo == 'naive_v1':
                                    naive_build[k_idx][N_idx][seed_idx] = d_calls
                                else:
                                    raise Exception("bad algo yo")
    print(ucb_build)
    for k_idx, k in enumerate(ks):
        print(np.mean(ucb_build[k_idx], axis = 1))
        plt.plot(Ns, np.mean(ucb_build[k_idx], axis = 1))
        plt.show()
    print(naive_build)
    for k_idx, k in enumerate(ks):
        print(np.mean(naive_build[k_idx], axis = 1))
        plt.plot(Ns, np.mean(naive_build[k_idx], axis = 1))
        plt.show()

if __name__ == '__main__':
    verify_logfiles()
    main()
