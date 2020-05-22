import sys
import os
import pstats
import numpy as np
import matplotlib.pyplot as plt

import snakevizcode

from generate_config import write_exp

FN_NAME = 'data_utils.py:104(empty_counter)'

def showx():
    plt.draw()
    plt.pause(1) # <-------
    input("<Hit Enter To Close>")
    plt.close()


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

def plot_slice(dcalls_array, vs_k_or_N, Ns, ks, algo, seeds):
    assert vs_k_or_N == 'N' or vs_k_or_N == 'k', "Bad slice param"

    if vs_k_or_N == 'k':
        kNs = ks
        Nks = Ns
    elif vs_k_or_N == 'N':
        kNs = Ns
        Nks = ks

    for kN_idx, kN in enumerate(kNs):
        if vs_k_or_N == 'k':
            plt.title(algo + " scaling with N for k = " + str(kN))
        elif vs_k_or_N == 'N':
            plt.title(algo + " scaling with k for N = " + str(kN))
        plt.xlabel("N")
        for seed_idx, seed in enumerate(seeds):
            if vs_k_or_N == 'k':
                plt.plot(Nks, dcalls_array[kN_idx, :, seed_idx], 'o')
            elif vs_k_or_N == 'N':
                plt.plot(Nks, dcalls_array[:, kN_idx, seed_idx], 'o')
        if vs_k_or_N == 'k':
            plt.plot(Nks, np.mean(dcalls_array[kN_idx, :, :], axis = 1), 'b-') # Slice a specific k, get a 2D array
        elif vs_k_or_N == 'N':
            plt.plot(Nks, np.mean(dcalls_array[:, kN_idx, :], axis = 1), 'b-') # Slice a specific N, get a 2D array
        showx()

def show_plots(vs_k_or_N, build_or_swap, Ns, ks, seeds, algos, dataset, metric):
    dcalls_array = np.zeros((len(ks), len(Ns), len(seeds)))

    if build_or_swap == 'build':
        prefix = 'profiles/p-B-'
    elif build_or_swap == 'swap':
        prefix = 'profiles/p-S-'
    else:
        raise Exception("Error pi")

    log_prefix = 'L-'

    # Gather data
    for algo in algos:
        assert algo in ['ucb', 'naive_v1'], "Bad algo yo"
        for N_idx, N in enumerate(Ns):
            for k_idx, k in enumerate(ks):
                for seed_idx, seed in enumerate(seeds):
                    profile_fname = prefix + algo + '-True-BS-v-0-k-' + str(k) + \
                        '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'
                    if os.path.exists(profile_fname):
                        p = pstats.Stats(profile_fname)
                        for row in snakevizcode.table_rows(p):
                            if FN_NAME in row:
                                dcalls = row[0][1]
                                dcalls_array[k_idx][N_idx][seed_idx] = dcalls
                    else:
                        print("Warning: profile not found for ", profile_fname)

    # Show data
    for algo in algos:
        plot_slice(dcalls_array, 'k', Ns, ks, algo, seeds)
        plot_slice(dcalls_array, 'N', Ns, ks, algo, seeds)

def main():
    algos = ['ucb']#, 'naive_v1']
    dataset = 'MNIST'
    metric = 'L2'

    Ns = [1000, 3000, 10000, 30000, 70000]
    ks = [2, 3, 4]#, 5, 10, 20, 30]
    seeds = range(4)

    build_profiles = [os.path.join('profiles', x) for x in os.listdir('profiles') if os.path.isfile(os.path.join('profiles', x)) and x != '.DS_Store' and x[:2] == 'p-B']
    swap_profiles = [os.path.join('profiles', x) for x in os.listdir('profiles') if os.path.isfile(os.path.join('profiles', x)) and x != '.DS_Store' and x[:2] == 'p-S']

    show_plots('k', 'build', Ns, ks, seeds, algos, dataset, metric)



if __name__ == '__main__':
    verify_logfiles()
    main()
