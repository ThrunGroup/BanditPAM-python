'''
TODO: Need to spot check this file's results, but the plots look ok
'''

import sys
import os
import pstats
import numpy as np
import matplotlib.pyplot as plt

import snakevizcode

from generate_config import write_exp

FN_NAME = 'data_utils.py:129(empty_counter)'

def showx():
    plt.draw()
    plt.pause(1) # <-------
    input("<Hit Enter To Close>")
    plt.close()


def verify_logfiles():
    ucb_logfiles = [os.path.join('profiles', x) for x in os.listdir('profiles') if os.path.isfile(os.path.join('profiles', x)) and x != '.DS_Store' and x[:5] == 'L-ucb']
    for u_lfile in sorted(ucb_logfiles):
        n_lfile = u_lfile.replace('ucb', 'naive_v1')
        if not os.path.exists(n_lfile):
            print("Warning: no naive experiment", n_lfile)
        else:
            disagreement = False
            with open(u_lfile, 'r') as fin1:
                with open(n_lfile, 'r') as fin2:
                    l1_1 = fin1.readline()
                    l1_2 = fin1.readline()

                    l2_1 = fin2.readline()
                    l2_2 = fin2.readline()

                    # NOTE: This is a stricter condition than necessary, enforcing both build and swap agreement instead of just swap
                    if l1_1 != l2_1 or l1_2 != l2_2:
                        disagreement = True

            if disagreement:
                print("\n")
                print(l1_2.strip())
                print(l2_2.strip())
                print("ERROR: Results for", u_lfile, n_lfile, "disagree!!")
            else:
                print("OK: Results for", u_lfile, n_lfile, "agree")


def plot_slice(dcalls_array, fix_k_or_N, Ns, ks, algo, seeds, build_or_swap):
    assert fix_k_or_N == 'N' or fix_k_or_N == 'k', "Bad slice param"

    if fix_k_or_N == 'k':
        kNs = ks
        Nks = Ns
    elif fix_k_or_N == 'N':
        kNs = Ns
        Nks = ks

    for kN_idx, kN in enumerate(kNs):
        if fix_k_or_N == 'k':
            plt.title(algo + " " + build_or_swap.upper() + " scaling with N for k = " + str(kN))
            plt.xlabel("N")
            plt.plot(Nks, np.mean(dcalls_array[kN_idx, :, :], axis = 1), 'b-') # Slice a specific k, get a 2D array
            for seed_idx, seed in enumerate(seeds):
                plt.plot(Nks, dcalls_array[kN_idx, :, seed_idx], 'o')
                print(dcalls_array[kN_idx, :, seed_idx])

        elif fix_k_or_N == 'N':
            plt.title(algo + " " + build_or_swap.upper() + " scaling with k for N = " + str(kN))
            plt.xlabel("k")
            plt.xticks(np.arange(0, 110, 10))
            plt.plot(Nks, np.mean(dcalls_array[:, kN_idx, :], axis = 1), 'b-') # Slice a specific N, get a 2D array
            for seed_idx, seed in enumerate(seeds):
                plt.plot(Nks, dcalls_array[:, kN_idx, seed_idx], 'o')
                print(dcalls_array[:, kN_idx, seed_idx])

        showx()

def get_swap_T(logfile):
    '''
    Hacky
    '''
    with open(logfile, 'r') as fin:
        line = fin.readline()
        while line != 'Swap Logstring:\n':
            line = fin.readline()

        line = fin.readline()
        # WARNING: CONDITION IS BROKEN -- need to update!
        # Hacky patch to make sure it has letters
        assert line.strip(':').lower().islower(), "Line is actually:" + line

        T = 0
        line = fin.readline()
        # WARNING: CONDITION IS BROKEN -- need to update
        # Hacky patch to make sure it has letters
        while not line.strip(':').lower().islower():
            T += 1
            line = fin.readline()
    return T

def show_plots(fix_k_or_N, build_or_swap, Ns, ks, seeds, algos, dataset, metric):
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
                                if build_or_swap == 'build':
                                    dcalls_array[k_idx][N_idx][seed_idx] = dcalls
                                elif build_or_swap == 'swap':
                                    logfile = 'profiles/L-' + algo + '-True-BS-v-0-k-' + str(k) + \
                                        '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'
                                    T = get_swap_T(logfile)
                                    dcalls_array[k_idx][N_idx][seed_idx] = dcalls / T
                    else:
                        print("Warning: profile not found for ", profile_fname)

    # Show data
    for algo in algos:
        plot_slice(dcalls_array, fix_k_or_N, Ns, ks, algo, seeds, build_or_swap)

def main():
    algos = ['ucb']#, 'naive_v1']
    dataset = 'MNIST'
    metric = 'L2'

    Ns = [1000, 3000, 10000, 30000, 70000]
    # ks = [2, 3, 4, 5, 10, 20, 30]

    # Ns = [1000]
    # ks = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]#, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    ks = [10]
    seeds = range(42, 52)

    # By calling these functions twice, we're actually mining the data from the profiles twice.
    # Not a big deal but should fix
    show_plots('k', 'build', Ns, ks, seeds, algos, dataset, metric)
    show_plots('k', 'swap', Ns, ks, seeds, algos, dataset, metric)
    # show_plots('N', 'build', Ns, ks, seeds, algos, dataset, metric)
    # show_plots('N', 'swap', Ns, ks, seeds, algos, dataset, metric)



if __name__ == '__main__':
    verify_logfiles()
    print("FILES VERIFIED\n\n")
    main()
