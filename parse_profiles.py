'''
TODO: Need to spot check this file's results, but the plots look ok
'''

import sys
import os
import pstats
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def plot_slice(dcalls_array, fix_k_or_N, Ns, ks, algo, seeds, build_or_swap, take_log = True):
    assert fix_k_or_N == 'N' or fix_k_or_N == 'k', "Bad slice param"

    if fix_k_or_N == 'k':
        kNs = ks
        Nks = Ns
    elif fix_k_or_N == 'N':
        kNs = Ns
        Nks = ks

    for kN_idx, kN in enumerate(kNs):
        if fix_k_or_N == 'k':
            if take_log:
                np_data = np.log10(dcalls_array)
                Nks_plot = np.log10(Nks)
            else:
                np_data = dcalls_array
                Nks_plot = Nks

            plt.title(algo + " " + build_or_swap.upper() + " scaling with N for k = " + str(kN))
            plt.xlabel("N")
            means = np.mean(np_data[kN_idx, :, :], axis = 1)
            plt.plot(Nks_plot, means, 'b-') # Slice a specific k, get a 2D array
            for seed_idx, seed in enumerate(seeds):
                plt.plot(Nks_plot, np_data[kN_idx, :, seed_idx], 'o')
                print(np_data[kN_idx, :, seed_idx])

            bars = 1.96 * np.std(np_data[kN_idx, :, :], axis = 1) # Slice a specific k, get a 2D array
            # plt.errorbar(Nks, np.mean(np_data[kN_idx, :, :], axis = 1), yerr = bars, ecolor='red', elinewidth=3, zorder = 100)
            plt.errorbar(Nks_plot, means, yerr = bars,  fmt='+', capsize=3, elinewidth=2, markeredgewidth=2, color='black', label='Nominal 95% CI', zorder=100)
            print("Summary:")
            print(Nks_plot, means)

        elif fix_k_or_N == 'N':
            raise Exception("Need to update with above")
            plt.title(algo + " " + build_or_swap.upper() + " scaling with k for N = " + str(kN))
            plt.xlabel("k")
            plt.xticks(np.arange(0, 110, 10))
            plt.plot(Nks, np.mean(np_data[:, kN_idx, :], axis = 1), 'b-') # Slice a specific N, get a 2D array
            for seed_idx, seed in enumerate(seeds):
                plt.plot(Nks, np_data[:, kN_idx, seed_idx], 'o')
                print(np_data[:, kN_idx, seed_idx])

        showx()

def plot_slice_sns(dcalls_array, fix_k_or_N, Ns, ks, algo, seeds, build_or_swap, take_log = True):
    assert fix_k_or_N == 'N' or fix_k_or_N == 'k', "Bad slice param"

    if fix_k_or_N == 'k':
        kNs = ks
        Nks = Ns
    elif fix_k_or_N == 'N':
        kNs = Ns
        Nks = ks

    for kN_idx, kN in enumerate(kNs):
        if fix_k_or_N == 'k':
            if take_log:
                np_data = np.log10(dcalls_array)
                Nks_plot = np.log10(Nks)
            else:
                np_data = dcalls_array
                Nks_plot = Nks

            sns.set()
            sns.set_style('white')

            fig, ax = plt.subplots(figsize = (7,7))

            d = {'N': Nks_plot}#, 'avg_d_calls': np.mean(np_data[kN_idx, :, :], axis = 1)}
            for seed_idx, seed in enumerate(seeds):
                d["seed_" + str(seed)] = np_data[kN_idx, :, seed_idx]
            df = pd.DataFrame(data = d)

            melt_df = df.melt('N', var_name='cols', value_name='vals')
            melt_df['N'] += np.random.randn(melt_df['N'].shape[0]) * 0.04 # Add jitter
            sns.scatterplot(x="N", y="vals", data = melt_df, ax = ax, alpha = 0.6)
            # sns.scatterplot(x="N", y="avg_d_calls", data = df, ax = ax)

            bars = 1.96 * np.std(np_data[kN_idx, :, :], axis = 1) # Slice a specific k, get a 2D array
            means = np.mean(np_data[kN_idx, :, :], axis = 1)
            plt.errorbar(Nks_plot, means, yerr = bars, fmt = '+', capsize = 5, ecolor='black', elinewidth = 1.5, zorder = 100, mec='black', mew = 1.5)


            sl, icpt, r_val, p_val, _ = sp.stats.linregress(Nks_plot, means)
            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()
            plt.plot([x_min, x_max], [x_min * sl + icpt, x_max * sl + icpt], color='black', label='Linear fit with\n95%% confidence intervals\nslope=%0.3f'%(sl))
            print("Slope is:", sl)
            plt.legend(loc="upper left")
            # plt.xticks(Nks_plot.tolist(), ['10^3, 3*10^3, 10^4, 3*10^4, 7*10^4'])
            # locs, labels = plt.xticks()
            # plt.grid()

        elif fix_k_or_N == 'N':
            raise Exception("Fill this in")

        plt.xlabel("logN (base 10)")
        plt.ylabel("log(# distance computations) (base 10)")
        showx()
        # plt.savefig(algo + " " + build_or_swap.upper() + " scaling with N for k = " + str(kN) + '.pdf')

def get_swap_T(logfile):
    with open(logfile, 'r') as fin:
        line = fin.readline()
        while line[:10] != 'Num Swaps:':
            line = fin.readline()

        T = int(line.split(' ')[-1])
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
        plot_slice_sns(dcalls_array, fix_k_or_N, Ns, ks, algo, seeds, build_or_swap)

def main():
    algos = ['ucb']#, 'naive_v1']
    dataset = 'SCRNAPCA'
    metric = 'L2'

    Ns = [1000, 3000, 10000, 20000, 30000, 40000]
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
