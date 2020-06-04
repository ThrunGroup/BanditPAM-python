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

FN_NAME_1 = 'data_utils.py:129(empty_counter)'
FN_NAME_2 = 'data_utils.py:141(empty_counter)'
FN_NAME_3 = 'data_utils.py:142(empty_counter)'

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
    raise Exception("This needs to be updated")
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
            print(df)

            melt_df = df.melt('N', var_name='cols', value_name='vals')
            melt_df['N'] += np.random.randn(melt_df['N'].shape[0]) * 0.02 # Add jitter
            sns.scatterplot(x="N", y="vals", data = melt_df, ax = ax, alpha = 0.6)
            # sns.scatterplot(x="N", y="avg_d_calls", data = df, ax = ax)

            bars = 1.96 * np.std(np_data[kN_idx, :, :], axis = 1) # Slice a specific k, get a 2D array
            means = np.mean(np_data[kN_idx, :, :], axis = 1)
            plt.errorbar(Nks_plot, means, yerr = bars, fmt = '+', capsize = 5, ecolor='black', elinewidth = 1.5, zorder = 100, mec='black', mew = 1.5, label="95%% confidence interval")


            sl, icpt, r_val, p_val, _ = sp.stats.linregress(Nks_plot, means)
            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()
            plt.plot([x_min, x_max], [x_min * sl + icpt, x_max * sl + icpt], color='black', label='Linear fit \nslope=%0.3f'%(sl))

            if build_or_swap == 'build':
                # NOTE: kN^2 here in build step
                plt.plot([x_min, x_max], [np.log10(kN) + x_min * 2, np.log10(kN) + x_max * 2], color='red', label='$N^2$ algorithm (est)')
            elif build_or_swap == 'swap':
                # NOTE: N^2 if using FP1 trick and clusters are balanced
                # NOTE: Could also plot kN^2 for when clusters are not balanced
                plt.plot([x_min, x_max], [x_min * 2, x_max * 2], color='red', label='$N^2$ algorithm (est)'%(sl))
            else: # weighted
                plt.plot([x_min, x_max], [np.log10(kN) + x_min * 2, np.log10(kN) + x_max * 2], color='red', label='$N^2$ algorithm (est)'%(sl))

            print("Slope is:", sl)

            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles[::-1], labels[::-1], loc="upper left")

            # plt.xticks(Nks_plot.tolist(), ['10^3, 3*10^3, 10^4, 3*10^4, 7*10^4'])
            # locs, labels = plt.xticks()
            # plt.grid()

        elif fix_k_or_N == 'N':
            raise Exception("Fill this in")

        plt.xlabel("log10(N)")
        plt.ylabel("log10(# of distance computations)")
        plt.title("scRNA-PCA, $d = L2, k = 5$")
        showx()
        # plt.savefig('figures/scRNA-PCA-L2-k-5.pdf')

def get_swap_T(logfile):
    with open(logfile, 'r') as fin:
        line = fin.readline()
        while line[:10] != 'Num Swaps:':
            line = fin.readline()

        T = int(line.split(' ')[-1])
    return T

def show_plots(fix_k_or_N, build_or_swap, Ns, ks, seeds, algos, dataset, metric, dir_):
    dcalls_array = np.zeros((len(ks), len(Ns), len(seeds)))

    if build_or_swap == 'build':
        prefix = 'profiles/' + dir_ + '/p-B-'
    elif build_or_swap == 'swap':
        prefix = 'profiles/' + dir_ + '/p-S-'
    elif build_or_swap == 'weighted' or build_or_swap == 'weighted_T':
        pass
    else:
        raise Exception("Error pi")

    log_prefix = 'profiles/' + dir_ + '/L-'

    # Gather data
    for algo in algos:
        assert algo in ['ucb', 'naive_v1'], "Bad algo yo"
        for N_idx, N in enumerate(Ns):
            for k_idx, k in enumerate(ks):
                for seed_idx, seed in enumerate(seeds):
                    if build_or_swap == 'weighted' or build_or_swap == 'weighted_T':
                        prefix = 'profiles/' + dir_ + '/p-'

                        build_prefix = prefix + 'B-'
                        build_profile_name = build_prefix + algo + '-True-BS-v-0-k-' + str(k) + \
                            '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                        swap_prefix = prefix + 'S-'
                        swap_profile_name = swap_prefix + algo + '-True-BS-v-0-k-' + str(k) + \
                            '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                        logfile = log_prefix + algo + '-True-BS-v-0-k-' + str(k) + \
                            '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'

                        if not os.path.exists(build_profile_name):
                            raise Exception("Warning: profile not found for ", build_profile_name)
                        if not os.path.exists(build_profile_name):
                            raise Exception("Warning: profile not found for ", build_profile_name)
                        if not os.path.exists(logfile):
                            raise Exception("Warning: Log file not found for ", logfile)

                        T = get_swap_T(logfile)

                        b_p = pstats.Stats(build_profile_name)
                        for row in snakevizcode.table_rows(b_p):
                            if FN_NAME_1 in row or FN_NAME_2 in row or FN_NAME_3 in row:
                                if build_or_swap == 'weighted':
                                    dcalls_array[k_idx][N_idx][seed_idx] += row[0][1] # build + avg(swap)
                                elif build_or_swap == 'weighted_T':
                                    dcalls_array[k_idx][N_idx][seed_idx] += row[0][1] / (T + 1) # (build + swap) / (T + 1)
                                else:
                                    raise Exception("blank")

                        s_p = pstats.Stats(swap_profile_name)
                        for row in snakevizcode.table_rows(s_p):
                            if FN_NAME_1 in row or FN_NAME_2 in row or FN_NAME_3 in row:
                                if build_or_swap == 'weighted':
                                    dcalls_array[k_idx][N_idx][seed_idx] += row[0][1] / T # build + avg(swap)
                                elif build_or_swap == 'weighted_T':
                                    dcalls_array[k_idx][N_idx][seed_idx] += row[0][1] / (T + 1) # (build + swap) / (T + 1)
                                else:
                                    raise Exception("blank 2")

                    else:
                        assert build_or_swap == 'build' or build_or_swap == 'swap', "Error with build_or_swap"
                        profile_fname = prefix + algo + '-True-BS-v-0-k-' + str(k) + \
                            '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'
                        if os.path.exists(profile_fname):
                            p = pstats.Stats(profile_fname)
                            for row in snakevizcode.table_rows(p):
                                if FN_NAME_1 in row or FN_NAME_2 in row or FN_NAME_3 in row:
                                    dcalls = row[0][1]
                                    if build_or_swap == 'build':
                                        dcalls_array[k_idx][N_idx][seed_idx] = dcalls
                                    elif build_or_swap == 'swap':
                                        logfile = log_prefix + algo + '-True-BS-v-0-k-' + str(k) + \
                                            '-N-' + str(N) + '-s-' + str(seed) + '-d-' + dataset + '-m-' + metric + '-w-'
                                        T = get_swap_T(logfile)
                                        dcalls_array[k_idx][N_idx][seed_idx] = dcalls / T
                                    else:
                                        raise Exception("Averaging method not supported")
                        else:
                            print("Warning: profile not found for ", profile_fname)

    # Show data
    for algo in algos:
        plot_slice_sns(dcalls_array, fix_k_or_N, Ns, ks, algo, seeds, build_or_swap)

def main():
    algos = ['ucb']#, 'naive_v1']

    # for HOC4
    # dataset = 'HOC4'
    # metric = 'PRECOMP'
    # Ns = [1000, 2000, 3000, 3360]
    # ks = [3]
    # seeds = range(42, 52)
    # dir_ = 'HOC4_paper'

    #for MNIST L2
    # NOTE: Not using all exps since it looks like some didn't complete for higher seeds
    # dataset = 'MNIST'
    # metric = 'L2'
    # Ns = [3000, 10000, 30000, 70000]
    # ks = [10]
    # seeds = range(42, 52)
    # dir_ = 'MNIST_paper'

    #for MNIST COSINE
    # NOTE: not all exps complete
    # dataset = 'MNIST'
    # metric = 'COSINE'
    # Ns = [3000, 10000, 20000, 40000]
    # ks = [5]
    # seeds = range(42, 47)
    # # dir_ = 'MNIST_COSINE_paper'
    # dir_ = 'del_profiles'

    # #for scRNAPCA, L2, K = 10
    # dataset = 'SCRNAPCA'
    # metric = 'L2'
    # Ns = [10000, 20000, 30000, 40000]
    # ks = [10]
    # seeds = range(42, 52)
    # dir_ = 'SCRNAPCA_L2_k-10_paper' # NOTE: SCRNA_PCA_paper_more_some_incomplete contains data for some more values of N.

    # #for scRNAPCA, L2, K = 5
    # #NOTE: Not all experiments are done
    # dataset = 'SCRNAPCA'
    # metric = 'L2'
    # Ns = [10000, 20000, 30000, 40000]
    # ks = [5]
    # seeds = range(42, 45)
    # # dir_ = 'SCRNAPCA_L2_k-5_paper'
    # dir_ = 'del_profiles'

    # #for scRNA, L1, K = 5
    # #NOTE: Not all experiments are done
    dataset = 'SCRNA'
    metric = 'L1'
    Ns = [10000, 20000, 30000, 40000]
    ks = [5]
    seeds = range(42, 45)
    dir_ = 'SCRNA_L1_paper'


    # By calling these functions twice, we're actually mining the data from the profiles twice.
    # Not a big deal but should fix
    # show_plots('k', 'build', Ns, ks, seeds, algos, dataset, metric, dir_)
    # show_plots('k', 'swap', Ns, ks, seeds, algos, dataset, metric, dir_)
    # show_plots('k', 'weighted', Ns, ks, seeds, algos, dataset, metric, dir_)
    show_plots('k', 'weighted_T', Ns, ks, seeds, algos, dataset, metric, dir_)

    # show_plots('N', 'build', Ns, ks, seeds, algos, dataset, metric)
    # show_plots('N', 'swap', Ns, ks, seeds, algos, dataset, metric)


if __name__ == '__main__':
    # verify_logfiles()
    # print("FILES VERIFIED\n\n")

    main()
