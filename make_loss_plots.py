import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def showx():
    plt.draw()
    plt.pause(1) # <-------
    input("<Hit Enter To Close>")
    plt.close()

def get_file_loss(file_):
    if 'ucb' in file_ or 'naive_v1' in file_:
        num_lines = 4
    else:
        num_lines = 2

    with open(file_, 'r') as fin:
        line_idx = 0

        while line_idx < num_lines:
            line_idx += 1
            line = fin.readline()

        final_loss = line.split(' ')[-1]
        return float(final_loss)

def get_swaps(file_):
    with open(file_, 'r') as fin:
        swaps = []

        line = fin.readline()

        while line.strip() != 'Swap Logstring:': # Need to get past the 'swap:' line in build logstring
            line = fin.readline()

        while line.strip() != 'swap:':
            line = fin.readline()

        line = fin.readline()
        while line:
            medoids_swapped = line.split(' ')[-1].strip()
            swaps.append(medoids_swapped)
            line = fin.readline()

        last_old_medoid = medoids_swapped.split(',')[0]
        last_new_medoid = medoids_swapped.split(',')[1].strip()
        assert last_old_medoid == last_new_medoid, "The last swap should try to swap a medoid with itself"

        return swaps

def get_build_meds(file_):
    with open(file_, 'r') as fin:
        line = fin.readline()
    return line.strip()


def get_swap_meds(file_):
    with open(file_, 'r') as fin:
        line = fin.readline()
        line = fin.readline()
    return line.strip()

def verify_optimization_paths():
    loss_dir = 'profiles/Loss_plots_paper/'

    algos = ['naive_v1', 'ucb']
    seeds = range(10)
    Ns = [500, 1000, 1500, 2000, 2500, 3000]
    k = 5

    for N_idx, N in enumerate(Ns):
        for seed_idx, seed in enumerate(seeds):
            ucb_filename = loss_dir + 'L-ucb-True-BS-v-0-k-' + str(k) + '-N-' + str(N) + '-s-' + str(seed + 42) + '-d-MNIST-m-L2-w-'
            naive_filename = loss_dir + 'L-naive_v1-True-BS-v-0-k-' + str(k) + '-N-' + str(N) + '-s-' + str(seed + 42) + '-d-MNIST-m-L2-w-'

            ucb_built = get_build_meds(ucb_filename)
            ucb_swapped = get_swap_meds(ucb_filename)
            ucb_swaps = get_swaps(ucb_filename)

            naive_built = get_build_meds(naive_filename)
            naive_swapped = get_swap_meds(naive_filename)
            naive_swaps = get_swaps(naive_filename)

            if ucb_built != naive_built:
                print("Build medoids disagree on " + str(N) + ',' + str(seed))
                print(naive_built)
                print(ucb_built)

            if ucb_swapped != naive_swapped:
                print("Build medoids disagree on " + str(N) + ',' + str(seed))
                print(naive_swapped)
                print(ucb_swapped)


            if ucb_swaps != naive_swaps:
                print("Build medoids disagree on " + str(N) + ',' + str(seed))
                print(naive_swaps)
                print(ucb_swaps)

def get_FP_loss(N, seed):
    with open('manual_fastpam_losses.txt', 'r') as fin:
        prefix = "N=" + str(N) + ",seed=" + str(seed + 42)+":"

        line = fin.readline()
        while line[:len(prefix)] != prefix:
            # import ipdb; ipdb.set_trace()
            line = fin.readline()

        fp_loss = float(line.split(':')[-1])/N
        print(N, seed + 42, fp_loss)
        return fp_loss

def make_plots():
    loss_dir = 'profiles/Loss_plots_paper/'

    algos = ['naive_v1', 'ucb', 'clarans', 'em_style']
    seeds = range(10)
    Ns = [500, 1000, 1500, 2000, 2500, 3000]
    k = 5

    mult_jitter = 20

    alg_to_legend = {
        'naive_v1' : 'PAM',
        'ucb' : 'Bandit-PAM',
        'clarans' : 'CLARANS',
        'em_style' : 'Voronoi Iteration',
        'fp' : 'FastPAM',
    }

    ADD_JITTER = 35
    alg_to_add_jitter = {
        'ucb' : ADD_JITTER,
        'clarans' : -ADD_JITTER,
        'em_style' : ADD_JITTER,
        'fp' : -ADD_JITTER,
    }

    alg_color = {
        'ucb' : 'b',
        'clarans' : 'r',
        'em_style' : 'g',
        'fp' : 'y',
    }
    losses = np.zeros((len(Ns), len(algos) + 1, len(seeds)))

    for N_idx, N in enumerate(Ns):
        for algo_idx, algo in enumerate(algos):
            for seed_idx, seed in enumerate(seeds):
                filename = loss_dir + 'L-' + algo + '-True-BS-v-0-k-' + str(k) + '-N-' + str(N) + '-s-' + str(seed + 42) + '-d-MNIST-m-L2-w-'
                losses[N_idx, algo_idx, seed_idx] = get_file_loss(filename)

    # FastPAM special case
    for N_idx, N in enumerate(Ns):
        for seed_idx, seed in enumerate(seeds):
            losses[N_idx, 4, seed_idx] = get_FP_loss(N, seed)

    # Normalize losses
    for N_idx, N in enumerate(Ns):
        for seed_idx, seed in enumerate(seeds):
            naive_value = losses[N_idx, 0, seed_idx]
            losses[N_idx, :, seed_idx] /= naive_value

    sns.set()
    sns.set_style('white')
    fig, ax = plt.subplots(figsize = (6, 6))
    algos.append('fp') # NOTE
    for algo_idx, algo in enumerate(algos):
        if algo == 'naive_v1': continue

        d = {'N': Ns}
        for seed_idx, seed in enumerate(seeds):
            d["seed_" + str(seed)] = losses[:, algo_idx, seed_idx]
        df = pd.DataFrame(data = d)

        melt_df = df.melt('N', var_name='cols', value_name='vals')
        melt_df['N'] += np.random.randn(melt_df['N'].shape[0]) + alg_to_add_jitter[algo] # Add jitter
        sns.scatterplot(x="N", y="vals", data = melt_df, ax = ax, alpha = 0.6, label=alg_to_legend[algo])

        bars = 1.96 * np.std(losses[:, algo_idx, :], axis = 1) # Slice a specific algo, get a 2D array
        means = np.mean(losses[:, algo_idx, :], axis = 1)
        plt.errorbar(np.array(Ns) + alg_to_add_jitter[algo], means, yerr = bars, fmt = '+', capsize = 5, ecolor=alg_color[algo], elinewidth = 1.5, zorder = 100, mec=alg_color[algo], mew = 1.5)


    # for algo_idx, algo in enumerate(algos):
    #     plt.plot(Ns, np.mean(losses[:, algo_idx, :], axis = 1), label=alg_to_legend[algo])
    # plt.plot(Ns, np.mean(losses[:, 4, :], axis = 1), label='FastPAM') # Treat FastPAM as a special case

    plt.xlabel("$n$")
    plt.ylabel("Final Loss ($L$)")
    plt.legend()
    showx()


if __name__ == "__main__":
    loss_dir = 'profiles/Loss_plots_paper/'
    verify_optimization_paths()
    make_plots()

# print(get_file_loss(loss_dir + 'L-clarans-True-BS-v-0-k-5-N-500-s-46-d-MNIST-m-L2-w-', 2))
# print(get_file_loss(loss_dir + 'L-em_style-True-BS-v-0-k-5-N-500-s-46-d-MNIST-m-L2-w-', 2))
# print(get_file_loss(loss_dir + 'L-ucb-True-BS-v-0-k-5-N-500-s-46-d-MNIST-m-L2-w-', 4))
# print(get_file_loss(loss_dir + 'L-naive_v1-True-BS-v-0-k-5-N-500-s-46-d-MNIST-m-L2-w-', 4))
# print(get_swaps(loss_dir + 'L-naive_v1-True-BS-v-0-k-5-N-500-s-46-d-MNIST-m-L2-w-'))
# print(get_swaps(loss_dir + 'L-ucb-True-BS-v-0-k-5-N-500-s-46-d-MNIST-m-L2-w-'))
