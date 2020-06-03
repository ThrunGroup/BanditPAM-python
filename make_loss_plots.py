import numpy as np
import matplotlib.pyplot as plt

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

def verify_optimization_paths():
    pass

def make_plots():
    loss_dir = 'profiles/Loss_plots_paper/'

    algos = ['naive_v1', 'ucb', 'clarans', 'em_style']
    seeds = range(10)
    Ns = [500, 1000, 1500, 2000, 2500, 3000]
    k = 5

    losses = np.zeros((len(Ns), len(algos), len(seeds)))

    for N_idx, N in enumerate(Ns):
        for algo_idx, algo in enumerate(algos):
            for seed_idx, seed in enumerate(seeds):
                filename = loss_dir + 'L-' + algo + '-True-BS-v-0-k-' + str(k) + '-N-' + str(N) + '-s-' + str(seed + 42) + '-d-MNIST-m-L2-w-'
                losses[N_idx, algo_idx, seed_idx] = get_file_loss(filename)

    # Normalize losses
    for N_idx, N in enumerate(Ns):
        for seed_idx, seed in enumerate(seeds):
            naive_value = losses[N_idx, 0, seed_idx]
            losses[N_idx, :, seed_idx] /= naive_value

    for algo_idx, algo in enumerate(algos):
        plt.plot(Ns, np.mean(losses[:, algo_idx, :], axis = 1), label=algo)

    plt.legend()
    showx()


if __name__ == "__main__":
    make_plots()

# print(get_file_loss(loss_dir + 'L-clarans-True-BS-v-0-k-5-N-500-s-46-d-MNIST-m-L2-w-', 2))
# print(get_file_loss(loss_dir + 'L-em_style-True-BS-v-0-k-5-N-500-s-46-d-MNIST-m-L2-w-', 2))
# print(get_file_loss(loss_dir + 'L-ucb-True-BS-v-0-k-5-N-500-s-46-d-MNIST-m-L2-w-', 4))
# print(get_file_loss(loss_dir + 'L-naive_v1-True-BS-v-0-k-5-N-500-s-46-d-MNIST-m-L2-w-', 4))
