from data_utils import *

def generate_manual_MNIST_subsets(args):
    total_images, total_labels, sigma = load_data(args)
    seeds = range(42, 52)
    Ns = [500, 1000, 1500, 2000, 2500, 3000]
    for N in Ns:
        for seed in seeds:
            np.random.seed(seed)
            imgs = total_images[np.random.choice(range(len(total_images)), size = N, replace = False)]
            fname = 'manual_MNIST_subsets/MNIST-' + str(N) + '-s-' + str(seed) + '.csv'
            np.savetxt(fname, imgs)

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    generate_manual_MNIST_subsets(args)
