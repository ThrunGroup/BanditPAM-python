'''
Manually generate different subsets of MNIST to feed to the ELKI implementation
of FastPAM. This allows us to compute FastPAM's loss in a variety of
experiments. For the other algorithms, we implement them directly or use
existing implementations.

The losses from the ELKI implementation of FastPAM are used for Figure 1(a).
'''

from data_utils import *

def generate_manual_MNIST_subsets(args):
    '''
    Generate 10 different random subsets of size N, for each N in Ns.
    args.dataset must equal 'MNIST'.
    '''
    assert args.dataset == 'MNIST', "Can only do make subsets for MNIST"
    total_images, total_labels, sigma = load_data(args)
    seeds = range(42, 52)
    Ns = [500, 1000, 1500, 2000, 2500, 3000]
    for N in Ns:
        for seed in seeds:
            np.random.seed(seed)
            imgs = total_images[np.random.choice(range(len(total_images)), size = N, replace = False)]
            fname = 'ELKI/manual_MNIST_subsets/MNIST-' + str(N) + '-s-' + str(seed) + '.csv'
            np.savetxt(fname, imgs)

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    generate_manual_MNIST_subsets(args)
