'''
A number of convenience functions used by the different algorithms.
Also includes some constants.

There are 5 functions that call d and therefore require the an explicit metric:
- cost_fn
- cost_fn_difference
- cost_fn_difference_FP1
- get_best_distances
- estimate_sigma
- medoid_swap
'''

import os
import sys
import numpy as np
import mnist
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import pickle

from zss import simple_distance, Node
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances

DECIMAL_DIGITS = 5
SIGMA_DIVISOR = 1

def get_args(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help = 'print debugging output', action = 'count', default = 0)
    parser.add_argument('-k', '--num_medoids', help = 'Number of medoids', type = int, default = 3)
    parser.add_argument('-N', '--sample_size', help = 'Sampling size of dataset', type = int, default = 700)
    parser.add_argument('-s', '--seed', help = 'Random seed', type = int, default = 42)
    parser.add_argument('-d', '--dataset', help = 'Dataset to use', type = str, default = 'MNIST')
    parser.add_argument('-m', '--metric', help = 'Metric to use (L1 or L2)', type = str)
    parser.add_argument('-f', '--force', help = 'Recompute Experiments', action = 'store_true')
    parser.add_argument('-p', '--fast_pam1', help = 'Use FastPAM1 optimization', action = 'store_true')
    parser.add_argument('-r', '--fast_pam2', help = 'Use FastPAM2 optimization', action = 'store_true')
    parser.add_argument('-w', '--warm_start_medoids', help = 'Initial medoids to start with', type = str, default = '')
    parser.add_argument('-B', '--build_ao_swap', help = 'Build or Swap, B = just build, S = just swap, BS = both', type = str, default = 'BS')
    parser.add_argument('-e', '--exp_config', help = 'Experiment configuration file to use', required = False)
    args = parser.parse_args(arguments)
    return args

def load_data(args):
    '''
    Load the different datasets, as a numpy matrix if possible. In the case of
    HOC4 and HOC18, load the datasets as a list of trees.
    '''
    if args.dataset == 'MNIST':
        N = 70000
        m = 28
        sigma = 0.7
        train_images = mnist.train_images()
        train_labels = mnist.train_labels()

        test_images = mnist.test_images()
        test_labels = mnist.test_labels()

        total_images = np.append(train_images, test_images, axis = 0)
        total_labels = np.append(train_labels, test_labels, axis = 0)

        assert((total_images == np.vstack((train_images, test_images))).all())
        assert((total_labels == np.hstack((train_labels, test_labels))).all()) # NOTE: hstack since 1-D
        assert(total_images.shape == (N, m, m))
        assert(total_labels.shape == (N,))

        # Normalizing images
        return total_images.reshape(N, m * m) / 255, total_labels, sigma
    elif args.dataset == "SCRNA":
        file = 'person1/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices/NUMPY_OUT/np_data.npy'
        data_ = np.load(file)
        sigma = 25
        return data_, None, sigma
    elif args.dataset == "SCRNAPCA":
        file = 'person1/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices/analysis_csv/pca/projection.csv'
        df = pd.read_csv(file, sep=',', index_col = 0)
        np_arr = df.to_numpy()
        sigma = 0.01
        return np_arr, None, sigma
    elif args.dataset == 'HOC4':
        dir_ = 'hoc_data/hoc4/trees/'
        tree_files = [dir_ + tree for tree in os.listdir(dir_) if tree != ".DS_Store"]
        trees = []
        for tree_f in sorted(tree_files):
            with open(tree_f, 'rb') as fin:
                tree = pickle.load(fin)
                trees.append(tree)

        if args.verbose >= 1:
            print("NUM TREES:", len(trees))

        return trees, None, 0.0
    elif args.dataset == 'HOC18':
        dir_ = 'hoc_data/hoc18/trees/'
        tree_files = [dir_ + tree for tree in os.listdir(dir_) if tree != ".DS_Store"]
        trees = []
        for tree_f in tree_files:
            with open(tree_f, 'rb') as fin:
                tree = pickle.load(fin)
                trees.append(tree)

        if args.verbose >= 1:
            print("NUM TREES:", len(trees))

        return trees, None, 0.0
    elif args.dataset == 'GAUSSIAN':
        dataset = create_gaussians(args.sample_size, ratio = 0.6, seed = args.seed, visualize = False)
        return dataset
    else:
        raise Exception("Didn't specify a valid dataset")

def init_logstring():
    '''
    Create an empty logstring with the desired fields. The logstrings will be
    updated by the algorithms.
    '''

    logstring = {
        'loss' : {},
        'compute_exactly' : {},
        'p' : {},
        'sigma' : {},
        'swap' : {},
    }
    return logstring

def update_logstring(logstring, k, best_distances, compute_exactly, p, sigma, swap = None):
    '''
    Update a given logstring (python dict) with the results of a BUILD or SWAP
    iteration.
    '''

    logstring['loss'][k] = np.mean(best_distances)
    logstring['compute_exactly'][k] = compute_exactly
    logstring['p'][k] = p

    if type(sigma) == list:
        logstring['sigma'][k] = ""
        logstring['sigma'][k] += " min: " + str(round(sigma[0], DECIMAL_DIGITS))
        logstring['sigma'][k] += " 25th: " + str(round(sigma[1], DECIMAL_DIGITS))
        logstring['sigma'][k] += " median: " + str(round(sigma[2], DECIMAL_DIGITS))
        logstring['sigma'][k] += " 75th: " + str(round(sigma[3], DECIMAL_DIGITS))
        logstring['sigma'][k] += " max: " + str(round(sigma[4], DECIMAL_DIGITS))
        logstring['sigma'][k] += " mean: " + str(round(sigma[5], DECIMAL_DIGITS))
    else:
        logstring['sigma'][k] = sigma

    if swap is not None:
        logstring['swap'][k] = str(swap[0]) + ',' + str(swap[1])

    return logstring

def empty_counter():
    '''
    Empty function that is called once for every distance call. Allows for easy
    counting of the number of distance calls
    '''

    pass

def d(x1, x2, metric = None):
    '''
    Computes the distance between x1 and x2. If x2 is a list, computes the
    distance between x1 and every x2.

    Later, these computations should be memoized. But if you really want to
    memoize these computations, you should make the empty_counter calls happen
    first (they won't happen otherwise), and break the actual distance
    computation into a sub-function with memoization. Do this carefully to avoid
    problems with multithreading.
    '''
    assert len(x1.shape) == len(x2.shape), "Arrays must be of the same dimensions in distance computation"
    if len(x1.shape) > 1:
        # NOTE: x1.shape is NOT the same as x2.shape! In particular, x1 is being BROADCAST to x2.
        # NOTE: So make sure you know what you're doing -- x1 and x2 are not symmetric

        # WARNING:
        # Currently, the code is structured in cost_fn and cost_fn_difference_FP1 such that
        # we're only doing 1 arm at a time -- so x1.shape should be 1.
        # Distance call accuracy is enforced by having the double loop over
        # x1.shape[0] and x2.shape[0]

        # x1 is (1, d), x2 is (n, d) return should be (n)
        assert x1.shape[0] == 1, "X1 is misshapen!"
        for _unused1 in range(x1.shape[0]):
            for _unused2 in range(x2.shape[0]):
                empty_counter()

        if metric == "L2":
            return np.linalg.norm(x1 - x2, ord = 2, axis = 1)
        elif metric == "L1":
            return np.linalg.norm(x1 - x2, ord = 1, axis = 1)
        elif metric == "COSINE":
            return pairwise_distances(x1, x2, metric = 'cosine').reshape(-1)
        else:
            raise Exception("Bad metric specified")

    else:
        assert x1.shape == x2.shape # 1 datapoint each, of dimension d
        assert len(x1.shape) == 1
        empty_counter()

        if metric == "L2":
            return np.linalg.norm(x1 - x2, ord = 2)
        elif metric == "L1":
            return np.linalg.norm(x1 - x2, ord = 1)
        elif metric == "COSINE":
            return cosine(x1, x2)
        else:
            raise Exception("Bad metric specified")

def d_tree(x1, x2, metric = None, dist_mat = None):
    '''
    Use this function for computing the edit distance between two trees.
    Supports both on-the-fly computation (metric == 'TREE') as well as using the
    precomputed distance matrix (metric == 'PRECOMP')
    '''

    if metric == 'TREE':
        # Compute the tree edit distance on-the-fly
        assert metric == 'TREE', "Bad args to tree distance fn"
        assert type(x1) == Node, "First arg must always be a single node" # Code is currently structured to loop over arms
        if type(x2) == Node:
            # 1-on-1 comparison
            empty_counter()
            return simple_distance(x1, x2)
        elif type(x2) == np.ndarray:
            for _unused in x2:
                empty_counter()
            return np.array([simple_distance(x1, x2_elem) for x2_elem in x2])
        else:
            raise Exception("Bad x2 type tree distance fn")
    elif metric == 'PRECOMP':
        # Use the precomputed distance matrix
        assert dist_mat is not None, "Must pass distance matrix!"
        assert type(x1) == int or type(x1) == np.int64, "Must pass x1 as an int"
        if type(x2) == int or type(x2) == np.int64:
            # 1-on-1 comparison
            empty_counter()
            return dist_mat[x1, x2]
        elif type(x2) == np.ndarray:
            for _unused in x2:
                empty_counter()
            return np.array([dist_mat[x1, x2_elem] for x2_elem in x2])
        else:
            raise Exception("Bad x2 type tree distance fn", type(x2))
    else:
        raise Exception('Bad metric argument to tree distance function')

def cost_fn(dataset, tar_idx, ref_idx, best_distances, metric = None, use_diff = True, dist_mat = None):
    '''
    Returns the "cost" of adding the pointpoint tar as a medoid:
    distances from tar to ref if it's less than the existing best distance,
    best_distances[ref_idx] otherwise

    This is called by the BUILD step of naive PAM and BanditPAM (ucb_pam).

    Contains special cases for handling trees, both with precomputed distance
    matrix and on-the-fly computation.
    '''
    if metric == 'TREE':
        assert type(dataset[tar_idx]) == Node, "Misshapen!"
        if use_diff:
            return np.minimum(d_tree(dataset[tar_idx], dataset[ref_idx], metric), best_distances[ref_idx]) - best_distances[ref_idx]
        return np.minimum(d_tree(dataset[tar_idx], dataset[ref_idx], metric), best_distances[ref_idx])
    elif metric == 'PRECOMP':
        assert type(dataset[tar_idx]) == Node, "Misshapen!"
        # Need to pass indices of nodes instead of nodes themselves
        if use_diff:
            return np.minimum(d_tree(tar_idx, ref_idx, metric, dist_mat), best_distances[ref_idx]) - best_distances[ref_idx]
        return np.minimum(d_tree(tar_idx, ref_idx, metric, dist_mat), best_distances[ref_idx])
    else:
        if use_diff:
            return np.minimum(d(dataset[tar_idx].reshape(1, -1), dataset[ref_idx], metric), best_distances[ref_idx]) - best_distances[ref_idx]
        return np.minimum(d(dataset[tar_idx].reshape(1, -1), dataset[ref_idx], metric), best_distances[ref_idx])

def cost_fn_difference(imgs, swaps, tmp_refs, current_medoids, metric = None):
    '''
    Do not use this function. Always run experiments with the FastPAM1
    optimization, because it yields the same result.

    Returns the difference in costs for the tmp_refs if we were to perform the
    swap in swaps. Let c1 = swap[0], c2 = swap[1]. Then there are 4 cases:
      - The current best distance uses c1, a currently assigned medoid, and c2 would become the new closest medoid
      - The current best distance uses c1, but swapping it to c2 would mean a totally different medoid c3 becomes the closest
      - The current best distance does NOT use c1, and c2 would become the new closest medoid
      - The current distance does NOT use c1, and c2 would also NOT be the new closest medoid, so the point is unaffected
    '''

    raise Exception('This function is no longer supported. Please use FP1')

    num_targets = len(swaps)
    reference_best_distances, reference_closest_medoids, reference_second_best_distances = get_best_distances(current_medoids, imgs, subset = tmp_refs, return_second_best = True, metric = metric, dist_mat = dist_mat)
    new_losses = np.zeros(num_targets)
    N = len(imgs)

    for s_idx, s in enumerate(swaps):
        raise Exception("This fn does not support tree edit distance / precomp yet. May not be an issue;comment this line out if you're OK with that.")
        # WARNING: When referring to best_distances, use indices. Otherwise, use tmp_refs[indices]
        # This is because best_distance is computed above and only returns the re-indexed subset
        old_medoid = current_medoids[s[0]]
        new_medoid = s[1]
        case1 = np.where(reference_closest_medoids == old_medoid)[0] # INDICES
        case2 = np.where(reference_closest_medoids != old_medoid)[0] # INDICES
        # NOTE: Many redundant computations of d here -- imgs[new_medoid] is the new medoid in lots of swaps!
        new_medoid_distances = d(imgs[new_medoid].reshape(1, -1), imgs[tmp_refs], metric)
        new_losses[s_idx] += np.sum( np.minimum( new_medoid_distances[case1], reference_second_best_distances[case1] ) ) #case1
        new_losses[s_idx] += np.sum( np.minimum( new_medoid_distances[case2], reference_best_distances[case2] ) ) #case2

    new_losses /= len(tmp_refs)

    return new_losses

def cost_fn_difference_FP1(imgs, swaps, tmp_refs, current_medoids, metric = None, return_sigma = False, use_diff = True, dist_mat = None):
    '''
    Returns the new losses if we were to perform the swaps in swaps, as in
    cost_fn_difference above, but using the FastPAM1 optimization.

    NOTE:
    The FastPAM1 optimization consists of two mini-optimizations:
        (a) Cache d(x_old, x_ref) for every pair x_old and x_ref, since this doesn't change with x_n -- and keep track of the second best distance in case you're gonna use that
        (b) Cache d(x_new, x_ref) for every pair x_new and x_ref, since this doesn't change with old
    Then compute Delta_TD for every pair (x_old, x_new) using these CACHED values

    Both (a) and (b) are implemented.

    See cases in comment for cost_fn_difference; same cases appear here.
    '''
    num_targets = len(swaps)
    reference_best_distances, reference_closest_medoids, reference_second_best_distances = get_best_distances(current_medoids, imgs, subset = tmp_refs, return_second_best = True, metric = metric, dist_mat = dist_mat)

    new_losses = np.zeros(num_targets)
    sigmas = np.zeros(num_targets)

    N = len(imgs)

    distinct_new_medoids = set([s[1] for s in swaps])
    ALL_new_med_distances = np.zeros((len(distinct_new_medoids), len(tmp_refs))) # WARNING: Re-indexing distinct elems!!
    reidx_lookup = {}
    for d_n_idx, d_n in enumerate(distinct_new_medoids):
        reidx_lookup[d_n] = d_n_idx # Smarter way to do this?
        if metric == 'TREE':
            ALL_new_med_distances[d_n_idx] = d_tree(imgs[d_n], imgs[tmp_refs], metric)
        elif metric == 'PRECOMP':
            # Must pass indices to precomp instead of nodes
            ALL_new_med_distances[d_n_idx] = d_tree(d_n, tmp_refs, metric, dist_mat)
        else:
            ALL_new_med_distances[d_n_idx] = d(imgs[d_n].reshape(1, -1), imgs[tmp_refs], metric)


    for s_idx, s in enumerate(swaps):
        # WARNING: When referring to best_distances, use indices. Otherwise, use tmp_refs[indices]
        # This is because best_distance is computed above and only returns the re-indexed subset
        old_medoid = current_medoids[s[0]]
        new_medoid = s[1]
        case1 = np.where(reference_closest_medoids == old_medoid)[0] # List of indices
        case2 = np.where(reference_closest_medoids != old_medoid)[0] # List of indices
        new_medoid_distances = ALL_new_med_distances[reidx_lookup[new_medoid]]
        case1_losses = np.minimum( new_medoid_distances[case1], reference_second_best_distances[case1] )
        case2_losses = np.minimum( new_medoid_distances[case2], reference_best_distances[case2] )

        if use_diff:
            case1_losses -= reference_best_distances[case1]
            case2_losses -= reference_best_distances[case2]

        new_losses[s_idx] = np.sum(case1_losses) + np.sum(case2_losses)

        if return_sigma:
            sigmas[s_idx] = np.std(np.hstack((case1_losses, case2_losses))) / SIGMA_DIVISOR

    new_losses /= len(tmp_refs)

    if return_sigma:
        return new_losses, sigmas

    return new_losses

def get_best_distances(medoids, dataset, subset = None, return_second_best = False, metric = None, dist_mat = None):
    '''
    For each point, calculate the minimum distance to any medoid.

    Do not call this from random fns which subsample the dataset, or your
    indices will be thrown off.
    '''
    assert len(medoids) >= 1, "Need to pass at least one medoid"
    assert not (return_second_best and len(medoids) < 2), "Need at least 2 medoids to avoid infs when asking for return_second_best"

    if metric == 'TREE':
        inner_d_fn = d_tree
    elif metric == 'PRECOMP':
        inner_d_fn = d_tree
        assert dist_mat is not None, "Need to pass dist_mat to get_best_distances"
    else:
        inner_d_fn = d

    if subset is None:
        N = len(dataset)
        refs = range(N)
    else:
        refs = subset

    # NOTE: Use a Heap or sorted linked list for best distance, second best
    # distance, third best distance, etc and pop as necessary if doing multiple
    # swaps

    best_distances = np.array([float('inf') for _ in refs])
    second_best_distances = np.array([float('inf') for _ in refs])
    closest_medoids = np.array([-1 for _ in refs])

    # NOTE: Could speed this up with array broadcasting and taking min across medoid axis
    for p_idx, point in enumerate(refs):
        for m in medoids:
            # WARNING: If dataset has been shuffled, than the medoids will refer to the WRONG medoids!!!
            if metric == 'PRECOMP':
                # NOTE: Can probably consolidate this with case below by just saying dist_mat = None if not precomp
                if inner_d_fn(m, point, metric, dist_mat) < best_distances[p_idx]:
                    second_best_distances[p_idx] = best_distances[p_idx]
                    best_distances[p_idx] = inner_d_fn(m, point, metric, dist_mat)
                    closest_medoids[p_idx] = m
                elif inner_d_fn(m, point, metric, dist_mat) < second_best_distances[p_idx]:
                    # Reach this case if the new medoid is between current 2nd and first, but not better than first
                    second_best_distances[p_idx] = inner_d_fn(m, point, metric, dist_mat)
            else:
                if inner_d_fn(dataset[m], dataset[point], metric) < best_distances[p_idx]:
                    second_best_distances[p_idx] = best_distances[p_idx]
                    best_distances[p_idx] = inner_d_fn(dataset[m], dataset[point], metric)
                    closest_medoids[p_idx] = m
                elif inner_d_fn(dataset[m], dataset[point], metric) < second_best_distances[p_idx]:
                    # Reach this case if the new medoid is between current 2nd and first, but not better than first
                    second_best_distances[p_idx] = inner_d_fn(dataset[m], dataset[point], metric)

    if return_second_best:
        return best_distances, closest_medoids, second_best_distances
    return best_distances, closest_medoids


def gaussian(mu, sigma, x):
    '''
    Calculate the value of a Gaussian PDF for the given parameters.
    Could use scipy.stats.norm for this instead.
    '''
    exponent = (-0.5 * ((x - mu) / sigma)**2)
    return np.exp( exponent ) / (sigma * np.sqrt(2 * np.pi))

def estimate_sigma(dataset, N = None, metric = None):
    '''
    In early development, this fn was used to estimate sigma, as in
    sigma-sub-Gaussian. It's no longer used.
    '''

    if N is None:
        N = len(dataset)

    if N > 1000:
        print("Warning, this is going to be very slow for lots of images!")

    sample = dataset[np.random.choice(len(dataset), size = N, replace = False)]

    distances = np.zeros(N)
    for tar_idx, tar in enumerate(sample):
        this_tar_distances = np.zeros(N)
        for ref_idx, ref in enumerate(sample):
            this_tar_distances[ref_idx] = d(sample[tar_idx], sample[ref_idx], metric = metric)
        distances[tar_idx] = np.mean(this_tar_distances)

    distances -= np.mean(distances)
    plt.hist(distances)
    if metric == "L1":
        for sigma in np.arange(5, 50, 5):
            x = np.arange(-200, 200, 0.1)
            y = gaussian(0, sigma, x)
            plt.plot(x, y * N, label=sigma)
        plt.legend()
        plt.show()
    elif metric == "L2":
        for sigma in np.arange(0.001, .01, 0.001):
            x = np.arange(-.1, 0.1, 0.001)
            y = gaussian(0, sigma, x)
            plt.plot(x, y , label=sigma)
        plt.legend()
        plt.show()
    else:
        raise Exception("bad metric in estimate_sigma")

# TODO: Explicitly pass metric instead of args.metric here
def medoid_swap(medoids, best_swap, imgs, loss, args, dist_mat = None):
    '''
    Swaps the medoid-nonmedoid pair in best_swap if it would lower the loss on
    the datapoints in imgs. Returns a string describing whether the swap was
    performed, as well as the new medoids and new loss.
    '''

    # NOTE Store these explicitly to avoid incorrect reference after medoids have been updated when printing
    orig_medoid = medoids[best_swap[0]]
    new_medoid = best_swap[1]

    new_medoids = medoids.copy()
    new_medoids.remove(orig_medoid)
    new_medoids.append(new_medoid)
    new_best_distances, new_closest_medoids = get_best_distances(new_medoids, imgs, metric = args.metric, dist_mat = dist_mat)
    new_loss = np.mean(new_best_distances)
    performed_or_not = ''
    if new_loss < loss:
        performed_or_not = "SWAP PERFORMED"
        swap_performed = True
    else:
        performed_or_not = "NO SWAP PERFORMED"
        new_medoids = medoids

    if args.verbose >= 1:
        print("Tried to swap", orig_medoid, "with", new_medoid)
        print(performed_or_not)
        print("Old loss:", loss)
        print("New loss:", new_loss)

    return performed_or_not, new_medoids, min(new_loss, loss)

def visualize_medoids(dataset, medoids, visualization = 'tsne'):
    '''
    Helper function to visualize the given medoids of a dataset using t-SNE
    '''

    if visualization == 'tsne':
        X_embedded = TSNE(n_components=2).fit_transform(dataset)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='b')
        plt.scatter(X_embedded[medoids, 0], X_embedded[medoids, 1], c='r')
        plt.show()
    else:
        raise Exception('Bad Visualization Arg')

def create_gaussians(N, ratio = 0.6, seed = 42, visualize = True):
    '''
    Create some 2-D Gaussian toy data.
    '''

    np.random.seed(seed)
    cluster1_size = int(N * ratio)
    cluster2_size = N - cluster1_size

    cov1 = np.array([[1, 0], [0, 1]])
    cov2 = np.array([[1, 0], [0, 1]])

    mu1 = np.array([-10, -10])
    mu2 = np.array([10, 10])

    cluster1 = np.random.multivariate_normal(mu1, cov1, cluster1_size)
    cluster2 = np.random.multivariate_normal(mu2, cov2, cluster2_size)

    if visualize:
        plt.scatter(cluster1[:, 0], cluster1[:, 1], c='r')
        plt.scatter(cluster2[:, 0], cluster2[:, 1], c='b')
        plt.show()

    return np.vstack((cluster1, cluster2))


def extract_values(str_):
    '''
    Helper function for extracting the sigma statistics from a string in a
    logfile.
    '''
    float_arr = str_.split(' ')
    float_arr = [float(float_arr[idx]) for idx in range(1, 11, 2)]
    return float_arr

if __name__ == "__main__":
    create_gaussians(1000, 0.5, 42)

    ####### Use the code below to visualize the some medoids with t-SNE
    # args = get_args(sys.argv[1:])
    # total_images, total_labels, sigma = load_data(args)
    # np.random.seed(args.seed)
    # imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
    # visualize_medoids(imgs, [891, 392])
