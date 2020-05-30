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

DECIMAL_DIGITS = 5
SIGMA_DIVISOR = 1


'''
There are 5 functions that call d and therefore require the metric specification:
- cost_fn
- cost_fn_difference
- cost_fn_difference_FP1
- get_best_distances
- estimate_sigma
'''

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
    Load the entire (train + test) MNIST dataset
    returns: MNIST data reshaped into flattened arrays, all labels
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
        # if args.verbose >= 2:
        #     plt.imshow(train_images[0], cmap = 'gray')
        #     plt.show()

        # NOTE: Normalizing images
        return total_images.reshape(N, m * m) / 255, total_labels, sigma
    elif args.dataset == "SCRNA":
        #temp_df_ref = pd.read_csv('martin/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices/NUMPY_OUT/data.csv.gz', sep=',', compression='gzip', index_col=0)
        #temp_df_ref = pd.read_csv('martin/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices/NUMPY_OUT/data.csv', sep=',', index_col=0)
        file = 'martin/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices/NUMPY_OUT/np_data.npy'
        data_ = np.load(file)

        # sigma = estimate_sigma(data_, 300, metric="L2")
        sigma = 25 # NOTE: Really need to optimize this...
        return data_, None, sigma
    elif args.dataset == "SCRNAPCA":
        file = 'martin/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices/analysis_csv/pca/projection.csv'
        df = pd.read_csv(file, sep=',', index_col = 0)
        np_arr = df.to_numpy()
        # import ipdb; ipdb.set_trace()
        # sigma = estimate_sigma(np_arr, 300, metric = "L2")
        sigma = 0.01
        return np_arr, None, sigma
    elif args.dataset == 'HOC4':
        dir_ = 'hoc_data/hoc4/trees/'
        tree_files = [dir_ + tree for tree in os.listdir(dir_) if tree != ".DS_Store"]
        trees = []
        for tree_f in tree_files:
            with open(tree_f, 'rb') as fin:
                tree = pickle.load(fin)
                trees.append(tree)
        return trees, None, 0.0
    else:
        raise Exception("Didn't specify a valid dataset")

def init_logstring():
    logstring = {
        'loss' : {},
        'compute_exactly' : {},
        'p' : {},
        'sigma' : {},
    }
    return logstring

def update_logstring(logstring, k, best_distances, compute_exactly, p, sigma):
    logstring['loss'][k] = np.mean(best_distances)
    logstring['compute_exactly'][k] = compute_exactly
    logstring['p'][k] = p
    logstring['sigma'][k] = sigma
    return logstring

def empty_counter():
    pass

def d(x1, x2, metric = None):
    '''
    Note: If you really want to memoize these computations, you should probably
    make the empty_counter calls happen first (they won't happen otherwise), and
    break the actual distance computation into a sub-function with memoization.
    Do this carefully -- it might be a problem with multithreading.
    '''
    assert len(x1.shape) == len(x2.shape), "Arrays must be of the same dimensions in distance computation"
    if len(x1.shape) > 1:
        # NOTE: x1.shape is NOT the same as x2.shape! In particular, x1 is being BROADCAST to x2.
        # NOTE: SO MAKE SURE YOU KNOW WHAT YOU'RE DOING -- X1 AND X2 ARE NOT SYMMETRIC

        # WARNING:
        # Currently, the code is structured in cost_fn and cost_fn_difference_FP1 such that
        # we're only doing 1 arm at a time -- so x1.shape should be 1.
        # If x1.shape > 1, we were mis-calculating the number of distance computations (yikes!!)
        # This is ameliorated by having the double loop over x1.shape[0] and x2.shape[0]
        # Having the assertion is redundant but good defense programming for now
        # Existing experiments should be ok since x1.shape[0] was always 1
        # but should rerun all experiments just in case. Timestamp: Sunday, 5/24/2020 2:28PM

        assert x1.shape[0] == 1, "X1 is misshapen!"
        for _unused1 in range(x1.shape[0]):
            for _unused2 in range(x2.shape[0]):
                empty_counter()

        # NOTE: Assume first coordinate indexes tuples
        if metric == "L2":
            return np.linalg.norm(x1 - x2, ord = 2, axis = 1)
        elif metric == "L1":
            return np.linalg.norm(x1 - x2, ord = 1, axis = 1)
        else:
            raise Exception("Bad metric specified")

    else:
        # WARNING: See warning above. Extra-defensive here.
        assert x1.shape == x2.shape # 1 datapoint each, of dimension d
        assert len(x1.shape) == 1
        empty_counter()

        if metric == "L2":
            return np.linalg.norm(x1 - x2, ord = 2)
        elif metric == "L1":
            return np.linalg.norm(x1 - x2, ord = 1)
        else:
            raise Exception("Bad metric specified")

def d_tree(x1, x2, metric = None):
    assert metric == 'TREE', "Bad args to d_tree"
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
        raise Exception("Bad x2 type d_tree")

def cost_fn(dataset, tar_idx, ref_idx, best_distances, metric = None, use_diff = True):
    '''
    Returns the "cost" of point tar as a medoid:
    Distances from tar to ref if it's less than the existing best distance,
    best distance otherwise

    Use this only in the BUILD step
    '''
    if metric == 'TREE':
        assert type(dataset[tar_idx]) == Node, "Misshapen!"
        if use_diff:
            return np.minimum(d_tree(dataset[tar_idx], dataset[ref_idx], metric), best_distances[ref_idx]) - best_distances[ref_idx]
        return np.minimum(d_tree(dataset[tar_idx], dataset[ref_idx], metric), best_distances[ref_idx])
    else:
        if use_diff:
            return np.minimum(d(dataset[tar_idx].reshape(1, -1), dataset[ref_idx], metric), best_distances[ref_idx]) - best_distances[ref_idx]
        return np.minimum(d(dataset[tar_idx].reshape(1, -1), dataset[ref_idx], metric), best_distances[ref_idx])

# def cost_fn_difference_total(reference_dataset, full_dataset, target, current_medoids, best_distances):
def cost_fn_difference(imgs, swaps, tmp_refs, current_medoids, metric = None):
    '''
    Returns the "cost" of point tar as a medoid:
    Distances from tar to ref if it's less than the existing best distance,
    best distance otherwise
    '''
    # NOTE: Each member of swap is a PAIR

    # BUG: This function seems wrong
    # The gain is the difference between min(new_medoid, best_distance) - min(old_medoid, best_distance)
    # i.e. the difference in loss/distance for each point

    # BUG: What if the best_distances uses c1 as a medoid? Seems to be a bug here
    # IDEA: I think the best way to do this is to keep track of the medoid a point is assigned to,
    # to avoid re-assigning the medoids every time
    # Cases:
    #   - The current best distance uses c1, a currently assigned medoid, and c2 would become the new closest medoid
    #   - The current best distance uses c1, but swapping it to c2 would mean a totally different medoid c3 becomes the closest
    #   - The current best distance does NOT use c1, and c2 would become the new closest medoid
    #   - The current distance does NOT use c1, and c2 would also NOT be the new closest medoid, so the point is unaffected

    # NOTE: need to avoid performing this list modification; it's too expensive
    # NOTE: Very expensive for distance calls!
    # NOTE: How about using the "gain" in the loss instead?
    num_targets = len(swaps)
    reference_best_distances, reference_closest_medoids, reference_second_best_distances = get_best_distances(current_medoids, imgs, subset = tmp_refs, return_second_best = True, metric = metric)

    # gains = new - old .... should negative
    new_losses = np.zeros(num_targets)

    # for each swap
    #   for each ref point -- cases are on REF points
    #       if ref point is NOT assigned to o, new_losses += min(best_distances[ref_point], d(new_med, ref_point)) - best_distances[ref_point] (CASE1)
    #       if ref point IS assigned to o:
    #           if ref_point would be assigned to n: new_losses += d(new_med, ref_point) - best_distances[ref_point] -- CAN be positive (CASE2)
    #           else: new_losses += second_best_distances[ref_point] - best_distance[ref_point] -- WILL be positive (CASE3)
    #           Combine these (Cases 2 and 3) into CASE 2: min( d(new_med, ref_point), second_best_distances[ref_point]) - best_distances[ref_point]
    N = len(imgs)

    #############
    # Approach 1: DOESN'T WORK
    # import ipdb; ipdb.set_trace()
    #
    # # NOTE: right now loop is over for s in swaps. Actually only need to compute it for distinct old_medoids (hope compiler gets this)
    # case1s = np.array([np.where(reference_closest_medoids == current_medoids[s[0]])[0] for s in swaps]) # Non-square... only depends on s[0]
    # case2s = np.array([np.where(reference_closest_medoids != current_medoids[s[0]])[0] for s in swaps]) # Non-square... only depends on s[0]
    # # NOTE: right now loop is over for s in swaps. Could change this to be for only distinct new_medoids (hope compiler gets this)
    # new_medoid_distances = np.array([d(imgs[s[1]].reshape(1, -1), imgs[tmp_refs]) for s in swaps]) # Square... only depends on s[1]
    # new_losses += np.sum( np.minimum( new_medoid_distances[:, case1s], reference_second_best_distances[case1s] ), axis = 1) #case1
    # new_losses += np.sum( np.minimum( new_medoid_distances[:, case2s], reference_best_distances[case2s] ), axis = 1 ) #case2

    #######################
    # Approach 2:
    for s_idx, s in enumerate(swaps):
        # NOTE: WHEN REFERRING TO BEST_DISTANCES AND BEST_DISTANCES, USE INDICES. OTHERWISE, USE TMP_REFS[INDICES]!!
        # This is because best_distance is computed above and only returns the re-indexed subset
        old_medoid = current_medoids[s[0]]
        new_medoid = s[1]
        case1 = np.where(reference_closest_medoids == old_medoid)[0] # INDICES
        case2 = np.where(reference_closest_medoids != old_medoid)[0] # INDICES
        # NOTE: Many redundant computations of d here -- imgs[new_medoid] is the new medoid in lots of swaps!
        new_medoid_distances = d(imgs[new_medoid].reshape(1, -1), imgs[tmp_refs], metric)
        new_losses[s_idx] += np.sum( np.minimum( new_medoid_distances[case1], reference_second_best_distances[case1] ) ) #case1
        new_losses[s_idx] += np.sum( np.minimum( new_medoid_distances[case2], reference_best_distances[case2] ) ) #case2
        # NOTE: Can remove this since we're subtracting a constant from every candidate -- so not actually the difference
        # new_losses[s_idx] -= np.sum(reference_best_distances) # negative terms from both case1 and case2
    ##########################


    new_losses /= len(tmp_refs)

    return new_losses

def cost_fn_difference_FP1(imgs, swaps, tmp_refs, current_medoids, metric = None, return_sigma = False, use_diff = True):
    '''
    Returns the new losses if we were to perform the swaps in swaps

    NOTE:
    The FastPAM1 optimization consists of two mini-optimizations:
        (a) Cache d(x_old, x_ref) for every pair x_old and x_ref, since this doesn't change with x_n -- and keep track of the second best distance in case you're gonna use that
        (b) Cache d(x_new, x_ref) for every pair x_new and x_ref, since this doesn't change with old
    Then compute Delta_TD for every pair (x_old, x_new) using these CACHED values

    You have already incorporated the optimization (a) by keeping track of the second_best_distances and
    using the selector in your loop; this optimization is already included in the above function, cost_fn_difference

    You just need to implement optimization (b) -- should be easy!
    '''
    # IDEA: I think the best way to do this is to keep track of the medoid a point is assigned to,
    # to avoid re-assigning the medoids every time
    # Cases:
    #   - The current best distance uses c1, a currently assigned medoid, and c2 would become the new closest medoid
    #   - The current best distance uses c1, but swapping it to c2 would mean a totally different medoid c3 becomes the closest
    #   - The current best distance does NOT use c1, and c2 would become the new closest medoid
    #   - The current distance does NOT use c1, and c2 would also NOT be the new closest medoid, so the point is unaffected

    num_targets = len(swaps)
    reference_best_distances, reference_closest_medoids, reference_second_best_distances = get_best_distances(current_medoids, imgs, subset = tmp_refs, return_second_best = True, metric = metric)

    new_losses = np.zeros(num_targets)
    sigmas = np.zeros(num_targets)

    # for each swap
    #   for each ref point -- cases are on REF points
    #       if ref point is NOT assigned to o, new_losses += min(best_distances[ref_point], d(new_med, ref_point)) - best_distances[ref_point] (CASE1)
    #       if ref point IS assigned to o:
    #           if ref_point would be assigned to n: new_losses += d(new_med, ref_point) - best_distances[ref_point] -- CAN be positive (CASE2)
    #           else: new_losses += second_best_distances[ref_point] - best_distance[ref_point] -- WILL be positive (CASE3)
    #           Combine these (Cases 2 and 3) into CASE 2: min( d(new_med, ref_point), second_best_distances[ref_point]) - best_distances[ref_point]
    N = len(imgs)

    # Full FastPAM1 approach:
    distinct_new_medoids = set([s[1] for s in swaps])
    ALL_new_med_distances = np.zeros((len(distinct_new_medoids), len(tmp_refs))) ## NOTE: Re-indexing distinct elems!!
    reidx_lookup = {}
    for d_n_idx, d_n in enumerate(distinct_new_medoids):
        reidx_lookup[d_n] = d_n_idx # Smarter way to do this?
        if metric == 'TREE':
            ALL_new_med_distances[d_n_idx] = d_tree(imgs[d_n], imgs[tmp_refs], metric)
        else:
            ALL_new_med_distances[d_n_idx] = d(imgs[d_n].reshape(1, -1), imgs[tmp_refs], metric)


    for s_idx, s in enumerate(swaps):
        # NOTE: WHEN REFERRING TO BEST_DISTANCES AND BEST_DISTANCES, USE INDICES. OTHERWISE, USE TMP_REFS[INDICES]!!
        # This is because best_distance is computed above and only returns the re-indexed subset
        old_medoid = current_medoids[s[0]]
        new_medoid = s[1]
        case1 = np.where(reference_closest_medoids == old_medoid)[0] # INDICES
        case2 = np.where(reference_closest_medoids != old_medoid)[0] # INDICES
        # NOTE: Many redundant computations of d here -- imgs[new_medoid] is the new medoid in lots of swaps!
        new_medoid_distances = ALL_new_med_distances[reidx_lookup[new_medoid]]
        case1_losses = np.minimum( new_medoid_distances[case1], reference_second_best_distances[case1] )
        case2_losses = np.minimum( new_medoid_distances[case2], reference_best_distances[case2] )

        if use_diff:
            case1_losses -= reference_best_distances[case1]
            case2_losses -= reference_best_distances[case2]

        new_losses[s_idx] = np.sum(case1_losses) + np.sum(case2_losses)

        if return_sigma:
            # NOTE: Be careful here. We're defining the arm parameter as
            # the new loss, not the CHANGE in loss. So \sigma should be
            # the variance in the new induced loss, NOT the variance in the CHANGE
            # This shouldn't affect \sigma because the change = old - new and old is fixed
            sigmas[s_idx] = np.std(np.hstack((case1_losses, case2_losses))) / SIGMA_DIVISOR

    new_losses /= len(tmp_refs)

    if return_sigma:
        return new_losses, sigmas

    return new_losses

def get_best_distances(medoids, dataset, subset = None, return_second_best = False, metric = None):
    '''
    For each point, calculate the minimum distance to any medoid

    medoids: a list of medoid INDICES
    dataset: a numpy array of POINTS

    DO NOT CALL THIS FROM RANDOM FNS WHICH SAMPLE THE DATASET, E.G. UCB
    '''
    assert len(medoids) >= 1, "Need to pass at least one medoid"
    assert not (return_second_best and len(medoids) < 2), "Need at least 2 medoids to avoid infs when asking for return_second_best"

    if metric == 'TREE':
        inner_d_fn = d_tree
    else:
        inner_d_fn = d

    if subset is None:
        N = len(dataset)
        refs = range(N)
    else:
        refs = subset

    # NOTE: use a *Heap* or sorted linked list for BD, 2BD, 3BD etc and eject as necessary if doing multiple swaps

    best_distances = np.array([float('inf') for _ in refs])
    second_best_distances = np.array([float('inf') for _ in refs])
    closest_medoids = np.array([-1 for _ in refs])

    # Example: subset = [15, 32, 57] then loop is (p_idx, point) = (1, 15), (2, 32), (3, 57)
    # NOTE: Could speed this up with array broadcasting and taking min across medoid axis
    for p_idx, point in enumerate(refs):
        for m in medoids:
            # BUG, WARNING, NOTE: If dataset has been shuffled, than the medoids will refer to the WRONG medoids!!!
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
    # NOTE: Could use scipy.stats.norm for this
    exponent = (-0.5 * ((x - mu) / sigma)**2)
    return np.exp( exponent ) / (sigma * np.sqrt(2 * np.pi))

def estimate_sigma(dataset, N = None, metric = None):
    '''
    Use this to estimate sigma, as in sigma-sub-Gaussian
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

    distances -= np.mean(distances) # Center distances
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

def medoid_swap(medoids, best_swap, imgs, loss, args):
    # NOTE Store these explicitly to avoid incorrect reference after medoids have been updated when printing
    orig_medoid = medoids[best_swap[0]]
    new_medoid = best_swap[1]

    new_medoids = medoids.copy()
    new_medoids.remove(orig_medoid)
    new_medoids.append(new_medoid)
    new_best_distances, new_closest_medoids = get_best_distances(new_medoids, imgs, metric = args.metric)
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
    if visualization == 'tsne':
        X_embedded = TSNE(n_components=2).fit_transform(dataset)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='b')
        plt.scatter(X_embedded[medoids, 0], X_embedded[medoids, 1], c='r')
        plt.show()
    else:
        raise Exception('Bad Visualization Arg')
    

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
    visualize_medoids(imgs, [891, 392])
