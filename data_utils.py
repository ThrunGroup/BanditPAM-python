import os
import sys
import numpy as np
import mnist
import matplotlib.pyplot as plt
import argparse

DECIMAL_DIGITS = 5

def get_args(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help = 'print debugging output', action = 'count', default = 0)
    parser.add_argument('-k', '--num_medoids', help = 'Number of medoids', type = int, default = 3)
    parser.add_argument('-N', '--sample_size', help = 'Sampling size of dataset', type = int, default = 700)
    parser.add_argument('-s', '--seed', help = 'Random seed', type = int, default = 42)
    parser.add_argument('-d', '--dataset', help = 'Dataset to use', type = str, default = 'MNIST')
    parser.add_argument('-f', '--force', help = 'Recompute Experiments', action = 'store_true')
    parser.add_argument('-w', '--warm_start_medoids', help = 'Initial medoids to start with', type = str, default = '')
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
    else:
        raise Exception("Didn't specify a valid dataset")

def d(x1, x2):
    return np.linalg.norm(x1 - x2, ord = 2)

def cost_fn(dataset, tar_idx, ref_idx, best_distances):
    '''
    Returns the "cost" of point tar as a medoid:
    Distances from tar to ref if it's less than the existing best distance,
    best distance otherwise

    Use this only in the BUILD step
    '''
    return min(d(dataset[tar_idx], dataset[ref_idx]), best_distances[ref_idx])

def cost_fn_difference_total(reference_dataset, full_dataset, target, current_medoids, best_distances):
    '''
    Returns the "cost" of point tar as a medoid:
    Distances from tar to ref if it's less than the existing best distance,
    best distance otherwise
    '''
    # target should be a PAIR
    c1 = target[0]
    c2 = target[1]
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

    potential_medoids = current_medoids.copy()
    potential_medoids.remove(current_medoids[c1])
    potential_medoids.append(c2)
    # NOTE: Very expensive for distance calls!
    # NOTE: How about using the "gain" in the loss instead?
    new_best_distances = get_best_distances__overload(potential_medoids, reference_dataset, full_dataset)

    return np.mean(np.array(new_best_distances))

def get_best_distances(medoids, dataset):
    '''
    For each point, calculate the minimum distance to any medoid

    medoids: a list of medoid INDICES
    dataset: a numpy array of POINTS

    DO NOT CALL THIS FROM RANDOM FNS WHICH SAMPLE THE DATASET, E.G. UCB
    '''
    assert len(medoids) >= 1, "Need to pass at least one medoid"
    N = len(dataset)
    best_distances = [float('inf') for _ in range(N)]
    for p in range(N):
        for m in medoids:
            # BUG, WARNING, NOTE: If dataset has been shuffled, than the medoids will refer to the WRONG medoids!!!
            if d(dataset[m], dataset[p]) < best_distances[p]:
                best_distances[p] = d(dataset[m], dataset[p])
    return best_distances

# NOTE: Change this name
def get_best_distances__overload(medoids_list, ref_dataset, full_dataset):
    '''
    For each point, calculate the minimum distance to any medoid

    medoids_list: a list of medoid INDICES in the full_dataset
    ref_dataset: a numpy array of DATAPOINTS which we use to compute loss
    full_dataset: exists only for identifying the properly MEDOID DATAPOINTS

    DO NOT CALL THIS FROM RANDOM FNS WHICH SAMPLE THE DATASET, E.G. UCB
    '''
    assert len(medoids_list) >= 1, "Need to pass at least one medoid"
    medoids = full_dataset[medoids_list]
    N = len(ref_dataset)
    best_distances = [float('inf') for _ in range(N)]
    for p, ref_datapoint in enumerate(ref_dataset):
        for m in medoids:
            # BUG, WARNING, NOTE: If dataset has been shuffled, than the medoids will refer to the WRONG medoids!!!
            if d(m, ref_datapoint) < best_distances[p]:
                best_distances[p] = d(m, ref_datapoint)
    return best_distances

def gaussian(mu, sigma, x):
    # NOTE: Could use scipy.stats.norm for this
    exponent = (-0.5 * ((x - mu) / sigma)**2)
    return np.exp( exponent ) / (sigma * np.sqrt(2 * np.pi))

def estimate_sigma(dataset):
    '''
    Use this to estimate sigma, as in sigma-sub-Gaussian
    '''
    N = len(dataset)
    if N > 1000: print("Warning, this is going to be very slow for lots of images!")
    distances = np.zeros(N)
    for tar_idx, tar in enumerate(dataset):
        this_tar_distances = np.zeros(N)
        for ref_idx, ref in enumerate(dataset):
            this_tar_distances[ref_idx] = d(dataset[tar_idx], dataset[ref_idx])
        distances[tar_idx] = np.mean(this_tar_distances)

    distances -= np.mean(distances) # Center distances
    for sigma in np.arange(0.1, 1, 0.1):
        plt.hist(distances)
        x = np.arange(-3, 3, 0.1)
        y = gaussian(0, sigma, x)
        plt.plot(x, y * N)
        plt.show()

def medoid_swap(medoids, best_swap, imgs, loss, args):
    # NOTE Store these explicitly to avoid incorrect reference after medoids have been updated when printing
    orig_medoid = medoids[best_swap[0]]
    new_medoid = best_swap[1]

    new_medoids = medoids.copy()
    new_medoids.remove(orig_medoid)
    new_medoids.append(new_medoid)
    new_loss = np.mean(get_best_distances(new_medoids, imgs))
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
