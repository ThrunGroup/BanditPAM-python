import os
import sys
import numpy as np
import mnist
import matplotlib.pyplot as plt
import argparse


def get_args(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help = 'print debugging output', action = 'count', default = 0)
    parser.add_argument('-k', '--num_medoids', help = 'Number of medoids', type = int, default = 3)
    parser.add_argument('-N', '--sample_size', help = 'Sampling size of dataset', type = int, default = 700)
    parser.add_argument('-s', '--seed', help = 'Random seed', type = int, default = 42)
    parser.add_argument('-d', '--dataset', help = 'Dataset to use', type = str, default = 'MNIST')
    parser.add_argument('-f', '--force', help = 'Recompute Experiments', action = 'store_true')
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
        if args.verbose >= 2:
            print(train_images[0])
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
    '''
    return min(d(dataset[tar_idx], dataset[ref_idx]), best_distances[ref_idx])

def get_best_distances(medoids, dataset):
    '''
    For each point, calculate the minimum distance to any medoid
    '''
    assert len(medoids) >= 1, "Need to pass at least one medoid"
    N = len(dataset)
    best_distances = [float('inf') for _ in range(N)]
    for p in range(N):
        for m in medoids:
            if d(dataset[m], dataset[p]) < best_distances[p]:
                best_distances[p] = d(dataset[m], dataset[p])
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
