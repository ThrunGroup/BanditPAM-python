import sys
import numpy as np
import mnist
import matplotlib.pyplot as plt
import argparse

np.random.seed(42)

def get_args(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help = 'print debugging output', action = 'count', default = 0)
    parser.add_argument('-k', '--num_medoids', help = 'Number of medoids', type = int, default = 10)
    args = parser.parse_args(arguments)
    return args


def load_data(args):
    '''
    Load the entire (train + test) MNIST dataset
    returns: MNIST data reshaped into flattened arrays, all labels
    '''
    N = 70000
    m = 28
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
    if args.verbose >= 1:
        print(train_images[0])
    if args.verbose >= 2:
        plt.imshow(train_images[0], cmap = 'gray')
        plt.show()

    return total_images.reshape(N, m * m), total_labels

def d(x1, x2):
    return np.linalg.norm(x1 - x2, ord = 2)

def get_best_distances(medoids, imgs):
    '''
    For each point, calculate the minimum distance to any medoid
    '''
    assert len(medoids) >= 1, "Need to pass at least one medoid"
    length = len(imgs)
    best_distances = [float('inf') for _ in range(length)]
    for p in range(length):
        for m in medoids:
            if d(imgs[m], imgs[p]) < best_distances[p]:
                best_distances[p] = d(imgs[m], imgs[p])
    return best_distances

def naive_build(args, imgs):
    '''
    Naively instantiates the medoids, corresponding to the BUILD step.
    Algorithm does so in a greedy way:
        for k in range(num_medoids):
            Add the medoid that will lead to lowest lost, conditioned on the
            previous medoids being fixed
    '''
    d_count = 0
    medoids = []
    length = len(imgs)
    best_distances = [float('inf') for _ in range(length)]
    for k in range(args.num_medoids):
        print("Finding medoid", k)
        # Greedily choose the point which minimizes the loss
        best_loss = float('inf')
        best_medoid = -1

        for target in range(length):
            if (target + 1) % 100 == 0: print(target)
            # if target in medoids: continue # Skip existing medoids NOTE: removing this optimization for complexity comparison

            loss = 0
            for reference in range(length):
                # if reference in medoids: continue # Skip existing medoids NOTE: removing this optimization for complexity comparison
                d_r_t = d(imgs[target], imgs[reference])
                d_count += 1
                loss += d_r_t if d_r_t < best_distances[reference] else best_distances[reference]

            if loss < best_loss:
                # So far, this new medoid is the best candidate
                best_loss = loss
                best_medoid = target

        # Once we have chosen the best medoid, reupdate the best distances
        # Don't do this OTF to avoid overwriting best_distances or requiring deep copy
        # Otherwise, we would have side-effects when recomputing best_distances and recursively backtracking
        # Also don't include these distance computations in the running metric because they could be computed OTF / tracked
        medoids.append(best_medoid)
        best_distances = get_best_distances(medoids, imgs)
        print(medoids)

    print(d_count, args.num_medoids * (length)**2)
    return medoids

def UCB_build():
    pass

def main():
    pass

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    total_images, total_labels = load_data(args)
    sample_size = 700
    imgs = total_images[np.random.choice(range(len(total_images)), size = sample_size, replace = False)]

    medoids = naive_build(args, imgs)

    if args.verbose >= 2:
        for m in medoids:
            print(total_images[m].reshape(28, 28))
