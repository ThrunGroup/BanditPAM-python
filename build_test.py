import sys
import numpy as np
import mnist
import matplotlib.pyplot as plt
import argparse

def get_args(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help = 'print debugging output', action = 'count', default = 0)
    parser.add_argument('-k', '--num_medoids', help = 'Number of medoids', type = int, default = 10)
    parser.add_argument('-N', '--sample_size', help = 'Sampling size of dataset', type = int, default = 700)
    parser.add_argument('-s', '--seed', help = 'Random seed', type = int, default = 42)
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

    # NOTE: Normalizing images
    return total_images.reshape(N, m * m) / 255, total_labels

def d(x1, x2):
    return np.linalg.norm(x1 - x2, ord = 2)

def cost_fn(imgs, tar_idx, ref_idx, best_distances):
    '''
    Returns the "cost" of point tar as a medoid:
    Distances from tar to ref if it's less than the existing best distance,
    best distance otherwise
    '''
    return min(d(imgs[tar_idx], imgs[ref_idx]), best_distances[ref_idx])

def get_best_distances(medoids, imgs):
    '''
    For each point, calculate the minimum distance to any medoid
    '''
    assert len(medoids) >= 1, "Need to pass at least one medoid"
    N = len(imgs)
    best_distances = [float('inf') for _ in range(N)]
    for p in range(N):
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
    N = len(imgs)
    best_distances = [float('inf') for _ in range(N)]
    for k in range(args.num_medoids):
        print("Finding medoid", k)
        # Greedily choose the point which minimizes the loss
        best_loss = float('inf')
        best_medoid = -1

        for target in range(N):
            if (target + 1) % 100 == 0: print(target)
            # if target in medoids: continue # Skip existing medoids NOTE: removing this optimization for complexity comparison

            loss = 0
            for reference in range(N):
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

    print("Distances computations:", d_count, "k*n^2:", args.num_medoids * (N)**2)
    return medoids

def UCB_build(args, imgs):
    # When to stop sampling an arm? Ans: When lcb >= ucb_1
    # How to determine sigma? Ans: Seems to be determined manually by looking at distribution of data. Confidence bound scales as sigma (makes sense)

    ### Parameters
    N = len(imgs)
    p = 1e-2
    sigma = 0.7
    num_samples = np.zeros(N)
    estimates = np.zeros(N)
    medoids = []
    best_distances = [float('inf') for _ in range(N)]
    batch_size = 100 # NOTE: What should this init_size be? 20? Also note that this will result in (very minor) inefficiencies when batch_size > 1

    def sample_for_targets(imgs, targets, batch_size):
        # NOTE: Fix this with array broadcasting
        N = len(imgs)
        estimates = np.zeros(len(targets))
        tmp_refs = np.array(np.random.choice(N, size = batch_size, replace = False), dtype='int')
        for tar_idx, target in enumerate(targets):
            distances = np.zeros(batch_size)
            for tmp_idx, tmp in enumerate(tmp_refs):
                ## tmp is the actually index of the reference point, tmp_idx just numerates them)
                distances[tmp_idx] = cost_fn(imgs, target, tmp, best_distances) # NOTE: depends on other medoids too!
            estimates[tar_idx] = np.mean(distances)
        return estimates

    # Iteratively:
    # Pretend each previous arm is fixed.
    # For new arm candidate, true parameter is the TRUE loss when using the point as medoid
        # As a substitute, can measure the "gain" of using this point -- negative DECREASE in distance (the lower the distance, the better)
    # We sample this using UCB algorithm to get confidence bounds on what that loss will be
    # Update ucb, lcb, and empirical estimate by sampling WITH REPLACEMENT(NOTE)
        # If more than n points, just compute exactly -- otherwise, there's a failure mode where
        # Two points very close together require shittons of samples to distinguish their mean distance

    for k in range(args.num_medoids):
        print("Finding medoid", k)
        ## Initialization
        step_count = 0
        candidates = range(N)
        lcbs = -100 * np.ones(N)
        ucbs = -100 * np.ones(N)

        # Pull arms, update ucbs and lcbs
        while(len(candidates) > 1): # NOTE: Should also probably restrict absolute distance in cb_delta?
            print("Step count", step_count, candidates)
            step_count += 1
            # NOTE: Don't update all estimates, just pulled arms
            estimates[candidates] = (((step_count - 1) * estimates[candidates]) + sample_for_targets(imgs, candidates, batch_size)) / step_count
            cb_delta = sigma * np.sqrt(np.log(1 / p) / (batch_size * step_count))
            lcbs[candidates] = estimates[candidates] - cb_delta
            ucbs[candidates] = estimates[candidates] + cb_delta

            # Determine arms to pull
            best_ucb = ucbs.min()
            candidates = np.where(lcbs < best_ucb)[0]
        print("Medoid:", candidates)
        medoids.append(candidates[0])
        best_distances = get_best_distances(medoids, imgs)
    print(medoids)


def gaussian(mu, sigma, x):
    # NOTE: Could use scipy.stats.norm for this
    exponent = (-0.5 * ((x - mu) / sigma)**2)
    return np.exp( exponent ) / (sigma * np.sqrt(2 * np.pi))

def estimate_sigma(imgs):
    N = len(imgs)
    if N > 1000: print("Warning, this is going to be very slow for lots of images!")
    distances = np.zeros(N)
    for tar_idx, tar in enumerate(imgs):
        this_tar_distances = np.zeros(N)
        for ref_idx, ref in enumerate(imgs):
            this_tar_distances[ref_idx] = d(imgs[tar_idx], imgs[ref_idx])
        distances[tar_idx] = np.mean(this_tar_distances)

    distances -= np.mean(distances) # Center distances
    plt.hist(distances)
    x = np.arange(-3, 3, 0.1)
    y = gaussian(0, 0.7, x)
    plt.plot(x, y * N)
    plt.show()


def main(sys_args):
    args = get_args(sys.argv[1:])
    np.random.seed(args.seed)
    total_images, total_labels = load_data(args)
    sample_size = args.sample_size
    imgs = total_images[np.random.choice(range(len(total_images)), size = sample_size, replace = False)]
    #estimate_sigma(imgs)
    # Sigma = 0.7 looks ok

    medoids = naive_build(args, imgs)
    # 595 is true medoid at 6387.411136116143
    # 285 is close second at 6392.1460710 -- not sure why not normalizing gives a problem
    medoids = [595, 306, 392, 319, 23, 558, 251, 118, 448, 529]

    if args.verbose >= 2:
        for m in medoids:
            print(total_images[m].reshape(28, 28) * 255)

    # UCB_build(args, imgs)


if __name__ == "__main__":
    main(sys.argv)
