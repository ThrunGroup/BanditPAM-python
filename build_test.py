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
    parser.add_argument('-k', '--num_medoids', help = 'Number of medoids', default = 10)
    args = parser.parse_args(arguments)
    return args


def load_data(args):
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
        plt.imshow(train_images[0], cmap='gray')
        plt.show()

    return total_images.reshape(N, m * m), total_labels

def d(x1, x2):
    return np.linalg.norm(x1 - x2, ord=2)

def get_best_distances(medoids, imgs):
    assert len(medoids) >= 1
    sample_size = 700
    best_distances = [float('inf') for _ in range(N)]
    for p in range(N):
        for m in medoids:
            if d(imgs[m], imgs[p]) < best_distances[p]:
                best_distances[p] = d(imgs[m], imgs[p])
    return best_distances

def naive_build(args, total_imgs):
    sample_size = 700
    # import ipdb; ipdb.set_trace()
    imgs = total_imgs[np.random.choice(range(len(total_imgs)), size = sample_size, replace = False)]
    medoids = []
    best_distances = [float('inf') for _ in range(N)]
    for k in range(args.num_medoids):
        print("Finding medoid", k)
        # Greedily choose the point which minimizes the loss
        best_loss = float('inf')
        best_medoid = -1

        for target in range(N):
            if (target + 1) % 100 == 0: print(target)
            if target in medoids: continue # Skip existing medoids

            loss = 0
            for reference in range(N):
                if d(imgs[target], imgs[reference]) < best_distances[reference]:
                    loss += d(imgs[target], imgs[reference])
                else:
                    loss += best_distances[reference]

            if loss < best_loss:
                best_loss = loss
                best_medoid = target

        # Once we have chosen the best medoid, reupdate the best distances
        # Don't do this OTF to avoid overwriting best_distances or requiring deep copy
        # Don't include these distance computations in the running metric because they could be computed OTF / tracked
        medoids.append(best_medoid)
        best_distances = get_best_distances(medoids, imgs)
        print(medoids)

    return medoids

def UCB_build():
    pass



def main():
    pass

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    total_images, total_labels = load_data(args)
    print(naive_build(args, total_images))
