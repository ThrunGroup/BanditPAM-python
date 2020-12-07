'''
Uses the EM-style (Voronoi Iteration) approach to computing k-medoids.
We use the implementation from sklearn_extra.
'''

from sklearn_extra.cluster import KMedoids
import numpy as np

from data_utils import *

def EM_build_and_swap(args):
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    if args.metric != "L2":
        raise Exception("EM does not support metrics other than L2")

    imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
    metric = 'euclidean'
    kmedoids = KMedoids(n_clusters = args.num_medoids, metric = metric, random_state = None).fit(imgs)
    medoids = kmedoids.medoid_indices_.tolist()
    best_distances, closest_medoids = get_best_distances(medoids, imgs, metric='L2')
    loss = np.mean(best_distances)

    if args.verbose >= 1:
        print("Final results:")
        print(medoids)
        print(loss)

    return medoids, loss

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    EM_build_and_swap(args)
