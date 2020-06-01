'''
Using the EM-style (Voronoi Iteration) approach to computing k-medoids
According to Martin, sklearn_extra does this
'''

#TODO: Verify that under the hood, sklearn uses EM-style

from sklearn_extra.cluster import KMedoids
import numpy as np

from data_utils import *

def EM_build_and_swap(args):
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    if args.metric == 'TREE':
        raise Exception("I don't think sklearn_extra works for non-matrices")
        imgs = np.random.choice(total_images, size = args.sample_size, replace = False)
    else:
        imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]

    if args.metric == "L2":
        metric = 'euclidean'
    else:
        raise Exception("bad metric for sklearn")

    kmedoids = KMedoids(n_clusters = args.num_medoids, metric = metric, random_state = None).fit(imgs)
    medoids = kmedoids.medoid_indices_.tolist()
    best_distances, closest_medoids = get_best_distances(medoids, imgs, metric='L2')
    # reference_best_distances, reference_closest_medoids = get_best_distances([714,694,765,507,737], imgs, metric='L2')
    # print(np.mean(reference_best_distances))
    loss = np.mean(best_distances)
    print(loss)
    return loss


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    EM_build_and_swap(args)
