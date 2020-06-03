import numpy as np
import copy

from data_utils import *

'''
The pyclustering implementation of CLARANS is far too slow.
We implement our own below.
'''

# def CLARANS_build_and_swap(args):
#     num_swaps = -1
#     final_loss = -1
#
#     total_images, total_labels, sigma = load_data(args)
#     np.random.seed(args.seed)
#     if args.metric == 'TREE':
#         raise Exception("I don't think CLARANS works for non-matrices")
#         imgs = np.random.choice(total_images, size = args.sample_size, replace = False)
#     else:
#         imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
#
#     if args.metric == "L2":
#         metric = 'euclidean'
#     else:
#         raise Exception("bad metric for CLARANS")
#
#     """
#     Copied from documentation:
#     @brief Constructor of clustering algorithm CLARANS.
#     @details The higher the value of maxneighbor, the closer is CLARANS to K-Medoids, and the longer is each search of a local minima.
#
#     @param[in] data (list): Input data that is presented as list of points (objects), each point should be represented by list or tuple.
#     @param[in] number_clusters (uint): Amount of clusters that should be allocated.
#     @param[in] numlocal (uint): The number of local minima obtained (amount of iterations for solving the problem).
#     @param[in] maxneighbor (uint): The maximum number of neighbors examined.
#     """
#     NUMLOCAL = 10
#     MAXNEIGHBOR = args.num_medoids
#     # import ipdb; ipdb.set_trace()
#     clarans_instance = clarans(imgs, args.num_medoids, NUMLOCAL, MAXNEIGHBOR)
#     clarans_instance.process()
#     medoids = clarans_instance.get_medoids()
#     best_distances, closest_medoids = get_best_distances(medoids, imgs, metric='L2')
#     # reference_best_distances, reference_closest_medoids = get_best_distances([714,694,765,507,737], imgs, metric='L2')
#     # print(np.mean(reference_best_distances))
#     loss = np.mean(best_distances)
#     print(loss)
#      return loss

def CLARANS_build_and_swap(args):
    NUMLOCAL = 20
    MAXNEIGHBOR = args.num_medoids

    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    if args.metric == 'TREE':
        raise Exception("I don't think CLARANS works for non-matrices")
        imgs = np.random.choice(total_images, size = args.sample_size, replace = False)
    else:
        imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]

    if args.metric != "L2":
        raise Exception("bad metric for CLARANS")

    best_loss = float('inf')
    best_medoids = []

    for i in range(NUMLOCAL):
        medoids = np.random.choice(len(imgs), args.num_medoids, replace = False).tolist()
        best_distances, closest_medoids = get_best_distances(medoids, imgs, metric = args.metric)
        loss = np.mean(best_distances)
        for j in range(MAXNEIGHBOR):
            swap = [np.random.choice(len(medoids)), np.random.choice(len(imgs))]
            performed_or_not, medoids, loss = medoid_swap(medoids, swap, imgs, loss, args)

        if loss < best_loss:
            best_medoids = copy.deepcopy(medoids)
            best_loss = loss

    print("Final results:")
    print(best_medoids)
    print(best_loss)
    return best_medoids, best_loss

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    CLARANS_build_and_swap(args)
