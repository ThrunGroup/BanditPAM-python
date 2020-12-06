'''
The only existing implementation of CLARANS, in pyclustering, is far too slow.
We implement our own below.

Based on
Raymond T. Ng and Jiawei Han,
“CLARANS: A Method for Clustering Objectsfor Spatial Data Mining”
IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. 14, NO. 5,
SEPTEMBER/OCTOBER 2002.
http://www.cs.ecu.edu/dingq/CSCI6905/readings/CLARANS.pdf
'''

import numpy as np
import copy

from data_utils import *


def CLARANS_build_and_swap(args):
    '''
    The CLARANS algorithm works by:
    a) Randomly initializing a set of medoids
    b) Repeatedly swapping a random medoid-nonmedoid pair, if it would lower the
        overall loss, MAXNEIGHBOR times
    c) Repeating this entire process NUMLOCAL times to find NUMLOCAL local
        minima of medoid assignments
    d) Returning the best set of medoids that was found throughout the entire
        process
    '''

    NUMLOCAL = 20
    MAXNEIGHBOR = args.num_medoids

    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    if args.metric == 'TREE':
        raise Exception("CLARANS is not yet implemented for non-matrix data")
    else:
        imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]

    if args.metric != "L2":
        raise Exception("CLARANS does not support metrics other than L2")

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

    if args.verbose >= 1:
        print("Final results:")
        print(best_medoids)
        print(best_loss)

    return best_medoids, best_loss

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    CLARANS_build_and_swap(args)
