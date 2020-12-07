'''
This is an "optimized" version of PAM, and should be used to certify
correctness, but is relatively slow since it's still O(kn^2) in each iteration.

In particular, it contains the following optimizations over naive_pam:
1. n --> (n-k) in the BUILD step (but not in the SWAP step for convenience)
2. Array broadcasting instead of looping over losses
3. FastPAM1
'''
import itertools

from data_utils import *

def naive_build(args, imgs):
    '''
    Naively instantiates the medoids, corresponding to the BUILD step.
    Algorithm does so in a greedy way:
        for k in range(num_medoids):
            Add the medoid that will lead to lowest lost, conditioned on the
            previous medoids being fixed
    '''
    B_logstring = init_logstring()
    metric = args.metric
    d_count = 0
    N = len(imgs)

    if len(args.warm_start_medoids) > 0:
        warm_start_medoids = list(map(int, args.warm_start_medoids.split(',')))
        medoids = warm_start_medoids
        num_medoids_found = len(warm_start_medoids)
        best_distances, closest_medoids = get_best_distances(medoids, imgs, metric = metric)
    else:
        medoids = []
        num_medoids_found = 0
        best_distances = [float('inf') for _ in range(N)]

    for k in range(num_medoids_found, args.num_medoids):
        target_idcs = np.arange(N)
        target_idcs = np.delete(target_idcs, medoids) # n --> (n-k)
        target_imgs = imgs[target_idcs]
        losses = np.zeros(len(target_idcs))
        for t_reidx, t in enumerate(target_idcs):
            refs = np.arange(N)
            ref_imgs = imgs[refs]
            if metric == 'TREE':
                losses[t_reidx] = np.mean( np.minimum(d_tree(imgs[t], ref_imgs, metric = metric), best_distances)).round(DECIMAL_DIGITS)
            else:
                losses[t_reidx] = np.mean( np.minimum(d(imgs[t].reshape(1, -1), ref_imgs, metric = metric), best_distances)).round(DECIMAL_DIGITS)

        best_loss_reidx = np.where(losses == losses.min())[0][0]
        best_medoid = target_idcs[best_loss_reidx]
        medoids.append(best_medoid)
        best_distances, closest_medoids = get_best_distances(medoids, imgs, metric = metric)

        if args.verbose >= 1:
            print("Medoid Found: ", k, best_medoid)
            print(medoids)

        B_logstring = update_logstring(B_logstring, k, best_distances, N, None, None)

    return medoids, B_logstring

def naive_swap(args, imgs, init_medoids):
    '''
    Iteratively swap the medoid-nonmedoid pair that would lower the loss the
    most, amongst all such possible swaps. Iterated until convergence, i.e. when
    the total loss can no longer be lowered in such a manner.
    '''

    S_logstring = init_logstring()
    metric = args.metric
    k = len(init_medoids)
    N = len(imgs)
    max_iter = 1e4

    medoids = init_medoids.copy()
    best_distances, closest_medoids, second_best_distances = get_best_distances(medoids, imgs, return_second_best = True, metric = metric)
    loss = np.mean(best_distances)
    iter = 0
    swap_performed = True
    while swap_performed and iter < max_iter:
        iter += 1
        new_losses = np.inf * np.ones((k, N))
        swap_candidates = np.array(list(itertools.product(range(k), range(N))))

        if args.fast_pam1:
            new_losses = cost_fn_difference_FP1(imgs, swap_candidates, range(N), medoids, metric = metric).reshape(k, N)
        else:
            new_losses = cost_fn_difference(imgs, swap_candidates, range(N), medoids, metric = metric).reshape(k, N)

        new_losses = new_losses.round(DECIMAL_DIGITS)
        best_swaps = zip( np.where(new_losses == new_losses.min())[0], np.where(new_losses == new_losses.min())[1])
        best_swaps = list(best_swaps) # Is it possible to get first elem of zip object without converting to list?
        best_swap = best_swaps[0]

        old_medoid_x = medoids[best_swap[0]]
        new_medoid_x = best_swap[1]
        performed_or_not, medoids, loss = medoid_swap(medoids, best_swap, imgs, loss, args)
        S_logstring = update_logstring(S_logstring, iter - 1, loss, N * k, None, None, swap = (old_medoid_x, new_medoid_x))
        if performed_or_not == "NO SWAP PERFORMED":
            break

    return medoids, S_logstring, iter, loss

def naive_build_and_swap(args):
    '''
    Run the entire PAM algorithm, both the BUILD step and the SWAP step
    '''
    num_swaps = -1
    final_loss = -1

    # Randomly sample the data
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    if args.metric == 'TREE':
        imgs = np.random.choice(total_images, size = args.sample_size, replace = False)
    else:
        imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]

    # Perform BUILD step
    built_medoids = []
    B_logstring = {}
    if 'B' in args.build_ao_swap:
        built_medoids, B_logstring = naive_build(args, imgs)
        print("Built medoids", built_medoids)

    # Perform SWAP step
    swapped_medoids = []
    S_logstring = {}
    if 'S' in args.build_ao_swap:
        if built_medoids == [] and len(args.warm_start_medoids) < args.num_medoids:
            raise Exception("Invalid call to Swap step")

        if built_medoids == []:
            init_medoids = list(map(int, args.warm_start_medoids.split(',')))
            print("Swap init medoids:", init_medoids)
        else:
            init_medoids = built_medoids.copy()

        swapped_medoids, S_logstring, num_swaps, final_loss = naive_swap(args, imgs, init_medoids)
        print("Final medoids", swapped_medoids)

    return built_medoids, swapped_medoids, B_logstring, S_logstring, num_swaps, final_loss, -1 #-1 placeholder for uniq_d

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    naive_build_and_swap(args)
