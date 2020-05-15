'''
THIS CODE IS UNIFINISHED. FP2 IS NOT WORKING.
NEED TO MERGE THIS with  NAIVE_PAM_v1 (many changes made to Naive_PAM_v1)
'''


'''
This is an "optimized" version of PAM, and should be used to certify
correctness (in a relatively slow way).

In particular, it contains the following optimizations over naive_pam:
1. n --> (n-k) [BUILD done, SWAP not gonna do]
2. Array broadcasting instead of looping over losses [BUILD done, SWAP done]
3. FastPAM1 [SWAP done]
4. FastPAM2 [SWAP WIP]
5. LAB [WIP]
'''


from data_utils import *
import itertools

def naive_build(args, imgs):
    '''
    Naively instantiates the medoids, corresponding to the BUILD step.
    Algorithm does so in a greedy way:
        for k in range(num_medoids):
            Add the medoid that will lead to lowest lost, conditioned on the
            previous medoids being fixed
    '''
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
            ref_imgs = imgs[refs] # should be == imgs, since sampling all reference points #NOTE: what if we randomly choose subsample? Impt baseline
            losses[t_reidx] = np.mean( np.minimum(d(imgs[t].reshape(1, -1), ref_imgs, metric = metric), best_distances)).round(DECIMAL_DIGITS)

        best_loss_reidx = np.where(losses == losses.min())[0][0] # NOTE: what about duplicates? Chooses first I believe
        best_medoid = target_idcs[best_loss_reidx]

        medoids.append(best_medoid)
        best_distances, closest_medoids = get_best_distances(medoids, imgs, metric = metric)

        if args.verbose >= 1:
            print("Medoid Found: ", k, best_medoid)
            print(medoids)

    return medoids



def naive_swap(args, imgs, init_medoids):
    metric = args.metric
    k = len(init_medoids)
    N = len(imgs)
    max_iter = 1e4
    # NOTE: Right now can compute amongst all k*n arms. Later make this k*(n-k) -- don't consider swapping medoid w medoid

    medoids = init_medoids.copy()
    best_distances, closest_medoids, second_best_distances = get_best_distances(medoids, imgs, return_second_best = True, metric = metric)
    loss = np.mean(best_distances)
    iter = 0
    swap_performed = True
    while swap_performed and iter < max_iter: # not converged
        print("SWAP iter:", iter)
        iter += 1
        new_losses = np.inf * np.ones((k, N))

        swap_candidates = np.array(list(itertools.product(range(k), range(N)))) # A candidate is a PAIR

        if args.fast_pam1:
            new_losses = cost_fn_difference_FP1(imgs, swap_candidates, range(N), medoids, metric = metric).reshape(k, N)
        else:
            new_losses = cost_fn_difference(imgs, swap_candidates, range(N), medoids, metric = metric).reshape(k, N)

        new_losses = new_losses.round(DECIMAL_DIGITS)


        if args.fast_pam2:
            '''
            NOTE: This does not work yet!! Just performs original swaps
            '''
            tau = 0.0
            best_k_swaps_row = new_losses.argmin(axis = 1) # k potential swaps to perform

            # NOTE: Can simplify best_k_swaps to remove k in first column, it's redundant
            best_k_swaps = np.array(list(zip(range(k), best_k_swaps_row)))
            parent_losses = np.copy(new_losses[best_k_swaps.T[0], best_k_swaps.T[1]]) #NOTE: Fancy

            best_swap = np.unravel_index(new_losses.argmin(), new_losses.shape)
            performed_or_not, medoids, loss = medoid_swap(medoids, best_swap, imgs, loss, args) # perform the best swap
            if performed_or_not == "NO SWAP PERFORMED":
                break
            else:
                child_swap_performed = True
                # for the remaining <= k candidates, compute the new cost_fn_difference (for at most k swaps now, not k*N)
                while child_swap_performed:
                    if args.fast_pam1:
                        child_losses = cost_fn_difference_FP1(imgs, best_k_swaps, range(N), medoids, metric = metric).reshape(k)
                    else:
                        child_losses = cost_fn_difference(imgs, best_k_swaps, range(N), medoids, metric = metric).reshape(k)

                    if child_losses.min() >= 0:
                        child_swap_performed = False
                        break # only exit condition

                    assert child_losses.min() < 0, "Something's wrong"
                    # Take the one with the best new DTD
                    best_swap_idx = child_losses.argmin()
                    best_child_swap = best_k_swaps[best_swap_idx]
                    best_child_swap_loss = child_losses[best_swap_idx]

                    # if it's at least tau * new_losses of its swap, then:
                    #   perform the swap and repeat
                    if best_child_swap_loss < tau * child_losses[best_swap_idx]: # NOTE: < instead of <= to avoid errors with 0s
                        performed_or_not, medoids, loss = medoid_swap(medoids, best_child_swap, imgs, loss, args) # perform the best swap
                        if performed_or_not == "NO SWAP PERFORMED":
                            raise Exception("This should never happen")
                    else:
                        # delete best_swap_idx from best_k_swaps
                        best_k_swaps = np.delete(best_k_swaps, best_swap_idx, axis = 0)

                    parent_losses = np.copy(child_losses)


        else:
            best_swap = np.unravel_index(new_losses.argmin(), new_losses.shape)
            performed_or_not, medoids, loss = medoid_swap(medoids, best_swap, imgs, loss, args)
            if performed_or_not == "NO SWAP PERFORMED":
                break

    return medoids, iter

def naive_build_and_swap(args):
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
    # import ipdb; ipdb.set_trace()
    if 'B' in args.build_ao_swap:
        built_medoids = naive_build(args, imgs)
        print("Built medoids", built_medoids)

    swapped_medoids = []
    swap_iters = 0
    if 'S' in args.build_ao_swap:
        if built_medoids is None and len(args.warm_start_medoids) < args.num_medoids:
            raise Exception("Invalid call to Swap step")

        swapped_medoids, swap_iters = naive_swap(args, imgs, built_medoids.copy())
        print("Final medoids", swapped_medoids)

    return built_medoids, swapped_medoids, swap_iters

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    naive_build_and_swap(args)
