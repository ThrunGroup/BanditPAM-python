from data_utils import *

def naive_build(args, imgs):
    '''
    Naively instantiates the medoids, corresponding to the BUILD step.
    Algorithm does so in a greedy way:
        for k in range(num_medoids):
            Add the medoid that will lead to lowest lost, conditioned on the
            previous medoids being fixed
    '''
    d_count = 0
    N = len(imgs)

    if len(args.warm_start_medoids) > 0:
        warm_start_medoids = list(map(int, args.warm_start_medoids.split(',')))
        medoids = warm_start_medoids
        num_medoids_found = len(warm_start_medoids)
        best_distances = get_best_distances(medoids, imgs)
    else:
        medoids = []
        num_medoids_found = 0
        best_distances = [float('inf') for _ in range(N)]

    import ipdb
    for k in range(num_medoids_found, args.num_medoids):
        if args.verbose >= 1:
            print("Finding medoid", k)

        # Greedily choose the point which minimizes the loss
        best_loss = float('inf')
        best_medoid = -1
        # if k == 43: ipdb.set_trace()
        for target in range(N):
            if (target + 1) % 100 == 0 and args.verbose >= 1:
                print(target)
            # if target in medoids: continue # Skip existing medoids NOTE: removing this optimization for complexity comparison

            losses = np.zeros(N)
            # NOTE: SHould reference be allowed to be the target (sample itself)?
            for reference in range(N):
                # if reference in medoids: continue # Skip existing medoids NOTE: removing this optimization for complexity comparison
                d_r_t = d(imgs[target], imgs[reference])
                d_count += 1
                losses[reference] = d_r_t if d_r_t < best_distances[reference] else best_distances[reference]
                # losses[reference] = min(d_r_t, best_distances[reference])

            if (target == 26 or target == 39) and k == 43: ipdb.set_trace()
            loss = np.mean(losses).round(DECIMAL_DIGITS)
            if loss < best_loss:
                best_loss = loss
                best_medoid = target

        # Once we have chosen the best medoid, reupdate the best distances
        # Don't do this OTF to avoid overwriting best_distances or requiring deep copy
        # Otherwise, we would have side-effects when recomputing best_distances and recursively backtracking
        # Also don't include these distance computations in the running metric because they could be computed OTF / tracked
        if args.verbose >= 1:
            print("Medoid Found: ", k, best_medoid)

        medoids.append(best_medoid)
        best_distances = get_best_distances(medoids, imgs)

    if args.verbose >= 1:
        print(medoids)

    return medoids

def naive_swap(args, imgs, init_medoids):
    k = len(init_medoids)
    N = len(imgs)
    max_iter = 1e4
    # NOTE: Right now can compute amongst all k*n arms. Later make this k*(n-k)

    medoids = init_medoids.copy()
    loss = np.mean(get_best_distances(medoids, imgs))
    iter = 0
    swap_performed = True
    while swap_performed and iter < max_iter: # not converged
        iter += 1

        # Identify best of k * (n-k) arms to swap by averaging new loss over all points
        new_losses = np.inf * np.ones((k, N))

        for k_idx, orig_medoid in enumerate(medoids):
            for swap_candidate in range(N):
                new_medoids = medoids.copy()
                new_medoids.remove(orig_medoid)
                new_medoids.append(swap_candidate)
                # NOTE: new_medoids's points need not be sorted, like original medoids!

                # NOTE: This get_best_distances fn is going to cost lots of calls! To include them or not? I think yes -- this is indeed what we are trying to cut down?
                tmp_best_distances = np.mean(get_best_distances(new_medoids, imgs))
                new_losses[k_idx, swap_candidate] = tmp_best_distances

        # Choose the minimum amongst all losses and perform the swap
        # NOTE: possible to get first elem of zip object without converting to list?
        best_swaps = zip( np.where(new_losses == new_losses.min())[0], np.where(new_losses == new_losses.min())[1])
        best_swaps = list(best_swaps)
        best_swap = best_swaps[0]


        new_medoids = medoids.copy()
        new_medoids.remove(medoids[best_swap[0]])
        new_medoids.append(best_swap[1])
        # Check new loss
        new_loss = np.mean(get_best_distances(new_medoids, imgs))
        performed_or_not = ''
        if new_loss < loss:
            performed_or_not = "SWAP PERFORMED"
            loss = new_loss
            swap_performed = True
            medoids = new_medoids
        else:
            performed_or_not = "NO SWAP PERFORMED"
            break

        if args.verbose >= 1:
            print("Tried to swap", medoids[best_swap[0]], "with", best_swap[1])
            print(performed_or_not)
            print("Old loss:", loss)
            print("New loss:", new_loss)

    return medoids

def naive_build_and_swap(args):
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
    built_medoids = naive_build(args, imgs)
    print("Built medoids", built_medoids)
    return built_medoids
    # swapped_medoids = naive_swap(args, imgs, built_medoids)
    # print("Final medoids", swapped_medoids)
    # return swapped_medoids

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    naive_build_and_swap(args)
