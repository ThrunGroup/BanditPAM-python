from data_utils import *
import itertools



def build_sample_for_targets(imgs, targets, batch_size, best_distances, metric = None):
    # NOTE: Fix this with array broadcasting
    N = len(imgs)
    estimates = np.zeros(len(targets))
    # NOTE: Should this sampling be done with replacement? And do I need shuffling?
    # NOTE: Also, should the point be able to sample itself?
    tmp_refs = np.array(np.random.choice(N, size = batch_size, replace = False), dtype = 'int')
    for tar_idx, target in enumerate(targets):
        estimates[tar_idx] = np.mean(cost_fn(imgs, target, tmp_refs, best_distances, metric = metric))
    return estimates.round(DECIMAL_DIGITS)

def UCB_build(args, imgs, sigma):
    ### Parameters
    metric = args.metric
    N = len(imgs)
    p = 1. / (N * 100)
    num_samples = np.zeros(N)
    estimates = np.zeros(N)

    if len(args.warm_start_medoids) > 0:
        warm_start_medoids = list(map(int, args.warm_start_medoids.split(',')))
        medoids = warm_start_medoids.copy()
        num_medoids_found = len(medoids)
        best_distances, closest_medoids = np.array(get_best_distances(medoids, imgs, metric = metric))
    else:
        medoids = []
        num_medoids_found = 0
        best_distances = np.inf * np.ones(N)

    # Iteratively:
    # Pretend each previous arm is fixed.
    # For new arm candidate, true parameter is the TRUE loss when using the point as medoid
        # As a substitute, can measure the "gain" of using this point -- negative DECREASE in distance (the lower the distance, the better)
    # We sample this using UCB algorithm to get confidence bounds on what that loss will be
    # Update ucb, lcb, and empirical estimate by sampling WITH REPLACEMENT(NOTE)
        # If more than n points, just compute exactly -- otherwise, there's a failure mode where
        # Two points very close together require shittons of samples to distinguish their mean distance

    for k in range(num_medoids_found, args.num_medoids):
        if args.verbose >= 1:
            print("Finding medoid", k)

        ## Initialization
        step_count = 0
        candidates = range(N) # Initially, consider all points
        lcbs = 1000 * np.ones(N)
        ucbs = 1000 * np.ones(N)
        T_samples = np.zeros(N)
        exact_mask = np.zeros(N)

        # NOTE: What should this batch_size be? 20? Also note that this will result in (very minor) inefficiencies when batch_size > 1
        original_batch_size = 100
        base = 1.3 # Right now, use constant batch size

        while(len(candidates) > 0):

            if args.verbose >= 1:
                print("Step count:", step_count, ", Candidates:", len(candidates), candidates)

            # NOTE: tricky computations below
            this_batch_size = int(original_batch_size * (base**step_count))

            compute_exactly = np.where((T_samples + this_batch_size >= N) & (exact_mask == 0))[0]
            if len(compute_exactly) > 0:
                if args.verbose >= 1:
                    print("COMPUTING EXACTLY ON STEP COUNT", step_count)

                estimates[compute_exactly] = build_sample_for_targets(imgs, compute_exactly, N, best_distances, metric = metric)
                lcbs[compute_exactly] = estimates[compute_exactly]
                ucbs[compute_exactly] = estimates[compute_exactly]
                exact_mask[compute_exactly] = 1
                T_samples[compute_exactly] += N
                candidates = np.setdiff1d(candidates, compute_exactly) # Remove compute_exactly points from candidates so they're bounds don't get updated below

            if len(candidates) == 0: break # The last candidates were computed exactly

            # Don't update all estimates, just pulled arms
            estimates[candidates] = \
                ((T_samples[candidates] * estimates[candidates]) + (this_batch_size * build_sample_for_targets(imgs, candidates, this_batch_size, best_distances, metric = metric))) / (this_batch_size + T_samples[candidates])
            T_samples[candidates] += this_batch_size

            cb_delta = sigma * np.sqrt(np.log(1 / p) / T_samples[candidates])
            lcbs[candidates] = estimates[candidates] - cb_delta
            ucbs[candidates] = estimates[candidates] + cb_delta

            candidates = np.where( (lcbs < ucbs.min()) & (exact_mask == 0) )[0]
            step_count += 1

        new_medoid = np.arange(N)[ np.where( lcbs == lcbs.min() ) ]
        # Breaks exact ties with first. Also converts array to int.
        # This does indeed happen, for example in ucb k = 50, n = 100, s = 42, d = MNIST
        new_medoid = new_medoid[0]

        if args.verbose >= 1:
            # BUG: What about duplicates?
            print(np.where( lcbs == lcbs.min() ))
            print("New Medoid:", new_medoid)

        medoids.append(new_medoid)
        best_distances, closest_medoids = get_best_distances(medoids, imgs, metric = metric)
        print("Computed exactly for:", exact_mask.sum())

    if args.verbose >=1:
        print(medoids)

    return medoids





def swap_sample_for_targets(imgs, targets, current_medoids, batch_size, FastPAM1 = False, metric = None):
    '''
    Note that targets is a TUPLE ( [o_1, o_2, o_3, ... o_m], [n_1, n_2, ... n_m] )
    The corresponding target swaps are [o_1, n_1], [o_2, n_2], .... [o_m, n_m]

    This fn should measure the "gain" from performing the swap
    '''
    # NOTE: Fix this with array broadcasting
    # Also generalize and consolidate it with the fn of the same name in the build step
    orig_medoids = targets[0]
    new_medoids = targets[1]
    assert len(orig_medoids) == len(new_medoids), "Must pass equal number of original medoids and new medoids"
    # NOTE: Need to preserve order of swaps that are passed!!! Otherwise estimates will be for the wrong swaps!
    # NOTE: Otherwise, estimates won't be indexed properly -- only ok if we do 1 target at a time

    swaps = list(zip(orig_medoids, new_medoids)) # Zip doesn't throw an error for unequal lengths, it just drops extraneous points

    N = len(imgs)
    k = len(current_medoids)

    # NOTE: Should this sampling be done with replacement? And do I need shuffling?
    # NOTE: Also, should the point be able to sample itself? ANS: Yes, in the case of outliers, for example

    tmp_refs = np.array(np.random.choice(N, size = batch_size, replace = False), dtype='int')
    if FastPAM1:
        estimates = cost_fn_difference_FP1(imgs, swaps, tmp_refs, current_medoids, metric = metric) # NOTE: depends on other medoids too!
    else:
        estimates = cost_fn_difference(imgs, swaps, tmp_refs, current_medoids, metric = metric)

    return estimates.round(DECIMAL_DIGITS)


def UCB_swap(args, imgs, sigma, init_medoids):
    metric = args.metric
    k = len(init_medoids)
    N = len(imgs)
    p = 1. / (N * k * 1000)
    max_iter = 1e4
    # NOTE: Right now can compute amongst all k*n arms. Later make this k*(n-k)

    medoids = init_medoids.copy()
    best_distances, closest_medoids = get_best_distances(medoids, imgs, metric = metric)
    loss = np.mean(best_distances)
    iter = 0
    swap_performed = True
    while swap_performed and iter < max_iter: # not converged
        iter += 1

        # NOTE: Performing a lot of redundant computation in this loop.
        # Can add the trick from FastPAM1 to only compute the points whose medoids have changed
        # Or maybe just compute the "benefit" of swapping medoids, instead of total distance?

        # Identify best of k * (n-k) arms to swap by averaging new loss over all points
        # Identify Candidates
        # Get samples for candidates
        # NOTE: Right now doing k*N targets, but shouldn't allow medoids to swap each other: get k(n-k)
        candidates = np.array(list(itertools.product(range(k), range(N)))) # A candidate is a PAIR
        lcbs = 1000 * np.ones((k, N)) # NOTE: Instantiating these as np.inf gives runtime errors and nans. Find a better way to do this instead of using 1000
        estimates = 1000 * np.ones((k, N))
        ucbs = 1000 * np.ones((k, N))

        T_samples = np.zeros((k, N))
        exact_mask = np.zeros((k, N))

        # NOTE: What should this batch_size be? 20? Also note that this will result in (very minor) inefficiencies when batch_size > 1
        original_batch_size = 100
        base = 1.3 # Right now, use constant batch size

        step_count = 0
        while(len(candidates) > 0):
            if args.verbose >= 1:
                print("\nSWAP Step count:", step_count)#, ", Candidates:", len(candidates), candidates)

            # NOTE: tricky computations below
            this_batch_size = int(original_batch_size * (base**step_count))

            comp_exactly_condition = np.where((T_samples + this_batch_size >= N) & (exact_mask == 0))
            compute_exactly = np.array(list(zip(comp_exactly_condition[0], comp_exactly_condition[1])))
            if len(compute_exactly) > 0:
                if args.verbose >= 1:
                    print("COMPUTING EXACTLY ON STEP COUNT", step_count)

                # import ipdb; ipdb.set_trace()
                exact_accesses = (compute_exactly[:, 0], compute_exactly[:, 1])
                estimates[exact_accesses] = swap_sample_for_targets(imgs, exact_accesses, medoids, N, args.fast_pam1, metric = metric)
                lcbs[exact_accesses] = estimates[exact_accesses]
                ucbs[exact_accesses] = estimates[exact_accesses]
                exact_mask[exact_accesses] = 1
                T_samples[exact_accesses] += N

                cand_condition = np.where( (lcbs < ucbs.min()) & (exact_mask == 0) ) # BUG: Fix this since it's 2D
                candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))

            if len(candidates) == 0: break # The last candidates were computed exactly

            # Don't update all estimates, just pulled arms
            accesses = (candidates[:, 0], candidates[:, 1])
            new_samples = swap_sample_for_targets(imgs, accesses, medoids, this_batch_size, args.fast_pam1, metric = metric)
            estimates[accesses] = \
                ((T_samples[accesses] * estimates[accesses]) + (this_batch_size * new_samples)) / (this_batch_size + T_samples[accesses])
            T_samples[accesses] += this_batch_size
            cb_delta = sigma * np.sqrt(np.log(1 / p) / T_samples[accesses])
            lcbs[accesses] = estimates[accesses] - cb_delta
            ucbs[accesses] = estimates[accesses] + cb_delta

            cand_condition = np.where( (lcbs < ucbs.min()) & (exact_mask == 0) ) # BUG: Fix this since it's 2D
            candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))
            step_count += 1

        # Choose the minimum amongst all losses and perform the swap
        # NOTE: possible to get first elem of zip object without converting to list?
        best_swaps = zip( np.where(lcbs == lcbs.min())[0], np.where(lcbs == lcbs.min())[1] )
        best_swaps = list(best_swaps)
        best_swap = best_swaps[0]

        print("Computed exactly for:", exact_mask.sum())
        performed_or_not, medoids, loss = medoid_swap(medoids, best_swap, imgs, loss, args)
        if performed_or_not == "NO SWAP PERFORMED":
            break

    return medoids, iter

def UCB_build_and_swap(args):
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
    if 'B' in args.build_ao_swap:
        built_medoids = UCB_build(args, imgs, sigma)
        print("Built medoids", built_medoids)

    swapped_medoids = []
    swap_iters = 0
    if 'S' in args.build_ao_swap:
        if built_medoids is None and len(args.warm_start_medoids) < args.num_medoids:
            raise Exception("Invalid call to Swap step")

        swapped_medoids, swap_iters = UCB_swap(args, imgs, sigma, built_medoids.copy())
        print("Final medoids", swapped_medoids)

    return built_medoids, swapped_medoids, swap_iters

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    UCB_build_and_swap(args)
