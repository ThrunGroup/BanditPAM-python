from data_utils import *
import itertools


def build_sample_for_targets(imgs, targets, batch_size, best_distances):
    # NOTE: Fix this with array broadcasting
    N = len(imgs)
    estimates = np.zeros(len(targets))
    # NOTE: Should this sampling be done with replacement? And do I need shuffling?
    # NOTE: Also, should the point be able to sample itself?
    tmp_refs = np.array(np.random.choice(N, size = batch_size, replace = False), dtype='int')
    for tar_idx, target in enumerate(targets):
        distances = np.zeros(batch_size)
        for tmp_idx, tmp in enumerate(tmp_refs):
            distances[tmp_idx] = cost_fn(imgs, target, tmp, best_distances) # NOTE: depends on other medoids too!
        estimates[tar_idx] = np.mean(distances)
    return estimates

def UCB_build(args, imgs, sigma):
    ### Parameters
    N = len(imgs)
    p = 1e-6
    num_samples = np.zeros(N)
    estimates = np.zeros(N)

    if len(args.warm_start_medoids) > 0:
        warm_start_medoids = list(map(int, args.warm_start_medoids.split(',')))
        medoids = warm_start_medoids
        num_medoids_found = len(warm_start_medoids)
        best_distances = get_best_distances(medoids, imgs)
    else:
        medoids = []
        num_medoids_found = 0
        best_distances = [float('inf') for _ in range(N)]

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
        base = 1 # Right now, use constant batch size

        # Pull arms, update ucbs and lcbs
        while(len(candidates) > 1): # NOTE: Should also probably restrict absolute distance in cb_delta?
            if args.verbose >= 1:
                print("Step count:", step_count, ", Candidates:", len(candidates), candidates)

            # NOTE: tricky computations below
            this_batch_size = original_batch_size * (base**step_count)

            # Don't update all estimates, just pulled arms
            estimates[candidates] = \
                ((T_samples[candidates] * estimates[candidates]) + (this_batch_size * build_sample_for_targets(imgs, candidates, this_batch_size, best_distances))) / (this_batch_size + T_samples[candidates])
            T_samples[candidates] += this_batch_size

            # NOTE: Can further optimize this by putting this above the sampling paragraph just above this.
            compute_exactly = np.where((T_samples >= N) & (exact_mask == 0))[0]
            if len(compute_exactly) > 0:
                if args.verbose >= 1:
                    print("COMPUTING EXACTLY ON STEP COUNT", step_count)

                estimates[compute_exactly] = build_sample_for_targets(imgs, compute_exactly, N, best_distances)
                lcbs[compute_exactly] = estimates[compute_exactly]
                ucbs[compute_exactly] = estimates[compute_exactly]
                exact_mask[compute_exactly] = 1
                T_samples[compute_exactly] += N
                candidates = np.setdiff1d(candidates, compute_exactly) # Remove compute_exactly points from candidates so they're bounds don't get updated below

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
        best_distances = get_best_distances(medoids, imgs)

    if args.verbose >=1:
        print(medoids)

    return medoids





def swap_sample_for_targets(imgs, targets, current_medoids, batch_size):
    '''
    Note that targets is a TUPLE (original_medoid, new_candidate)
    This fn should measure the "gain" from performing the swap
    '''
    # NOTE: Fix this with array broadcasting
    # Also generalize and consolidate it with the fn of the same name in the build step
    assert len(targets) == 1, "This function is not indexed properly for more than 1 target"
    N = len(imgs)
    k = len(current_medoids)
    estimates = np.zeros(len(targets))
    # NOTE: Should this sampling be done with replacement? And do I need shuffling?
    # NOTE: Also, should the point be able to sample itself?
    tmp_refs = np.array(np.random.choice(N, size = batch_size, replace = False), dtype='int')
    best_distances = get_best_distances(current_medoids, imgs)
    for tar_idx, target in enumerate(targets): # NOTE: Here, target is a PAIR
        estimates[tar_idx] = cost_fn_difference_total(imgs[tmp_refs], imgs, target, current_medoids, best_distances) # NOTE: depends on other medoids too!
    # NOTE: I don't think estimates is indexed properly, i.e. by tuples
    return estimates


def UCB_swap(args, imgs, sigma, init_medoids):
    p = 1e-8
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
        base = 1 # Right now, use constant batch size

        step_count = 0
        while(len(candidates) > 1):
            if args.verbose >= 1:
                print("\nSWAP Step count:", step_count)#, ", Candidates:", len(candidates), candidates)

            # NOTE: tricky computations below
            this_batch_size = original_batch_size * (base**step_count)

            # Don't update all estimates, just pulled arms
            for c in candidates:
                index_tup = (c[0], c[1])
                new_samples = swap_sample_for_targets(imgs, [index_tup], medoids, this_batch_size)
                estimates[index_tup] = \
                    ((T_samples[index_tup] * estimates[index_tup]) + (this_batch_size * new_samples)) / (this_batch_size + T_samples[index_tup])
                T_samples[index_tup] += this_batch_size
                cb_delta = sigma * np.sqrt(np.log(1 / p) / T_samples[index_tup])
                lcbs[index_tup] = estimates[index_tup] - cb_delta
                ucbs[index_tup] = estimates[index_tup] + cb_delta

            # NOTE: Can further optimize this by putting this above the sampling paragraph just above this.
            comp_exactly_condition = np.where((T_samples >= N) & (exact_mask == 0))
            compute_exactly = list(zip(comp_exactly_condition[0], comp_exactly_condition[1]))
            if len(compute_exactly) > 0:
                if args.verbose >= 1:
                    print("COMPUTING EXACTLY ON STEP COUNT", step_count)

                for c in compute_exactly:
                    index_tup = (c[0], c[1])
                    estimates[index_tup] = swap_sample_for_targets(imgs, [index_tup], medoids, N)
                    lcbs[index_tup] = estimates[index_tup]
                    ucbs[index_tup] = estimates[index_tup]
                    exact_mask[index_tup] = 1
                    T_samples[index_tup] += N
                cand_condition = np.where( (lcbs < ucbs.min()) & (exact_mask == 0) ) # BUG: Fix this since it's 2D
                candidates = list(zip(cand_condition[0], cand_condition[1]))

            cand_condition = np.where( (lcbs < ucbs.min()) & (exact_mask == 0) ) # BUG: Fix this since it's 2D
            candidates = list(zip(cand_condition[0], cand_condition[1]))
            step_count += 1

        # Choose the minimum amongst all losses and perform the swap
        # NOTE: possible to get first elem of zip object without converting to list?
        best_swaps = zip( np.where(lcbs == lcbs.min())[0], np.where(lcbs == lcbs.min())[1] )
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

def UCB_build_and_swap(args):
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
    built_medoids = UCB_build(args, imgs, sigma)
    print("Built medoids", built_medoids)
    return built_medoids
    # swapped_medoids = UCB_swap(args, imgs, sigma, built_medoids)
    # print("Final medoids", swapped_medoids)
    # return swapped_medoids

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    UCB_build_and_swap(args)
