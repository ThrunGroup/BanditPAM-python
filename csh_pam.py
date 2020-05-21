from data_utils import *
import itertools
import math


'''
NOTE:
- Need to make sigma adaptive as in UCB
- Need to tune T and initial batch size
'''



def build_sample_for_targets(imgs, targets, batch_size, best_distances, metric = None):
    # NOTE: Fix this with array broadcasting
    N = len(imgs)
    estimates = np.zeros(len(targets))
    # NOTE: Should this sampling be done with replacement? And do I need shuffling?
    # NOTE: Also, should the point be able to sample itself?
    tmp_refs = np.array(np.random.choice(N, size = batch_size, replace = False), dtype = 'int')
    for tar_idx, target in enumerate(targets):
        if best_distances[0] == np.inf:
            # No medoids have been assigned, can't use the difference in loss
            costs = cost_fn(imgs, target, tmp_refs, best_distances, metric = metric, use_diff = False)
        else:
            costs = cost_fn(imgs, target, tmp_refs, best_distances, metric = metric, use_diff = True)

        estimates[tar_idx] = np.mean(costs)

    return estimates.round(DECIMAL_DIGITS)

def CSH_build(args, imgs, sigma):
    B_logstring = init_logstring()
    metric = args.metric
    N = len(imgs)
    num_samples = np.zeros(N)
    estimates = np.zeros(N)
    T = 16 * N * np.log(N)

    if len(args.warm_start_medoids) > 0:
        warm_start_medoids = list(map(int, args.warm_start_medoids.split(',')))
        medoids = warm_start_medoids.copy()
        num_medoids_found = len(medoids)
        best_distances, closest_medoids = np.array(get_best_distances(medoids, imgs, metric = metric))
    else:
        medoids = []
        num_medoids_found = 0
        best_distances = np.inf * np.ones(N)

    for k in range(num_medoids_found, args.num_medoids):
        if args.verbose >= 1:
            print("Finding medoid", k)

        step_count = 0
        candidates = range(N)
        T_samples = np.zeros(N)
        exact_mask = np.zeros(N)

        while(len(candidates) > 0):
            if args.verbose >= 1:
                print("Step count:", step_count, ", Candidates:", len(candidates))#, candidates)

            # NOTE: Potential issues of surpassing sampling budget T if there are ties around the median
            T_r = 10 * T / (len(candidates) * math.ceil(np.log2(N)))
            this_batch_size = int(min(max(1, T_r), N))

            compute_exactly = np.where((T_samples + this_batch_size >= N) & (exact_mask == 0))[0]
            if len(compute_exactly) > 0:
                if args.verbose >= 1:
                    print("COMPUTING EXACTLY ON STEP COUNT", step_count)

                estimates[compute_exactly] = build_sample_for_targets(imgs, compute_exactly, N, best_distances, metric = metric)
                exact_mask[compute_exactly] = 1
                T_samples[compute_exactly] += N
                candidates = np.setdiff1d(candidates, compute_exactly)

            if len(candidates) == 0: break

            sample_costs = build_sample_for_targets(imgs, candidates, this_batch_size, best_distances, metric = metric)
            estimates[candidates] = \
                ((T_samples[candidates] * estimates[candidates]) + (this_batch_size * sample_costs)) / (this_batch_size + T_samples[candidates])
            T_samples[candidates] += this_batch_size

            # NOTE: Need estimates[candidates] below, in case some estimates drop below a prior median -- don't want to pick up the old ones that were discarded
            # In other words, only estimates[candidates] are the active arms. But what about those computed exactly and removed?
            # Actually... computing should only ever happen for all arms together at the last step
            # NOTE: Potential issue where old arms that were discarded get picked up again if the median goes *UP*
            # Resolved by taking intersection with current candidates
            median_return = np.median(estimates[candidates])
            # NOTE: It is possible that median == max! For example, with sparse data, or even on MNIST with N = 10000. Then this may cause a problem of resampling a lot
            if median_return == np.max(estimates[candidates]):
                    candidates = np.intersect1d(candidates, np.where(estimates <= median_return)[0]) # Strictly less to chop off heavy ail
            else:
                candidates = np.intersect1d(candidates, np.where(estimates <= median_return)[0])

            step_count += 1

        new_medoid = np.arange(N)[ np.where( estimates == estimates.min() ) ]
        new_medoid = new_medoid[0] # Breaks exact ties with first. Also converts array to int.
        if args.verbose >= 1:
            print("New Medoid:", new_medoid)

        medoids.append(new_medoid)
        best_distances, closest_medoids = get_best_distances(medoids, imgs, metric = metric)
        print("Computed exactly for:", exact_mask.sum())
        B_logstring = update_logstring(B_logstring, k, best_distances, exact_mask.sum(), 0.0, T)

    return medoids, B_logstring





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
        estimates = cost_fn_difference_FP1(imgs, swaps, tmp_refs, current_medoids, metric = metric, return_sigma = False)
    else:
        estimates = cost_fn_difference(imgs, swaps, tmp_refs, current_medoids, metric = metric)

    return estimates.round(DECIMAL_DIGITS)


def CSH_swap(args, imgs, sigma, init_medoids):
    S_logstring = init_logstring()
    metric = args.metric
    k = len(init_medoids)
    N = len(imgs)
    max_iter = 1e1
    T = 16 * k * N * np.log(N)

    medoids = init_medoids.copy()
    # NOTE: best_distances is NOT updated in future rounds - the analogy from build is broken. Maybe rename the variable
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
        estimates = 1000 * np.ones((k, N))

        T_samples = np.zeros((k, N))
        exact_mask = np.zeros((k, N))

        step_count = 0
        while(len(candidates) > 0):
            if args.verbose >= 1:
                print("SWAP Step count:", step_count, ", Candidates:", len(candidates))#, candidates)

            # NOTE: Potential issues of surpassing sampling budget T if there are ties around the median
            T_r = T / (candidates.size * math.ceil(np.log2(N)))
            this_batch_size = int(min(max(1, T_r), N))

            comp_exactly_condition = np.where((T_samples + this_batch_size >= N) & (exact_mask == 0))
            compute_exactly = np.array(list(zip(comp_exactly_condition[0], comp_exactly_condition[1])))
            if len(compute_exactly) > 0:
                if args.verbose >= 1:
                    print("COMPUTING EXACTLY ON STEP COUNT", step_count)

                exact_accesses = (compute_exactly[:, 0], compute_exactly[:, 1])
                estimates[exact_accesses] = swap_sample_for_targets(imgs, exact_accesses, medoids, N, args.fast_pam1, metric = metric)
                exact_mask[exact_accesses] = 1
                T_samples[exact_accesses] += N

                median_return = np.median(estimates[exact_accesses])
                cand_condition = np.where(estimates <= median_return)
                new_candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))
                #candidates = np.array([elem for elem in set(tuple(elem) for elem in new_candidates) & set(tuple(elem) for elem in candidates)])
                #NOTE: Is this safe?
                candidates = np.array([])

            if len(candidates) == 0: break

            accesses = (candidates[:, 0], candidates[:, 1])
            new_samples = swap_sample_for_targets(imgs, accesses, medoids, this_batch_size, args.fast_pam1, metric = metric)

            estimates[accesses] = \
                ((T_samples[accesses] * estimates[accesses]) + (this_batch_size * new_samples)) / (this_batch_size + T_samples[accesses])
            T_samples[accesses] += this_batch_size

            median_return = np.median(estimates[accesses])
            if median_return == np.max(estimates[accesses]):
                    cand_condition = np.where(estimates <= median_return) # Strictly less to chop off heavy ail
            else:
                cand_condition = np.where(estimates <= median_return)

            new_candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))
            # NOTE: Make this more elegant
            candidates = np.array([elem for elem in set(tuple(elem) for elem in new_candidates) & set(tuple(elem) for elem in candidates)])
            step_count += 1

        # Choose the minimum amongst all losses and perform the swap
        # NOTE: possible to get first elem of zip object without converting to list?
        best_swaps = zip( np.where(estimates == estimates.min())[0], np.where(estimates == estimates.min())[1] )
        best_swaps = list(best_swaps)
        best_swap = best_swaps[0]

        print("Computed exactly for:", exact_mask.sum())
        performed_or_not, medoids, loss = medoid_swap(medoids, best_swap, imgs, loss, args)

        S_logstring = update_logstring(S_logstring, iter - 1, loss, exact_mask.sum(), 0.0, T)
        if performed_or_not == "NO SWAP PERFORMED":
            break

    return medoids, S_logstring

def CSH_build_and_swap(args):
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]

    built_medoids = []
    B_logstring = {}
    if 'B' in args.build_ao_swap:
        built_medoids, B_logstring = CSH_build(args, imgs, sigma)
        print("Built medoids", built_medoids)

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

        swapped_medoids, S_logstring = CSH_swap(args, imgs, sigma, init_medoids)
        print("Final medoids", swapped_medoids)

    return built_medoids, swapped_medoids, B_logstring, S_logstring

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    CSH_build_and_swap(args)
