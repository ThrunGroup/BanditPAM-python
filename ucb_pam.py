'''
Contains the UCB-based implementation of PAM (dubbed BanditPAM).

This is the core algorithm.
'''

from data_utils import *
import itertools

def build_sample_for_targets(imgs, targets, batch_size, best_distances, metric = None, return_sigma = False, dist_mat = None):
    '''
    For the given targets, which are candidate points to be assigned as medoids
    during a build step, we compute the changes in loss they would induce
    on a subsample of batch_size reference points (tmp_refs).

    The returned value is an array of estimated changes in loss for each target.
    '''

    # TODO: Improve this with array broadcasting
    N = len(imgs)
    estimates = np.zeros(len(targets))
    sigmas = np.zeros(len(targets))
    tmp_refs = np.array(np.random.choice(N, size = batch_size, replace = False), dtype = 'int')
    for tar_idx, target in enumerate(targets):
        if best_distances[0] == np.inf:
            # No medoids have been assigned, can't use the difference in loss
            costs = cost_fn(imgs, target, tmp_refs, best_distances, metric = metric, use_diff = False, dist_mat = dist_mat)
        else:
            costs = cost_fn(imgs, target, tmp_refs, best_distances, metric = metric, use_diff = True, dist_mat = dist_mat)

        estimates[tar_idx] = np.mean(costs)
        if return_sigma:
            sigmas[tar_idx] = np.std(costs) / SIGMA_DIVISOR

    if return_sigma:
        return estimates.round(DECIMAL_DIGITS), sigmas, tmp_refs

    return estimates.round(DECIMAL_DIGITS), None, tmp_refs

def UCB_build(args, imgs, sigma, dist_mat = None):
    '''
    Performs the BUILD step of BanditPAM. Analogous to the BUILD step of PAM,
    BanditPAM assigns the initial medoids one-by-one by choosing the point at
    each step that would lower the total loss the most. Instead of computing the
    change in loss for every other point, it estimates these changes in loss.
    '''

    B_logstring = init_logstring()

    ### Parameters
    metric = args.metric
    N = len(imgs)
    p = 1. / (N * 1000)
    num_samples = np.zeros(N)
    estimates = np.zeros(N)

    cache_computed = np.zeros((N, N))

    if len(args.warm_start_medoids) > 0:
        warm_start_medoids = list(map(int, args.warm_start_medoids.split(',')))
        medoids = warm_start_medoids.copy()
        num_medoids_found = len(medoids)
        best_distances, closest_medoids = np.array(get_best_distances(medoids, imgs, metric = metric, dist_mat = dist_mat))
    else:
        medoids = []
        num_medoids_found = 0
        best_distances = np.inf * np.ones(N)

    for k in range(num_medoids_found, args.num_medoids):
        compute_sigma = True

        if args.verbose >= 1:
            print("Finding medoid", k)

        ## Initialization
        step_count = 0
        candidates = range(N) # Initially, consider all points
        lcbs = 1000 * np.ones(N)
        ucbs = 1000 * np.ones(N)
        T_samples = np.zeros(N)
        exact_mask = np.zeros(N)
        sigmas = np.zeros(N)

        original_batch_size = 100
        base = 1 # Right now, use constant batch size

        while(len(candidates) > 0):
            if args.verbose >= 1:
                print("Step count:", step_count, ", Candidates:", len(candidates), candidates)

            this_batch_size = int(original_batch_size * (base**step_count))

            # Find the points whose change in loss should be computed exactly,
            # because >= N reference points have already been sampled.
            compute_exactly = np.where((T_samples + this_batch_size >= N) & (exact_mask == 0))[0]
            if len(compute_exactly) > 0:
                if args.verbose >= 1:
                    print("COMPUTING EXACTLY ON STEP COUNT", step_count)

                estimates[compute_exactly], _, calc_refs = build_sample_for_targets(imgs, compute_exactly, N, best_distances, metric = metric, return_sigma = False, dist_mat = dist_mat)
                lcbs[compute_exactly] = estimates[compute_exactly]
                ucbs[compute_exactly] = estimates[compute_exactly]
                exact_mask[compute_exactly] = 1
                T_samples[compute_exactly] += N
                candidates = np.setdiff1d(candidates, compute_exactly) # Remove compute_exactly points from candidates so they're bounds don't get updated below

                for c_e in compute_exactly:
                    for c_r in calc_refs:
                        cache_computed[c_e, c_r] = 1

            if len(candidates) == 0: break # The last remaining candidates were computed exactly

            # Gather more evaluations of the change in loss for some reference points
            if compute_sigma:
                # Estimate sigma from the data, if necessary
                sample_costs, sigmas, calc_refs = build_sample_for_targets(imgs, candidates, this_batch_size, best_distances, metric = metric, return_sigma = True, dist_mat = dist_mat)
                compute_sigma = False
            else:
                sample_costs, _, calc_refs = build_sample_for_targets(imgs, candidates, this_batch_size, best_distances, metric = metric, return_sigma = False, dist_mat = dist_mat)

            for c_e in candidates:
                for c_r in calc_refs:
                    cache_computed[c_e, c_r] = 1
            print("Unique distances computed:", np.sum(cache_computed))

            # Update running average of estimates and confidence bounce
            estimates[candidates] = \
                ((T_samples[candidates] * estimates[candidates]) + (this_batch_size * sample_costs)) / (this_batch_size + T_samples[candidates])
            T_samples[candidates] += this_batch_size
            cb_delta = sigmas[candidates] * np.sqrt(np.log(1 / p) / T_samples[candidates])
            lcbs[candidates] = estimates[candidates] - cb_delta
            ucbs[candidates] = estimates[candidates] + cb_delta

            candidates = np.where( (lcbs < ucbs.min()) & (exact_mask == 0) )[0]
            step_count += 1

        # Breaks exact ties with first. Also converts array to int.
        # This does indeed happen, for example in ucb k = 50, n = 100, s = 42, d = MNIST
        new_medoid = np.arange(N)[ np.where( lcbs == lcbs.min() ) ]
        new_medoid = new_medoid[0]

        if args.verbose >= 1:
            print("New Medoid:", new_medoid)

        medoids.append(new_medoid)
        best_distances, closest_medoids = get_best_distances(medoids, imgs, metric = metric, dist_mat = dist_mat)
        print("Computed exactly for:", exact_mask.sum())

        # get information about sigmas: min, 25, median, 75, max, mean
        sigma_arr = [np.min(sigmas), np.quantile(sigmas, 0.25), np.median(sigmas), np.quantile(sigmas, 0.75), np.max(sigmas), np.mean(sigmas)]
        B_logstring = update_logstring(B_logstring, k, best_distances, exact_mask.sum(), p, sigma_arr)

    return medoids, B_logstring, cache_computed


def swap_sample_for_targets(imgs, targets, current_medoids, batch_size, FastPAM1 = False, metric = None, return_sigma = False, dist_mat = None):
    '''
    For the given targets (potential swaps) during a swap step, we compute the
    changes in loss they would induce on a subsample of batch_size reference
    points (tmp_refs) when the swap is performed.

    The returned value is an array of estimated changes in loss for each target
    (swap).
    '''
    # NOTE: Improve this with array broadcasting
    # Also generalize and consolidate it with the fn of the same name in the build step
    orig_medoids = targets[0]
    new_medoids = targets[1]
    assert len(orig_medoids) == len(new_medoids), "Must pass equal number of original medoids and new medoids"
    # NOTE: Need to preserve order of swaps that are passed - otherwise estimates will be for the wrong swaps
    # I.e. there will be an error if estimates aren't indexed properly -- only ok if we do 1 target at a time

    # WARNING: Zip doesn't throw an error for unequal lengths, it just drops extraneous points
    swaps = list(zip(orig_medoids, new_medoids))

    N = len(imgs)
    k = len(current_medoids)

    tmp_refs = np.array(np.random.choice(N, size = batch_size, replace = False), dtype='int')
    if FastPAM1:
        if return_sigma:
            estimates, sigmas = cost_fn_difference_FP1(imgs, swaps, tmp_refs, current_medoids, metric = metric, return_sigma = True, dist_mat = dist_mat) # NOTE: depends on other medoids too!
            return estimates.round(DECIMAL_DIGITS), sigmas, tmp_refs
        else:
            estimates = cost_fn_difference_FP1(imgs, swaps, tmp_refs, current_medoids, metric = metric, return_sigma = False, dist_mat = dist_mat) # NOTE: depends on other medoids too!
    else:
        # NOTE: Return_sigma currently only supported with FP1 trick; need to add it to cost_fn_difference too
        raise Exception("Do not use this! Doesn't support dist_mat")
        estimates = cost_fn_difference(imgs, swaps, tmp_refs, current_medoids, metric = metric)

    return estimates.round(DECIMAL_DIGITS), None, tmp_refs


def UCB_swap(args, imgs, sigma, init_medoids, dist_mat = None, cache_computed = None):
    '''
    Performs the SWAP step of BanditPAM. Analogous to the SWAP step of PAM,
    BanditPAM chooses medoids to swap with non-medoids by performing the swap
    that would lower the total loss the most at each step. Instead of computing
    the exact change in loss for every other point, it estimates these changes.
    '''

    S_logstring = init_logstring()
    metric = args.metric
    k = len(init_medoids)
    N = len(imgs)
    p = 1. / (N * k * 1000)
    max_iter = 1e4

    medoids = init_medoids.copy()
    # NOTE: best_distances is NOT updated in future rounds - the analogy from build is broken. Maybe rename the variable
    best_distances, closest_medoids = get_best_distances(medoids, imgs, metric = metric, dist_mat = dist_mat)
    loss = np.mean(best_distances)
    iter = 0
    swap_performed = True
    while swap_performed and iter < max_iter: # not converged
        compute_sigma = True
        iter += 1

        candidates = np.array(list(itertools.product(range(k), range(N)))) # A candidate is a PAIR
        lcbs = 1000 * np.ones((k, N)) # NOTE: Instantiating these as np.inf gives runtime errors and nans. Find a better way to do this instead of using 1000
        estimates = 1000 * np.ones((k, N))
        ucbs = 1000 * np.ones((k, N))

        T_samples = np.zeros((k, N))
        exact_mask = np.zeros((k, N))

        original_batch_size = 100
        base = 1 # Right now, use constant batch size

        step_count = 0
        while(len(candidates) > 0):
            if args.verbose >= 1:
                print("SWAP Step count:", step_count)

            this_batch_size = int(original_batch_size * (base**step_count))

            # Find swaps whose returns should be computed exactly, because >= N
            # reference points have already been sampled
            comp_exactly_condition = np.where((T_samples + this_batch_size >= N) & (exact_mask == 0))
            compute_exactly = np.array(list(zip(comp_exactly_condition[0], comp_exactly_condition[1])))
            if len(compute_exactly) > 0:
                if args.verbose >= 1:
                    print("COMPUTING EXACTLY ON STEP COUNT", step_count)

                exact_accesses = (compute_exactly[:, 0], compute_exactly[:, 1])
                estimates[exact_accesses], _, calc_refs = swap_sample_for_targets(imgs, exact_accesses, medoids, N, args.fast_pam1, metric = metric, return_sigma = False, dist_mat = dist_mat)
                lcbs[exact_accesses] = estimates[exact_accesses]
                ucbs[exact_accesses] = estimates[exact_accesses]
                exact_mask[exact_accesses] = 1
                T_samples[exact_accesses] += N

                cand_condition = np.where( (lcbs < ucbs.min()) & (exact_mask == 0) )
                candidates = np.array(list(zip(cand_condition[0], cand_condition[1])))

                for e_a in exact_accesses:
                    for c_r in calc_refs:
                        cache_computed[e_a[0], c_r] = 1
                        cache_computed[e_a[1], c_r] = 1

            if len(candidates) == 0: break # The last candidates were computed exactly

            # Gather more evaluations of the change in loss for some reference points
            accesses = (candidates[:, 0], candidates[:, 1])
            if compute_sigma:
                # Estimate sigma from the data, if necessary
                new_samples, sigmas, calc_refs = swap_sample_for_targets(imgs, accesses, medoids, this_batch_size, args.fast_pam1, metric = metric, return_sigma = True, dist_mat = dist_mat)
                sigmas = sigmas.reshape(k, N) # So that can access it with sigmas[accesses] below
                compute_sigma = False
            else:
                new_samples, _, calc_refs = swap_sample_for_targets(imgs, accesses, medoids, this_batch_size, args.fast_pam1, metric = metric, return_sigma = False, dist_mat = dist_mat)

            for acc_ in accesses:
                for c_r in calc_refs:
                    cache_computed[acc_[0], c_r] = 1
                    cache_computed[acc_[1], c_r] = 1
            print("Unique distances computed:", np.sum(cache_computed))

            # Update running average of estimates and confidence bounce
            estimates[accesses] = \
                ((T_samples[accesses] * estimates[accesses]) + (this_batch_size * new_samples)) / (this_batch_size + T_samples[accesses])
            T_samples[accesses] += this_batch_size
            # NOTE: Sigmas is contains a value for EVERY arm, even non-candidates, so need [accesses]
            cb_delta = sigmas[accesses] * np.sqrt(np.log(1 / p) / T_samples[accesses])
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

        old_medoid_x = medoids[best_swap[0]]
        new_medoid_x = best_swap[1]

        print("Computed exactly for:", exact_mask.sum())
        performed_or_not, medoids, loss = medoid_swap(medoids, best_swap, imgs, loss, args, dist_mat = dist_mat)

        if original_batch_size >= len(imgs):
            # Corner case where sigmas aren't computed for too-small datasets
            # NOTE: This is different from the build step because in the build step, the sigmas are all initialized to 0's. Should be consistent between the two.
            sigma_arr = [0, 0, 0, 0, 0, 0]
        else:
            sigma_arr = [np.min(sigmas), np.quantile(sigmas, 0.25), np.median(sigmas), np.quantile(sigmas, 0.75), np.max(sigmas), np.mean(sigmas)]
        S_logstring = update_logstring(S_logstring, iter - 1, loss, exact_mask.sum(), p, sigma_arr, swap = (old_medoid_x, new_medoid_x))
        # TODO: Need to update swap_performed variable above, right now only breaking
        if performed_or_not == "NO SWAP PERFORMED":
            break

    return medoids, S_logstring, iter, loss

def UCB_build_and_swap(args):
    '''
    Run the entire BanditPAM algorithm, both the BUILD step and the SWAP step
    '''

    num_swaps = -1
    final_loss = -1
    dist_mat = None

    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    if args.metric == 'PRECOMP':
        dist_mat = np.loadtxt('tree-3630.dist')
        random_indices = np.random.choice(len(total_images), size = args.sample_size, replace = False)
        imgs = np.array([total_images[x] for x in random_indices])
        dist_mat = dist_mat[random_indices][:, random_indices]
    elif args.metric == 'TREE':
        imgs = np.random.choice(total_images, size = args.sample_size, replace = False)
    else:
        # Can remove range() here?
        imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]

    built_medoids = []
    B_logstring = {}
    if 'B' in args.build_ao_swap:
        built_medoids, B_logstring, cache_computed = UCB_build(args, imgs, sigma, dist_mat = dist_mat)
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

        swapped_medoids, S_logstring, num_swaps, final_loss = UCB_swap(args, imgs, sigma, init_medoids, dist_mat = dist_mat, cache_computed = cache_computed)
        print("Final medoids", swapped_medoids)

    return built_medoids, swapped_medoids, B_logstring, S_logstring, num_swaps, final_loss

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    UCB_build_and_swap(args)
