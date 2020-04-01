from data_utils import *


def UCB_build(args, imgs, sigma):
    ### Parameters
    N = len(imgs)
    p = 1e-6
    num_samples = np.zeros(N)
    estimates = np.zeros(N)
    medoids = []
    best_distances = [float('inf') for _ in range(N)]
    # NOTE: What should this batch_size be? 20? Also note that this will result in (very minor) inefficiencies when batch_size > 1

    def sample_for_targets(imgs, targets, batch_size):
        # NOTE: Fix this with array broadcasting
        N = len(imgs)
        estimates = np.zeros(len(targets))
        # NOTE: Should this sampling be done with replacement? And do I need shuffling?
        # NOTE: Also, should the point be able to sample itself?
        tmp_refs = np.array(np.random.choice(N, size = batch_size, replace = False), dtype='int')
        for tar_idx, target in enumerate(targets):
            distances = np.zeros(batch_size)
            for tmp_idx, tmp in enumerate(tmp_refs):
                ## tmp is the actually index of the reference point, tmp_idx just numerates them)
                distances[tmp_idx] = cost_fn(imgs, target, tmp, best_distances) # NOTE: depends on other medoids too!
            estimates[tar_idx] = np.mean(distances)
        return estimates

    # Iteratively:
    # Pretend each previous arm is fixed.
    # For new arm candidate, true parameter is the TRUE loss when using the point as medoid
        # As a substitute, can measure the "gain" of using this point -- negative DECREASE in distance (the lower the distance, the better)
    # We sample this using UCB algorithm to get confidence bounds on what that loss will be
    # Update ucb, lcb, and empirical estimate by sampling WITH REPLACEMENT(NOTE)
        # If more than n points, just compute exactly -- otherwise, there's a failure mode where
        # Two points very close together require shittons of samples to distinguish their mean distance

    for k in range(args.num_medoids):
        print("Finding medoid", k)
        ## Initialization
        step_count = 0
        candidates = range(N) # Initially, consider all points
        lcbs = -100 * np.ones(N)
        ucbs = -100 * np.ones(N)
        T_samples = np.zeros(N)
        exact_mask = np.zeros(N)

        original_batch_size = 100
        base = 1 # Right now, use constant batch size

        # Pull arms, update ucbs and lcbs
        while(len(candidates) > 1): # NOTE: Should also probably restrict absolute distance in cb_delta?
            if args.verbose >= 1: print("Step count:", step_count, ", Candidates:", len(candidates), candidates)
            step_count += 1

            # NOTE: tricky computations below
            this_batch_size = original_batch_size * (base**step_count)

            # Don't update all estimates, just pulled arms
            estimates[candidates] = \
                ((T_samples[candidates] * estimates[candidates]) + (this_batch_size * sample_for_targets(imgs, candidates, this_batch_size))) / (this_batch_size + T_samples[candidates])
            T_samples[candidates] += this_batch_size

            # NOTE: Can further optimize this by putting this above the sampling paragraph just above this.
            compute_exactly = np.where((T_samples >= N) & (exact_mask == 0))[0]
            if len(compute_exactly > 0):
                # import ipdb; ipdb.set_trace()
                print("COMPUTING EXACTLY ON STEP COUNT", step_count)
                estimates[compute_exactly] = sample_for_targets(imgs, compute_exactly, N)
                lcbs[compute_exactly] = estimates[compute_exactly]
                ucbs[compute_exactly] = estimates[compute_exactly]
                exact_mask[compute_exactly] = 1
                T_samples[compute_exactly] += N
                candidates = np.setdiff1d(candidates, compute_exactly) # Remove compute_exactly points from candidates so they're bounds don't get updated below

            cb_delta = sigma * np.sqrt(np.log(1 / p) / T_samples[candidates])
            lcbs[candidates] = estimates[candidates] - cb_delta
            ucbs[candidates] = estimates[candidates] + cb_delta

            candidates = np.where( (lcbs < ucbs.min()) & (exact_mask == 0) )[0]

        print(np.where( lcbs == lcbs.min() ))
        new_medoid = np.arange(N)[ np.where( lcbs == lcbs.min() ) ]
        print("New Medoid:", new_medoid)
        medoids.append(new_medoid) #BUG: Choose the lowest lcb (candidates will no longer contain exactly computed points). What about duplicates?
        best_distances = get_best_distances(medoids, imgs)
    print(medoids)
    return medoids

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
    medoids = UCB_build(args, imgs)
    print(medoids)
