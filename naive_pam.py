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
    medoids = []
    N = len(imgs)
    best_distances = [float('inf') for _ in range(N)]
    for k in range(args.num_medoids):
        if args.verbose >= 1:
            print("Finding medoid", k)

        # Greedily choose the point which minimizes the loss
        best_loss = float('inf')
        best_medoid = -1
        for target in range(N):
            if (target + 1) % 100 == 0 and args.verbose >= 1:
                print(target)
            # if target in medoids: continue # Skip existing medoids NOTE: removing this optimization for complexity comparison

            loss = 0
            # NOTE: SHould reference be allowed to be the target (sample itself)?
            for reference in range(N):
                # if reference in medoids: continue # Skip existing medoids NOTE: removing this optimization for complexity comparison
                d_r_t = d(imgs[target], imgs[reference])
                d_count += 1
                loss += d_r_t if d_r_t < best_distances[reference] else best_distances[reference]

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


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
    medoids = naive_build(args, imgs)
    print(medoids)
