experiments = [

    # Need to verify:
    # BUILD:
    # 1. UCB scales as nlogn for k ~ N/500
    # 2. Naive scales as n^2 for k ~ N / 500
    # 3. UCB scales linearly with k
    # 4. Naive scales linearly with k


    # Exp number : MUST respect same order of argparse
    # script, verbosity, num_medoids, sample_size, seed, dataset

    # 595 is true medoid at 6387.411136116143
    # 285 is close second at 6392.1460710 -- not sure why not normalizing gives a problem
    # medoids = [595, 306, 392, 319, 23, 558, 251, 118, 448, 529]
    # '0' : ['naive', 0, 3, 500, 42, 'MNIST'],
    # '1' : ['ucb', 0, 3, 500, 42, 'MNIST'],


    # n = 100, 300, 1000, 3000, 10000, 30000
    # k = 1, 2, 3, 4, 5
    # Below are all such pairs (n, k) for UCB and naive

    ####################################
    # BUILD ONLY:
    ####################################

    # Can't use k = 1 because need 2 medoids for second_best_distances

    # Scaling with N, k = 2:
    ['ucb', 'BS', 0, 2, 100, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 2, 100, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 2, 300, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 2, 300, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 2, 1000, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 2, 1000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 2, 3000, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 2, 3000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 2, 10000, 42, 'MNIST', ''],
    # ['naive_v1', 'BS', 0, 2, 10000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 2, 30000, 42, 'MNIST', ''],
    # ['naive_v1', 'BS', 0, 2, 30000, 42, 'MNIST', ''],


    # Scaling with N, k = 3:
    ['ucb', 'BS', 0, 3, 100, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 3, 100, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 3, 300, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 3, 300, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 3, 1000, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 3, 1000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 3, 3000, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 3, 3000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 3, 10000, 42, 'MNIST', ''],
    # ['naive_v1', 'BS', 0, 3, 10000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 3, 30000, 42, 'MNIST', ''],
    # ['naive_v1', 'BS', 0, 3, 30000, 42, 'MNIST', ''],


    # Scaling with N, k = 4:
    ['ucb', 'BS', 0, 4, 100, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 4, 100, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 4, 300, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 4, 300, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 4, 1000, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 4, 1000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 4, 3000, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 4, 3000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 4, 10000, 42, 'MNIST', ''],
    # ['naive_v1', 'BS', 0, 4, 10000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 4, 30000, 42, 'MNIST', ''],
    # ['naive_v1', 'BS', 0, 4, 30000, 42, 'MNIST', ''],



    # Scaling with N, k = 5:
    ['ucb', 'BS', 0, 5, 100, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 5, 100, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 5, 300, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 5, 300, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 5, 1000, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 5, 1000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 5, 3000, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 5, 3000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 5, 10000, 42, 'MNIST', ''],
    # ['naive_v1', 'BS', 0, 5, 10000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 5, 30000, 42, 'MNIST', ''],
    # ['naive_v1', 'BS', 0, 5, 30000, 42, 'MNIST', ''],
]
