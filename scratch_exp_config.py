experiments = [
    # Exp number : MUST respect same order of argparse
    # script, B/S, verbosity, num_medoids, sample_size, seed, dataset, metric, warm start medoids

    # ['ucb', 'BS', 0, 2, 1000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 2, 3000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 2, 10000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 2, 30000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 2, 70000, 42, 'MNIST', 'L2', ''],

    ['csh', 'BS', 3, 2, 1000, 42, 'MNIST', 'L2', ''],
    ['csh', 'BS', 0, 2, 3000, 42, 'MNIST', 'L2', ''],
    ['csh', 'BS', 0, 2, 10000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 2, 30000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 2, 70000, 42, 'MNIST', 'L2', ''],
    #
    # ['naive_v1', 'BS', 0, 2, 1000, 42, 'MNIST', 'L2', ''],
    # ['naive_v1', 'BS', 0, 2, 3000, 42, 'MNIST', 'L2', ''],
    # ['naive_v1', 'BS', 0, 2, 10000, 42, 'MNIST', 'L2', ''],



    # ['ucb', 'BS', 0, 3, 1000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 3, 3000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 3, 10000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 3, 30000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 3, 70000, 42, 'MNIST', 'L2', ''],

    ['csh', 'BS', 0, 3, 1000, 42, 'MNIST', 'L2', ''],
    ['csh', 'BS', 0, 3, 3000, 42, 'MNIST', 'L2', ''],
    ['csh', 'BS', 0, 3, 10000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 3, 30000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 3, 70000, 42, 'MNIST', 'L2', ''],
    #
    # ['naive_v1', 'BS', 0, 3, 1000, 42, 'MNIST', 'L2', ''],
    # ['naive_v1', 'BS', 0, 3, 3000, 42, 'MNIST', 'L2', ''],
    # ['naive_v1', 'BS', 0, 3, 10000, 42, 'MNIST', 'L2', ''],




    # ['ucb', 'BS', 0, 4, 1000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 4, 3000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 4, 10000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 4, 30000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 4, 70000, 42, 'MNIST', 'L2', ''],

    # ['csh', 'BS', 0, 4, 1000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 4, 3000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 4, 10000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 4, 30000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 4, 70000, 42, 'MNIST', 'L2', ''],
    #
    # ['naive_v1', 'BS', 0, 4, 1000, 42, 'MNIST', 'L2', ''],
    # ['naive_v1', 'BS', 0, 4, 3000, 42, 'MNIST', 'L2', ''],
    # ['naive_v1', 'BS', 0, 4, 10000, 42, 'MNIST', 'L2', ''],



    # ['ucb', 'BS', 0, 5, 1000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 5, 3000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 5, 10000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 5, 30000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 5, 70000, 42, 'MNIST', 'L2', ''],

    # ['csh', 'BS', 0, 5, 1000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 5, 3000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 5, 10000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 5, 30000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 5, 70000, 42, 'MNIST', 'L2', ''],
    #
    # ['naive_v1', 'BS', 0, 5, 1000, 42, 'MNIST', 'L2', ''],
    # ['naive_v1', 'BS', 0, 5, 3000, 42, 'MNIST', 'L2', ''],
    # ['naive_v1', 'BS', 0, 5, 10000, 42, 'MNIST', 'L2', ''],


    # ['ucb', 'BS', 0, 10, 1000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 10, 3000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 10, 10000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 10, 30000, 42, 'MNIST', 'L2', ''],
    # ['ucb', 'BS', 0, 10, 70000, 42, 'MNIST', 'L2', ''],

    # ['csh', 'BS', 0, 10, 1000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 10, 3000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 10, 10000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 10, 30000, 42, 'MNIST', 'L2', ''],
    # ['csh', 'BS', 0, 10, 70000, 42, 'MNIST', 'L2', ''],
    #
    # ['naive_v1', 'BS', 0, 10, 1000, 42, 'MNIST', 'L2', ''],
    # ['naive_v1', 'BS', 0, 10, 3000, 42, 'MNIST', 'L2', ''],
    # ['naive_v1', 'BS', 0, 10, 10000, 42, 'MNIST', 'L2', ''],
]
