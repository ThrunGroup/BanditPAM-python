experiments = [
    # Exp number : MUST respect same order of argparse
    # script, B/S, verbosity, num_medoids, sample_size, seed, dataset, metric, warm start medoids

    # ['ucb', 'BS', 0, 5, 3000, 42, 'SCRNA', 'L1', ''],
    # ['clarans', 'BS', 0, 5, 10000, 42, 'MNIST', 'L2', ''],

    ['ucb', 'BS', 0, 2, 1000, 42, 'MNIST', 'L2', ''],
    ['ucb', 'BS', 0, 5, 1000, 42, 'MNIST', 'L2', ''],
    ['ucb', 'BS', 0, 10, 1000, 42, 'MNIST', 'L2', ''],
    ['ucb', 'BS', 0, 30, 1000, 42, 'MNIST', 'L2', ''],
    ['ucb', 'BS', 0, 50, 1000, 42, 'MNIST', 'L2', ''],
    ['ucb', 'BS', 0, 100, 1000, 42, 'MNIST', 'L2', ''],
    ['ucb', 'BS', 0, 200, 1000, 42, 'MNIST', 'L2', ''],
]
