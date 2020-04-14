experiments = [
    # Exp number : MUST respect same order of argparse
    # script, verbosity, num_medoids, sample_size, seed, dataset

    # 595 is true medoid at 6387.411136116143
    # 285 is close second at 6392.1460710 -- not sure why not normalizing gives a problem
    # medoids = [595, 306, 392, 319, 23, 558, 251, 118, 448, 529]
    # '0' : ['naive', 0, 3, 500, 42, 'MNIST'],
    # '1' : ['ucb', 0, 3, 500, 42, 'MNIST'],


    # GRID Search:
    # n = 20, 100, 500, 2000, 5000, 10000, 30000, 70000
    # k = 1, 3, 10, 50
    ['ucb', 'B', 0, 5, 100, 42, 'MNIST', ''],
    ['ucb', 'B', 0, 5, 300, 42, 'MNIST', ''],
    ['ucb', 'B', 0, 5, 1000, 42, 'MNIST', ''],
    ['ucb', 'B', 0, 5, 3000, 42, 'MNIST', ''],
    ['ucb', 'B', 0, 5, 10000, 42, 'MNIST', ''],

    ['ucb', 'BS', 0, 5, 100, 42, 'MNIST', ''],
    ['ucb', 'BS', 0, 5, 300, 42, 'MNIST', ''],
    ['ucb', 'BS', 0, 5, 1000, 42, 'MNIST', ''],
    ['ucb', 'BS', 0, 5, 3000, 42, 'MNIST', ''],
    ['ucb', 'BS', 0, 5, 10000, 42, 'MNIST', ''],

    ['naive_v1', 'B', 0, 5, 100, 42, 'MNIST', ''],
    ['naive_v1', 'B', 0, 5, 300, 42, 'MNIST', ''],
    ['naive_v1', 'B', 0, 5, 1000, 42, 'MNIST', ''],
    ['naive_v1', 'B', 0, 5, 3000, 42, 'MNIST', ''],
    ['naive_v1', 'B', 0, 5, 10000, 42, 'MNIST', ''],

    ['naive_v1', 'BS', 0, 5, 100, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 5, 300, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 5, 1000, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 5, 3000, 42, 'MNIST', ''],
    ['naive_v1', 'BS', 0, 5, 10000, 42, 'MNIST', ''],
]
