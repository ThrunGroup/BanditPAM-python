experiments = [
    # Exp number : MUST respect same order of argparse
    # script, verbosity, num_medoids, sample_size, seed, dataset

    # 595 is true medoid at 6387.411136116143
    # 285 is close second at 6392.1460710 -- not sure why not normalizing gives a problem
    # medoids = [595, 306, 392, 319, 23, 558, 251, 118, 448, 529]
    # '0' : ['naive', 0, 3, 500, 42, 'MNIST'],
    # '1' : ['ucb', 0, 3, 500, 42, 'MNIST'],


    # n = 100, 300, 1000, 3000, 10000
    # k = 1, 3, 5, 10, 20

    ####################################
    # BUILD ONLY:
    ####################################

    # Scaling with N:
    ['ucb', 'BS', 0, 3, 1000, 42, 'MNIST', 'L2', ''],
    ['ucb', 'BS', 0, 3, 3000, 42, 'MNIST', 'L2', ''],
    ['ucb', 'BS', 0, 3, 10000, 42, 'MNIST', 'L2', ''],
    ['ucb', 'BS', 0, 3, 30000, 42, 'MNIST', 'L2', ''],
    # ['naive_v1', 'BS', 0, 5, 100, 42, 'MNIST', 'L2', ''],
]
