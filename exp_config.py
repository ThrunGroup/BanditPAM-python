experiments = {
    # Exp number : MUST respect same order of argparse
    # script, verbosity, num_medoids, sample_size, seed, dataset

    # 595 is true medoid at 6387.411136116143
    # 285 is close second at 6392.1460710 -- not sure why not normalizing gives a problem
    # medoids = [595, 306, 392, 319, 23, 558, 251, 118, 448, 529]
    # '0' : ['naive', 0, 3, 700, 42, 'MNIST'],
    # '1' : ['ucb', 0, 3, 700, 42, 'MNIST'],

    # '2' : ['naive', 0, 3, 2000, 42, 'MNIST'],
    # '3' : ['ucb', 0, 3, 2000, 42, 'MNIST'],
    #
    # '4' : ['naive', 0, 3, 5000, 42, 'MNIST'],
    # '5' : ['ucb', 0, 3, 5000, 42, 'MNIST'],

    # '6' : ['naive', 0, 3, 10000, 42, 'MNIST'],
    '7' : ['ucb', 5, 3, 10000, 42, 'MNIST'],
}
