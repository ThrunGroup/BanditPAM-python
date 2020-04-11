experiments = {
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
    # 0 : ['naive', 0, 1, 100, 42, 'MNIST', []],
    # 1 : ['ucb', 0, 1, 100, 42, 'MNIST', []],
    # 2 : ['naive', 0, 3, 100, 42, 'MNIST', []],
    # 3 : ['ucb', 0, 3, 100, 42, 'MNIST', []],
    # 4 : ['naive', 0, 10, 100, 42, 'MNIST', []],
    # 5 : ['ucb', 0, 10, 100, 42, 'MNIST', []],
    # 6 : ['naive', 0, 50, 100, 42, 'MNIST', []],
    # 7 : ['ucb', 0, 50, 100, 42, 'MNIST', []],
    # 8 : ['naive', 0, 1, 500, 42, 'MNIST', []],
    # 9 : ['ucb', 0, 1, 500, 42, 'MNIST', []],
    # 10 : ['naive', 0, 3, 500, 42, 'MNIST', []],
    # 11 : ['ucb', 0, 3, 500, 42, 'MNIST', []],
    # 12 : ['naive', 0, 10, 500, 42, 'MNIST', []],
    # 13 : ['ucb', 0, 10, 500, 42, 'MNIST', []],
    # 14 : ['naive', 0, 50, 500, 42, 'MNIST', []],
    # 15 : ['ucb', 0, 50, 500, 42, 'MNIST', []],
    # 16 : ['naive', 0, 1, 2000, 42, 'MNIST', []],
    # 17 : ['ucb', 0, 1, 2000, 42, 'MNIST', []],
    # 18 : ['naive', 0, 3, 2000, 42, 'MNIST', []],
    # 19 : ['ucb', 0, 3, 2000, 42, 'MNIST', []],
    # 20 : ['naive', 0, 10, 2000, 42, 'MNIST', []],
    # 21 : ['ucb', 0, 10, 2000, 42, 'MNIST', []],
    # 22 : ['naive', 0, 50, 2000, 42, 'MNIST', []],
    # 23 : ['ucb', 0, 50, 2000, 42, 'MNIST', []],
    # 24 : ['naive', 0, 1, 5000, 42, 'MNIST', []],
    # 25 : ['ucb', 0, 1, 5000, 42, 'MNIST', []],
    # 26 : ['naive', 0, 3, 5000, 42, 'MNIST', []],
    # 27 : ['ucb', 0, 3, 5000, 42, 'MNIST', []],
    # 28 : ['naive', 0, 10, 5000, 42, 'MNIST', []],
    # 29 : ['ucb', 0, 10, 5000, 42, 'MNIST', []],
    # 30 : ['naive', 0, 50, 5000, 42, 'MNIST', []],
    # 31 : ['ucb', 0, 50, 5000, 42, 'MNIST', []],
    # 32 : ['naive', 0, 1, 10000, 42, 'MNIST', []],
    # 33 : ['ucb', 0, 1, 10000, 42, 'MNIST', []],
    # 34 : ['naive', 0, 3, 10000, 42, 'MNIST', []],
    # 35 : ['ucb', 0, 3, 10000, 42, 'MNIST', []],
    # 36 : ['naive', 0, 10, 10000, 42, 'MNIST', []],
    # 37 : ['ucb', 0, 10, 10000, 42, 'MNIST', []],
    # 38 : ['naive', 0, 50, 10000, 42, 'MNIST', []],
    # 39 : ['ucb', 0, 50, 10000, 42, 'MNIST', []],
    # 40 : ['naive', 0, 1, 30000, 42, 'MNIST', []],
    # 41 : ['ucb', 0, 1, 30000, 42, 'MNIST', []],
    # 42 : ['naive', 0, 3, 30000, 42, 'MNIST', []],
    # 43 : ['ucb', 0, 3, 30000, 42, 'MNIST', []],
    # 44 : ['naive', 0, 10, 30000, 42, 'MNIST', []],
    # 45 : ['ucb', 0, 10, 30000, 42, 'MNIST', []],
    # 46 : ['naive', 0, 50, 30000, 42, 'MNIST', []],
    # 47 : ['ucb', 0, 50, 30000, 42, 'MNIST', []],
    # 48 : ['naive', 0, 1, 70000, 42, 'MNIST', []],
    # 49 : ['ucb', 0, 1, 70000, 42, 'MNIST', []],
    # 50 : ['naive', 0, 3, 70000, 42, 'MNIST', []],
    # 51 : ['ucb', 0, 3, 70000, 42, 'MNIST', []],
    # 52 : ['naive', 0, 10, 70000, 42, 'MNIST', []],
    # 53 : ['ucb', 0, 10, 70000, 42, 'MNIST', []],
    # 54 : ['naive', 0, 50, 70000, 42, 'MNIST', []],
    # 55 : ['ucb', 0, 50, 70000, 42, 'MNIST', []],


    54 : ['ucb', 0, 50, 100, 42, 'MNIST', ''],
    55 : ['naive', 0, 50, 100, 42, 'MNIST', ''],
}
