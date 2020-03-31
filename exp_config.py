experiments = {
# Exp number : MUST respect same order of argparse
# script, verbosity, num_medoids, sample_size, seed, dataset

'0' : ['naive', 0, 3, 700, 42, 'MNIST'],
# 595 is true medoid at 6387.411136116143
# 285 is close second at 6392.1460710 -- not sure why not normalizing gives a problem
# medoids = [595, 306, 392, 319, 23, 558, 251, 118, 448, 529]


'1' : ['ucb', 0, 3, 700, 42, 'MNIST'],
# '2' : ['naive_pam.py', 0, 3, 2000, 42, 'MNIST']
# '3' : ['ucb_pam.py', 0, 3, 2000, 42, 'MNIST']
}
