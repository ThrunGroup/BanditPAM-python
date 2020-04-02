import itertools

ns = [100, 500, 2000, 5000, 10000, 30000, 70000]
ks = [1, 3, 10, 50, 100]

for idx,elem in enumerate(itertools.product(ns, ks)):
  print(str(idx) + " : ['ucb', 0, " + str(elem[1]) + ", " + str(elem[0]) + ", 42, 'MNIST'],")
