import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import *
from tests import Namespace

def showx():
    plt.draw()
    plt.pause(1) # <-------
    input("<Hit Enter To Close>")
    plt.close()


def get_fixed_sigma_dist(dataset, metric, target_size, subsamp_refs = False):
    '''
    Use this to estimate sigma, as in sigma-sub-Gaussian, for a couple of points
    '''
    arms = np.random.choice(len(dataset), size = target_size, replace = False)
    best_distances = np.inf * np.ones(len(dataset))
    for a_idx, a in enumerate(arms):
        if a_idx % 10 == 0: print(a_idx)

        if subsamp_refs:
            ref_idcs = np.random.choice(range(len(dataset)), size = 4000, replace = False)
        else:
            ref_idcs = range(len(dataset))

        costs = cost_fn(dataset, [a], ref_idcs, best_distances, metric = metric, use_diff = False, dist_mat = None)
        print(min(costs), np.mean(costs), max(costs))
        # WARNING: unsafe global references
        # if dataset_name == 'MNIST' and metric == 'L2':
        #     binsize = 0.125*2
        # elif dataset_name == 'SCRNAPCA' and metric == 'L2':
        #     binsize = 0.0125/4
        # elif dataset_name == 'SCRNA' and metric == 'L1':
        #     binsize = 5

        # bins = np.arange(min(means) - binsize, max(means) + binsize, binsize)

        metric_title = '$L_2$' if metric == 'L2' else '$L_1$'
        sns.distplot(costs,
            #bins = bins,
            kde_kws = {'label' : "Gaussian KDE"},
            )
        plt.title("Histogram of $\sigma$ for point " + str(a) + " in " + dataset_name + ", " + metric_title + " distance")
        plt.ylabel('Frequency')
        plt.xlabel('$L$')
        plt.legend(loc="upper right")
        showx()



def get_dist_of_means(dataset, metric, subsample_size, subsamp_refs = False):
    arms = np.random.choice(len(dataset), size = subsample_size, replace = False)
    means = np.zeros(len(arms))
    best_distances = np.inf * np.ones(len(dataset))
    for a_idx, a in enumerate(arms):
        if a_idx % 10 == 0: print(a_idx)

        if subsamp_refs:
            ref_idcs = np.random.choice(range(len(dataset)), size = 4000, replace = False)
        else:
            ref_idcs = range(len(dataset))

        costs = cost_fn(dataset, [a], ref_idcs, best_distances, metric = metric, use_diff = False, dist_mat = None)
        means[a_idx] = np.mean(costs)
    # import ipdb; ipdb.set_trace()
    print(max(means))

    # Warning: unsafe global references

    if dataset_name == 'MNIST' and metric == 'L2':
        binsize = 0.125*2
    elif dataset_name == 'SCRNAPCA' and metric == 'L2':
        binsize = 0.0125/4
    elif dataset_name == 'SCRNA' and metric == 'L1':
        binsize = 5

    bins = np.arange(min(means) - binsize, max(means) + binsize, binsize)

    print(dataset_name, metric)
    print(bins)
    metric_title = '$L_2$' if metric == 'L2' else '$L_1$'
    sns.distplot(means,
        bins = bins,
        kde_kws = {'label' : "Gaussian KDE"},
        )
    plt.title("Histogram of true arm parameters for " + dataset_name + ", " + metric_title + " distance")
    plt.ylabel('Frequency')
    plt.xlabel('$\mu$')
    plt.legend(loc="upper right")
    plt.show()

#min: 0.54499 25th: 1.01669 median: 1.18792 75th: 1.41559 max: 1.94342 mean: 1.21168
def extract_sigmas(str_):
    tokens = str_.split(' ')
    nums = [tokens[i] for i in range(1, 10, 2)]
    floats = list(map(float, nums))
    return floats

def make_MNIST_sigma_dist_example():
    # Taken from L-ucb-True-BS-v-0-k-5-N-70000-s-51-d-MNIST-m-L2-w-
    step1 = "min: 0.47623 25th: 1.03397 median: 1.21351 75th: 1.44496 max: 2.33667 mean: 1.27651"
    step2 = "min: 0.0 25th: 0.27304 median: 0.36676 75th: 0.48846 max: 1.28567 mean: 0.4005"
    step3 = "min: 0.0 25th: 0.21983 median: 0.30783 75th: 0.40999 max: 1.37472 mean: 0.32629"
    step4 = "min: 0.0 25th: 0.17077 median: 0.25251 75th: 0.3475 max: 1.2815 mean: 0.27272"
    step5 = "min: 0.0 25th: 0.11478 median: 0.19141 75th: 0.27925 max: 1.06646 mean: 0.20325"

    sigma_dist = np.zeros((5, 5))
    sigma_dist[0] = extract_sigmas(step1)
    sigma_dist[1] = extract_sigmas(step2)
    sigma_dist[2] = extract_sigmas(step3)
    sigma_dist[3] = extract_sigmas(step4)
    sigma_dist[4] = extract_sigmas(step5)
    print(sigma_dist)

    fig, ax = plt.subplots(figsize = (5, 5))
    plt.boxplot(sigma_dist.T, whis=100)
    plt.xlabel("BUILD medoid assignment ($k = 5$)")
    plt.ylabel("$\sigma_i$")
    plt.title("Distribution of $\sigma_i$ in each BUILD assignment")
    plt.savefig('figures/MNIST_sigmas_example.pdf')
    showx()

if __name__ == '__main__':
    # dataset_name = 'SCRNA'
    # metric = 'L1'
    # subsamp_refs = True
    #
    # args = Namespace(dataset = dataset_name, metric = metric)
    # points, labels, sigma = load_data(args)
    #
    # # get_dist_of_means(dataset = points, metric = metric, subsample_size = 100, subsamp_refs = subsamp_refs)
    # get_fixed_sigma_dist(dataset = points, metric = metric, target_size = 4, subsamp_refs = False)
    make_MNIST_sigma_dist_example()
