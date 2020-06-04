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


if __name__ == '__main__':
    dataset_name = 'SCRNA'
    metric = 'L1'
    subsamp_refs = True

    args = Namespace(dataset = dataset_name, metric = metric)
    points, labels, sigma = load_data(args)

    # get_dist_of_means(dataset = points, metric = metric, subsample_size = 100, subsamp_refs = subsamp_refs)
    get_fixed_sigma_dist(dataset = points, metric = metric, target_size = 4, subsamp_refs = False)
