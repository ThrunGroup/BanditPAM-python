from data_utils import *

# args = get_args(sys.argv[1:])
# np.random.seed(args.seed)
# total_images, total_labels = load_data(args)
# sample_size = args.sample_size
# imgs = total_images[np.random.choice(range(len(total_images)), size = sample_size, replace = False)]
# estimate_sigma(imgs)

# naive_times = np.array([14.201, 122.213, 122.213])
# ucb_times = np.array([22.277, 101.315, 101.315])

# naive_dcs = np.array([1470000, 12000000, 75000000, 300000000])
# ucb_dcs = np.array([1830100, 8592900, 31967900])
#
#
# x = np.arange(500, 10100, 100)
# quadratic =  3*(x**2)
# knlogn = 2.5e2*3*(x*np.log(x))
#
# plt.plot([700, 2000, 5000, 10000], naive_dcs, 'bo')
# plt.plot([700, 2000, 5000], ucb_dcs, 'ro')
# plt.plot(x, quadratic, 'b-')
# plt.plot(x, knlogn, 'r-')
# plt.show()

#
# plt.plot([700, 2000, 5000], naive_times)
# plt.plot([700, 2000, 5000], ucb_times)
# plt.show()


### For k = 1
# naive_B_times = np.array([10200, 40400, 90600, 160800, 251000])
# ucb_B_times = np.array([7210, 13045, 21095, 28155, 56225, 79390, 325300, 909715, 4470955])
#
# x = np.array([100, 200, 300, 400, 500, 1000, 2000, 10000, 70000])
# q_x = np.arange(100, 501)
#
# quadratic = (x**2) + 2*x
# quadratic_ucb = (x**2)
# qlogq_ucb = 5*x*np.log(x)
#
# # plt.plot(x[:-3], naive_B_times, 'bo')
# # plt.plot(x, quadratic, 'b-')
#
# plt.plot(x, ucb_B_times, 'go')
# plt.plot(x, qlogq_ucb, 'r-')
#
# plt.show()


#### for k = 10, build only
# x = [100, 300, 1000, 3000]
# xticks = np.arange(100, 3100, 100)
#
# naive_B_calls = np.array([107759, 922363, 10079346, 90236487])
# ucb_B_calls = np.array([167319, 1330223, 15952226, 95086987])
#
# plt.plot(x, np.log(naive_B_calls), 'bo')
# plt.plot(x, np.log(ucb_B_calls), 'ro')
# quadratic = (3*xticks)**2
# nlogn = (xticks*np.log(xticks))
#
# plt.plot(xticks, np.log(quadratic), 'b-')
# plt.plot(xticks, np.log(nlogn), 'r-')
# plt.title("Number of distance calls for k = 10 medoids")
# plt.xlabel("N")
# plt.ylabel("Calls to d")
# plt.show()



# BUILD ONLY, SCALING WITH N, k = 3, N = 300, 1000, 3000, 10000, 30000
x = [300, 1000, 3000, 10000, 30000]
xticks = np.arange(100, 3100, 100)

ucb_B_calls = np.array([230455, 2124463, 7012741, 34433219, 112562578])
naive_B_calls = np.array([273035, 3011023, 27032741])


plt.plot(x[:-2], naive_B_calls, 'bo')
plt.plot(x[:-2], ucb_B_calls[:-2], 'ro')
quadratic = (3e-1*xticks)**2
quadratic_naive = (1.7*xticks)**2
nlogn = (4e2*xticks*np.log(xticks))

plt.plot(xticks, quadratic, 'g-')
plt.plot(xticks, quadratic_naive, 'b-')
plt.plot(xticks, nlogn, 'r-')
plt.title("Number of distance calls for k = 3 medoids vs N, BUILD only")
plt.xlabel("N")
plt.ylabel("Calls to d")
plt.show()


# BUILD ONLY, SCALING WITH K: 1, 2, 3 and N = 3000
k = [1, 2, 3, 4, 5, 10, 20, 30]

naive_B_calls = np.array([9006000, 18017194, 27032741, 36051842, 45074455, 90236487])
ucb_B_calls = np.array([284820, 4345814, 7012741, 12470922, 17469475, 95086987, 275484806, 456494000])

linear_naive = 9e6*np.array(k)
linear_ucb = 1.9e7*(np.array(k) - 5)
plt.plot(k[:-2], naive_B_calls, 'bo')
plt.plot(k, ucb_B_calls, 'ro')
plt.plot(k, linear_naive)
plt.plot(k, linear_ucb)
plt.show()






# BUILD and SWAP, SCALING WITH N:
# k = 3, N = 300, 1000, 3000, 10000, 30000
# Need to check UCB is nlogn when counts are (d_calls_total - d_calls_build ) / Swap count
x = np.array([300, 1000, 3000, 10000, 30000])
xticks = np.arange(300, 30100, 100)
xticks_short = np.arange(300, 4100, 100)

naive_BS_calls = np.array([549143, 12059082, 81163507])
naive_BS_FP1_calls = np.array([369143, 6059082, 45136507])
naive_Bonly_calls = np.array([273035, 3011023, 27032741])
naive_swaps = np.array([1, 3, 2])

UCB_BS_calls = np.array([444573, 8159738, 20099646, 72461593, 463636348])
UCB_BS_FP1_calls = np.array([387773, 6325418, 15565246, 61494053, 363785788])

UCB_swaps = np.array([1, 3, 2, 2, 5])
UCB_Bonly_calls = np.array([230455, 2124463, 7012741, 34433219, 112562578])


quadratic = 3*(xticks_short)**2
quadratic_FP1 = 1*(xticks_short)**2

nlogn = 2.2e2*xticks*np.log(xticks)
nlogn_FP1 = 1.5e2*xticks*np.log(xticks)

plt.plot(x[:-2], (naive_BS_calls - naive_Bonly_calls)/naive_swaps, 'bo')
plt.plot(x[:-2], (naive_BS_FP1_calls - naive_Bonly_calls)/naive_swaps, 'go')
plt.plot(x, (UCB_BS_calls - UCB_Bonly_calls)/UCB_swaps , 'ro')
plt.plot(x, (UCB_BS_FP1_calls - UCB_Bonly_calls)/UCB_swaps , 'yo')
plt.plot(xticks_short, quadratic, 'b-')
plt.plot(xticks_short, quadratic_FP1, 'g-')
plt.plot(xticks, nlogn, 'r-')
plt.plot(xticks, nlogn_FP1, 'y-')
plt.plot()
plt.show()


# BUILD AND SWAP, SCALING WITH K: k: 2, 3, 4 and N = 3000
k = [2, 3, 4]

naive_BS_calls_FP1 = np.array([27058388, 45136507, 63244050])
ucb_BS_calls_FP1 = np.array([6889761, 15565246, 32008308])

naive_Bonly_calls_FP1 = np.array([18017194, 27032741, 36051842]) # This is the same as without FP1 since FP1 isn't used in build
ucb_Bonly_calls_FP1 = np.array([4345814, 7012741, 12470922]) # This is the same as without FP1 since FP1 isn't used in build

# Subtract p=True, N = 3000, Build only, k = 2, 3, 4

naive_swaps_FP1 = np.array([1, 2, 3])

# linear_naive = 9e6*np.array(k)
# linear_ucb = 1.9e7*(np.array(k) - 5)
plt.plot(k, (naive_BS_calls_FP1 - naive_Bonly_calls_FP1)/naive_swaps_FP1, 'bo')
plt.plot(k, (ucb_BS_calls_FP1 - ucb_Bonly_calls_FP1)/naive_swaps_FP1, 'ro')
# plt.plot(k, (ucb_BS_calls_FP1 - ucb_Bonly_calls_FP1), 'go')
# plt.plot(k, linear_naive)
# plt.plot(k, linear_ucb)
plt.show()
