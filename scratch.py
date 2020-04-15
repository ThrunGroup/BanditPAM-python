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



# Build only, k = 3, N = 300, 1000, 3000, 10000
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


# Build only, scaling with k: 1, 2, 3 and N = 3000
k = [1, 2, 3, 4, 5, 10]

naive_B_calls = np.array([9006000, 18017194, 27032741, 36051842, 45074455, 90236487])
ucb_B_calls = np.array([284820, 4345814, 7012741, 12470922, 17469475, 95086987])

linear_naive = 9e6*np.array(k)
linear_ucb = 3.5e6*(np.array(k))
plt.plot(k, naive_B_calls, 'bo')
plt.plot(k, ucb_B_calls, 'ro')
plt.plot(k, linear_naive)
plt.plot(k, linear_ucb)
plt.show()

# Build and Swap, k = 3, N = 1000, 3000, 10000


# Build and Swap, scaling with k: 1, 2, 3 and N = 3000
