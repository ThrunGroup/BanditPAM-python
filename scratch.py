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
x = [100, 300, 1000, 3000]
xticks = np.arange(100, 3100, 100)

naive_B_calls = np.array([107759, 922363, 10079346, 90236487])
ucb_B_calls = np.array([167319, 1330223, 15952226, 95086987])

plt.plot(x, np.log(naive_B_calls), 'bo')
plt.plot(x, np.log(ucb_B_calls), 'ro')
quadratic = (3*xticks)**2
nlogn = (xticks*np.log(xticks))

plt.plot(xticks, np.log(quadratic), 'b-')
plt.plot(xticks, np.log(nlogn), 'r-')
plt.title("Number of distance calls for k = 10 medoids")
plt.xlabel("N")
plt.ylabel("Calls to d")
plt.show()
