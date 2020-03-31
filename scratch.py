from data_utils import *

args = get_args(sys.argv[1:])
np.random.seed(args.seed)
total_images, total_labels = load_data(args)
sample_size = args.sample_size
imgs = total_images[np.random.choice(range(len(total_images)), size = sample_size, replace = False)]
estimate_sigma(imgs)
