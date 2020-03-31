from data_utils import *
from exp_config import experiments

def main(sys_args):
    args = get_args(sys.argv[1:])
    np.random.seed(args.seed)
    total_images, total_labels = load_data(args)
    imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]

    for exp in experiments:
        # run profile
        # save results to profiles/name


if __name__ == "__main__":
    main(sys.argv)
