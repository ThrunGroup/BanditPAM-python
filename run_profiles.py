from data_utils import *
from exp_config import experiments

import naive_pam
import ucb_pam

def remap_args(args, exp):
    args.verbose = exp[1]
    args.num_medoids = exp[2]
    args.sample_size = exp[3]
    args.seed = exp[4]
    args.dataset = exp[5]
    return args

def main(sys_args):
    args = get_args(sys.argv[1:])
    total_images, total_labels, sigma = load_data(args)

    for exp_idx in experiments:
        exp = experiments[exp_idx]
        np.random.seed(args.seed)
        args = remap_args(args, exp)
        imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
        if exp[0] == 'naive':
            naive_pam.naive_build(args, imgs)
        elif exp[0] == 'ucb':
            ucb_pam.UCB_build(args, imgs, sigma)
        else:
            raise Exception('Invalid algorithm specified')


        # run profile
        # save results to profiles/name


if __name__ == "__main__":
    main(sys.argv)
