from data_utils import *
from exp_config import experiments
import cProfile

import naive_pam
import ucb_pam

def remap_args(args, exp):
    args.verbose = exp[1]
    args.num_medoids = exp[2]
    args.sample_size = exp[3]
    args.seed = exp[4]
    args.dataset = exp[5]
    return args

def get_filename(exp, args):
    return exp[0] + \
        '-verbosity-' + str(args.verbose) + \
        '-num_medoids-' + str(args.num_medoids) + \
        '-sample_size-' + str(args.sample_size) + \
        '-seed-' + str(args.seed) + \
        '-dataset-' + args.dataset

def main(sys_args):
    args = get_args(sys.argv[1:]) # Uses default values for now as placeholder to instantiate args

    for exp_idx in experiments:
        exp = experiments[exp_idx]
        np.random.seed(args.seed)
        args = remap_args(args, exp)
        total_images, total_labels, sigma = load_data(args)
        imgs = total_images[np.random.choice(range(len(total_images)), size = args.sample_size, replace = False)]
        fname = os.path.join('profiles', get_filename(exp, args))

        # NOTE: Save medoids found to file
        if exp[0] == 'naive':
            cProfile.runctx('medoids = naive_pam.naive_build(args, imgs)', globals(), locals(), fname)
            print("run_profiles", medoids)
        elif exp[0] == 'ucb':
            cProfile.runctx('medoids = ucb_pam.UCB_build(args, imgs, sigma)', globals(), locals(), fname)
            print("run_profiles", medoids)
        else:
            raise Exception('Invalid algorithm specified')





if __name__ == "__main__":
    main(sys.argv)
