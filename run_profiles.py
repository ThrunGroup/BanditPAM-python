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
    args.warm_start_medoids = exp[-1] # Usually 6
    return args

def get_filename(exp, args):
    return exp[0] + \
        '-v-' + str(args.verbose) + \
        '-k-' + str(args.num_medoids) + \
        '-N-' + str(args.sample_size) + \
        '-s-' + str(args.seed) + \
        '-d-' + args.dataset + \
        '-w-' + args.warm_start_medoids

def main(sys_args):
    args = get_args(sys.argv[1:]) # Uses default values for now as placeholder to instantiate args

    for exp in experiments:
        args = remap_args(args, exp)
        fname = os.path.join('profiles', get_filename(exp, args))

        if os.path.exists(fname) and not args.force:
            print("Already have data for experiment", fname)
            continue
        else:
            print("Running exp:", fname)
            
        if exp[0] == 'naive':
            prof = cProfile.Profile()
            # NOTE: This approach is undocumented
            # See https://stackoverflow.com/questions/1584425/return-value-while-using-cprofile
            medoids = prof.runcall(naive_pam.naive_build_and_swap, args)
            prof.dump_stats(fname)
            with open(fname + '.medoids', 'w+') as fout:
                fout.write(','.join(map(str, medoids)))
        elif exp[0] == 'ucb':
            prof = cProfile.Profile()
            medoids = prof.runcall(ucb_pam.UCB_build_and_swap, args) # Need *[args, imgs] so [args, imgs] is not interpreted as args, imgs = [args, imgs], None and instead as args, imgs = args, imgs
            prof.dump_stats(fname)
            with open(fname + '.medoids', 'w+') as fout:
                fout.write(','.join(map(str, medoids)))
        else:
            raise Exception('Invalid algorithm specified')


if __name__ == "__main__":
    main(sys.argv)
