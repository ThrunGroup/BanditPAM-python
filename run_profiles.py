from data_utils import *
from exp_config import experiments
import cProfile

import naive_pam_v0
import naive_pam_v1
import ucb_pam

def remap_args(args, exp):
    args.build_ao_swap = exp[1]
    args.verbose = exp[2]
    args.num_medoids = exp[3]
    args.sample_size = exp[4]
    args.seed = exp[5]
    args.dataset = exp[6]
    args.warm_start_medoids = exp[-1] # Usually 6
    return args

def get_filename(exp, args):
    return exp[0] + \
        '-' + args.build_ao_swap + \
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

        if exp[0] == 'naive_v1':
            prof = cProfile.Profile()
            # NOTE: This approach is undocumented
            # See https://stackoverflow.com/questions/1584425/return-value-while-using-cprofile
            built_medoids, swapped_medoids = prof.runcall(naive_pam_v1.naive_build_and_swap, args)
            prof.dump_stats(fname)
            with open(fname + '.medoids', 'w+') as fout:
                fout.write("Built:", + ','.join(map(str, built_medoids)))
                fout.write("Swapped:", + ','.join(map(str, swapped_medoids)))
        elif exp[0] == 'ucb':
            prof = cProfile.Profile()
            built_medoids, swapped_medoids = prof.runcall(ucb_pam.UCB_build_and_swap, args) # Need *[args, imgs] so [args, imgs] is not interpreted as args, imgs = [args, imgs], None and instead as args, imgs = args, imgs
            prof.dump_stats(fname)
            with open(fname + '.medoids', 'w+') as fout:
                fout.write("Built:", + ','.join(map(str, built_medoids)))
                fout.write("Swapped:" + ','.join(map(str, swapped_medoids)))
        else:
            raise Exception('Invalid algorithm specified')


if __name__ == "__main__":
    main(sys.argv)
