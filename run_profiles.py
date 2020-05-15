from data_utils import *
import cProfile
import importlib

import naive_pam_v0
import naive_pam_v1
import ucb_pam

def remap_args(args, exp):
    # NOTE: FP1 arg is passed inconsistently to the experiment, as part of args Namespace
    # NOTE: To use FP1 optimization, call p run_profiles -e exp_config.py -p
    args.build_ao_swap = exp[1]
    args.verbose = exp[2]
    args.num_medoids = exp[3]
    args.sample_size = exp[4]
    args.seed = exp[5]
    args.dataset = exp[6]
    args.metric = exp[7]
    args.warm_start_medoids = exp[8]
    return args

def get_filename(exp, args):
    return exp[0] + \
        '-' + str(args.fast_pam1) + \
        '-' + args.build_ao_swap + \
        '-v-' + str(args.verbose) + \
        '-k-' + str(args.num_medoids) + \
        '-N-' + str(args.sample_size) + \
        '-s-' + str(args.seed) + \
        '-d-' + args.dataset + \
        '-m-' + args.metric + \
        '-w-' + args.warm_start_medoids

def parse_logstring(logstring):
    output = "\n"
    for k in logstring:
        output += "\t" + str(k) + ":\n"
        for round in logstring[k]:
            output += "\t\t" + str(round) + ": " + str(logstring[k][round]) + "\n"
    return output
def write_medoids(medoids_fname, built_medoids, swapped_medoids, B_logstring, S_logstring):
    with open(medoids_fname, 'w+') as fout:
        fout.write("Built:" + ','.join(map(str, built_medoids)))
        fout.write("\nSwapped:" + ','.join(map(str, swapped_medoids)))
        fout.write("\nBuild Logstring:" + parse_logstring(B_logstring))
        fout.write("\nSwap Logstring:" + parse_logstring(S_logstring))

def main(sys_args):
    args = get_args(sys.argv[1:]) # Uses default values for now as placeholder to instantiate args

    imported_config = importlib.import_module(args.exp_config.strip('.py'))
    for exp in imported_config.experiments:
        args = remap_args(args, exp)
        B_prof_fname = os.path.join('profiles', 'B-' + get_filename(exp, args))
        S_prof_fname = os.path.join('profiles', 'S-' + get_filename(exp, args))
        medoids_fname = os.path.join('profiles', 'medoids.' + get_filename(exp, args))

        if (os.path.exists(B_prof_fname) or os.path.exists(S_prof_fname)) and not args.force:
            print("Warning: already have data for experiment", B_prof_fname)
            print("Warning: already have data for experiment", S_prof_fname)
            continue
        else:
            print("Running exp:", B_prof_fname, S_prof_fname)

        # NOTE: This approach to profiling is undocumented
        # See https://stackoverflow.com/questions/1584425/return-value-while-using-cprofile
        tmp = args.build_ao_swap # Store as tmp variable because we modify below
        if exp[0] == 'naive_v1':
            computed_B = False
            if 'B' in tmp:
                args.build_ao_swap = 'B'
                prof = cProfile.Profile()
                built_medoids, _1, B_logstring, _2 = prof.runcall(naive_pam_v1.naive_build_and_swap, args)
                prof.dump_stats(B_prof_fname)
                computed_B = True
            if 'S' in tmp:
                args.build_ao_swap = 'S'
                assert computed_B, "ERROR: Using warm start medoids from a previous experiment"
                print(list(built_medoids))
                args.warm_start_medoids = ','.join(map(str,list(built_medoids)))
                prof = cProfile.Profile()
                _1, swapped_medoids, _2, S_logstring = prof.runcall(naive_pam_v1.naive_build_and_swap, args)
                prof.dump_stats(S_prof_fname)
            write_medoids(medoids_fname, built_medoids, swapped_medoids, B_logstring, S_logstring)
        elif exp[0] == 'ucb':
            computed_B = False
            if 'B' in tmp:
                args.build_ao_swap = 'B'
                prof = cProfile.Profile()
                built_medoids, _1, B_logstring, _2 = prof.runcall(ucb_pam.UCB_build_and_swap, args) # Need *[args, imgs] so [args, imgs] is not interpreted as args, imgs = [args, imgs], None and instead as args, imgs = args, imgs
                prof.dump_stats(B_prof_fname)
                computed_B = True
            if 'S' in tmp:
                args.build_ao_swap = 'S'
                assert computed_B, "ERROR: Using warm start medoids from a previous experiment"
                print(list(built_medoids))
                args.warm_start_medoids = ','.join(map(str,list(built_medoids)))
                prof = cProfile.Profile()
                _1, swapped_medoids, _2, S_logstring = prof.runcall(ucb_pam.UCB_build_and_swap, args) # Need *[args, imgs] so [args, imgs] is not interpreted as args, imgs = [args, imgs], None and instead as args, imgs = args, imgs
                prof.dump_stats(S_prof_fname)
            write_medoids(medoids_fname, built_medoids, swapped_medoids, B_logstring, S_logstring)
        else:
            raise Exception('Invalid algorithm specified')


if __name__ == "__main__":
    main(sys.argv)
