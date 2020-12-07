'''
This script contains scaffolding to run many experiments, listed in a config
file such as auto_exp_config.py.

This script will parse each line (= exp configuration) from the config file and
run the corresponding experiment. It can also run many experiments in parallel
by using the pool.apply_async calls instead of the explicit run_exp calls.

To use FP1 optimization, call:
`python run_profiles -e exp_config.py -p`
'''

import cProfile
import importlib
import multiprocessing as mp
import copy
import traceback

import naive_pam_v1
import ucb_pam
import clarans
import em_style

from data_utils import *

def remap_args(args, exp):
    '''
    Parses a config line (as a list) into an args variable (a Namespace).
    Note that --fast_pam1 flag (-p) must be passed to run_profiles.py in order
    to use the FastPAM1 (FP1) optimization. E.g.
    `python run_profiles -e exp_config.py -p`
    (This is inconsistent with the manner in which other args are passed)
    '''
    args.build_ao_swap = exp[1]
    args.verbose = exp[2]
    args.num_medoids = exp[3]
    args.sample_size = exp[4]
    args.seed = exp[5]
    args.dataset = exp[6]
    args.metric = exp[7]
    args.warm_start_medoids = exp[8]
    args.cache_computed = None
    return args

def get_filename(exp, args):
    '''
    Create the filename suffix for an experiment, given its configuration.
    '''
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
    '''
    Helper method to parse the logstrings (dictionaries) that are passed by the
    different algorithms. Used to write the logs to file (e.g. for debugging)
    '''
    output = "\n"
    for k in sorted(logstring): # Sort keys so they're consistently ordered
        output += "\t" + str(k) + ":\n"
        for round in logstring[k]:
            output += "\t\t" + str(round) + ": " + str(logstring[k][round]) + "\n"
    return output

def write_medoids(medoids_fname, built_medoids, swapped_medoids, B_logstring, S_logstring, num_swaps, final_loss, uniq_d):
    '''
    Write results of an experiment to the given file, including:
    medoids after BUILD step, medoids after SWAP step, etc.
    '''
    with open(medoids_fname, 'w+') as fout:
        fout.write("Built:" + ','.join(map(str, built_medoids)))
        fout.write("\nSwapped:" + ','.join(map(str, swapped_medoids)))
        fout.write("\nNum Swaps: " + str(num_swaps))
        fout.write("\nFinal Loss: " + str(final_loss))
        fout.write("\nUnique Distance Computations: " + str(uniq_d))
        fout.write("\nBuild Logstring:" + parse_logstring(B_logstring))
        fout.write("\nSwap Logstring:" + parse_logstring(S_logstring))

def run_exp(args, method_name, medoids_fname, B_prof_fname, S_prof_fname):
    '''
    Runs an experiment with the given parameters, and writes the results to the
    files (arguments ending in _fname). We run the BUILD step and SWAP step
    separately, so as to get different profiles for each step (so we can
    measure the number of distance calls in the BUILD step and SWAP step
    individually)

    Note that the approach below to profiling is undocumented.
    See https://stackoverflow.com/questions/1584425/return-value-while-using-cprofile
    '''

    tmp = args.build_ao_swap # Store as tmp variable because we modify below
    computed_B = False
    if 'B' in tmp:
        args.build_ao_swap = 'B'
        prof = cProfile.Profile()
        built_medoids, _1, B_logstring, _2, _3, _4, _5 = prof.runcall(method_name, args)
        prof.dump_stats(B_prof_fname)
        computed_B = True
    if 'S' in tmp:
        args.build_ao_swap = 'S'
        assert computed_B, "ERROR: BUILD step was not run, Using warm start medoids from a previous experiment"
        args.warm_start_medoids = ','.join(map(str, list(built_medoids)))
        prof = cProfile.Profile()
        _1, swapped_medoids, _2, S_logstring, num_swaps, final_loss, uniq_d = prof.runcall(method_name, args)
        prof.dump_stats(S_prof_fname)
    write_medoids(medoids_fname, built_medoids, swapped_medoids, B_logstring, S_logstring, num_swaps, final_loss, uniq_d)

def write_loss(medoids_fname, final_medoids, final_loss):
    '''
    Write the final medoids and final loss to a file.

    For some strange reason, this needs to be broken into a separate
    function or it sometimes doesn't write to the logfile correctly when using
    async calls.
    '''
    try:
        with open(medoids_fname, 'w+') as fout:
            print(final_medoids)
            fout.write("Swapped:" + ','.join(map(str, final_medoids)))
            fout.write("\nFinal Loss: " + str(final_loss))
    except:
        print("An exception occurred!")

def run_loss_exp(args, method_name, medoids_fname):
    '''
    This is the same function as run_exp, but only writes the loss and medoids
    instead of the profiles, full logstrings, etc. Used for Figure 1(a).
    '''
    final_medoids, final_loss = method_name(args)
    write_loss(medoids_fname, final_medoids, final_loss)

def main(sys_args):
    '''
    Run all the experiments in the experiments lists specified by the -e
    argument, and write the final results (including logstrings) to files. Can
    run multiple experiments in parallel by using the pool.apply_async calls
    below instead of the explicit run_exp calls.

    Note that clarans and em_style experiments are only run for loss comparison
    (in Figure 1(a)).
    '''
    args = get_args(sys.argv[1:]) # Uses default values for now as placeholder to instantiate args

    imported_config = importlib.import_module(args.exp_config.strip('.py'))
    pool = mp.Pool()
    for exp in imported_config.experiments:
        args = remap_args(args, exp)
        B_prof_fname = os.path.join('profiles', 'p-B-' + get_filename(exp, args))
        S_prof_fname = os.path.join('profiles', 'p-S-' + get_filename(exp, args))
        medoids_fname = os.path.join('profiles', 'L-' + get_filename(exp, args))

        if os.path.exists(medoids_fname) and not args.force:
            # Experiments have already been conducted
            print("Warning: already have data for experiment", B_prof_fname)
            print("Warning: already have data for experiment", S_prof_fname)
            continue
        else:
            print("Running exp:", B_prof_fname, S_prof_fname)

        '''
        WARNING: The apply_async calls below are NOT threadsafe. In particular,
        strings in python are lists, which means they are passed by reference.
        This means that if a NEW thread gets the SAME reference as the other
        threads, and updates the object, the OLD thread will write to the wrong
        file. Therefore, whenever using multiprocessing, need to copy.deepcopy()
        all the arguments. Don't need to do this for the explicit run_exp calls
        though since those references are used appropriately (executed
        sequentially)
        '''
        try:
            if exp[0] == 'naive_v1':
                # pool.apply_async(run_exp, args=(copy.deepcopy(args), naive_pam_v1.naive_build_and_swap, copy.deepcopy(medoids_fname), copy.deepcopy(B_prof_fname), copy.deepcopy(S_prof_fname))) # Copy inline to copy OTF
                run_exp(args, naive_pam_v1.naive_build_and_swap, medoids_fname, B_prof_fname, S_prof_fname)
            elif exp[0] == 'ucb':
                # pool.apply_async(run_exp, args=(copy.deepcopy(args), ucb_pam.UCB_build_and_swap, copy.deepcopy(medoids_fname), copy.deepcopy(B_prof_fname), copy.deepcopy(S_prof_fname)))
                run_exp(args, ucb_pam.UCB_build_and_swap, medoids_fname, B_prof_fname, S_prof_fname)
            elif exp[0] == 'clarans':
                # pool.apply_async(run_loss_exp, args=(copy.deepcopy(args), clarans.CLARANS_build_and_swap, copy.deepcopy(medoids_fname)))
                run_loss_exp(copy.deepcopy(args), clarans.CLARANS_build_and_swap, copy.deepcopy(medoids_fname))
            elif exp[0] == 'em_style':
                # pool.apply_async(run_loss_exp, args=(copy.deepcopy(args), em_style.EM_build_and_swap, copy.deepcopy(medoids_fname)))
                run_loss_exp(copy.deepcopy(args), em_style.EM_build_and_swap, copy.deepcopy(medoids_fname))
            else:
                raise Exception('Invalid algorithm specified')
        except Exception as e:
            track = traceback.format_exc()
            print(track)

    pool.close()
    pool.join()
    print("Finished")

if __name__ == "__main__":
    main(sys.argv)
