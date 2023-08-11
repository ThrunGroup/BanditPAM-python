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
import banditpam
import clarans

from data_utils import *

def remap_args(args, exp):
    '''
    Parses a config line (as a list) into an args variable (a Namespace).
    Note that --fast_pam1 flag (-p) must be passed to run_profiles.py in order
    to use the FastPAM1 (FP1) optimization. E.g.
    `python run_profiles -e exp_config.py -p`
    (This is inconsistent with the manner in which other args are passed)
    '''
    args.build_ao_swap = exp[1]  # TODO(@Adarsh321123): irrelevant?
    args.verbose = exp[2]  # TODO(@Adarsh321123): irrelevant
    args.num_medoids = exp[3]
    args.sample_size = exp[4]
    args.seed = exp[5]
    args.dataset = exp[6]
    args.metric = exp[7]
    args.warm_start_medoids = exp[8]  # TODO(@Adarsh321123): do we not use warm start??
    args.cache_computed = None  # TODO(@Adarsh321123): irrelevant?
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

def write_medoids(medoids_fname, built_medoids, swapped_medoids, num_swaps, final_loss):
    '''
    Write results of an experiment to the given file, including:
    medoids after BUILD step, medoids after SWAP step, etc.
    '''
    with open(medoids_fname, 'w+') as fout:
        fout.write("Built:" + ','.join(map(str, built_medoids)))
        fout.write("\nSwapped:" + ','.join(map(str, swapped_medoids)))
        fout.write("\nNum Swaps: " + str(num_swaps))
        fout.write("\nFinal Loss: " + str(final_loss))

def run_exp(args, object_name, medoids_fname, B_prof_fname, S_prof_fname):
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
    total_images, total_labels, sigma = load_data(args)
    np.random.seed(args.seed)
    if args.metric == 'PRECOMP':
        dist_mat = np.loadtxt('tree-3630.dist')
        random_indices = np.random.choice(len(total_images), size=args.sample_size, replace=False)
        imgs = np.array([total_images[x] for x in random_indices])
        dist_mat = dist_mat[random_indices][:, random_indices]
    elif args.metric == 'TREE':
        imgs = np.random.choice(total_images, size=args.sample_size, replace=False)
    else:
        # Can remove range() here?
        imgs = total_images[np.random.choice(range(len(total_images)), size=args.sample_size, replace=False)]
    # X = pd.read_csv('data/MNIST_70k.csv', sep=' ', header=None).to_numpy()
    # import ipdb;ipdb.set_trace()
    # np.random.seed(args.seed)
    # # Can remove range() here?
    # imgs = X[np.random.choice(range(len(X)), size=args.sample_size, replace=False)]
    computed_B = False
    # TODO(@Adarsh321123): clean up stuff below (e.g. warm start is not needed) (does it make sense to split the B and S anymore?)
    if 'B' in tmp:
        args.build_ao_swap = 'B'
        prof = cProfile.Profile()
        prof.runcall(object_name.fit, imgs, args.metric)
        built_medoids = object_name.build_medoids
        swapped_medoids = object_name.medoids
        # B_logstring and S_logstring are unnecessary for the main paper
        num_swaps = object_name.steps
        final_loss = object_name.average_loss
        # uniq_d is not needed either
        prof.dump_stats(B_prof_fname)
        computed_B = True
    # TODO(@Adarsh321123): this is essentially identical to the top if statement but is also run, is this needed?
    if 'S' in tmp:
        args.build_ao_swap = 'S'
        assert computed_B, "ERROR: BUILD step was not run, Using warm start medoids from a previous experiment"
        args.warm_start_medoids = ','.join(map(str, list(built_medoids)))
        prof = cProfile.Profile()
        prof.runcall(object_name.fit, imgs, args.metric)
        built_medoids = object_name.build_medoids
        swapped_medoids = object_name.medoids
        # B_logstring and S_logstring are unnecessary for the main paper
        num_swaps = object_name.steps
        final_loss = object_name.average_loss
        # uniq_d is not needed either
        prof.dump_stats(S_prof_fname)
    write_medoids(medoids_fname, built_medoids, swapped_medoids, num_swaps, final_loss)

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
                # TODO(@Adarsh321123): now this is broken since it passes the method and not the object name
                run_exp(args, naive_pam_v1.naive_build_and_swap, medoids_fname, B_prof_fname, S_prof_fname)
            elif exp[0] == 'ucb': # TODO(@Adarsh321123): change this to say BanditPAM throughout?
                # pool.apply_async(run_exp, args=(copy.deepcopy(args), ucb_pam.UCB_build_and_swap, copy.deepcopy(medoids_fname), copy.deepcopy(B_prof_fname), copy.deepcopy(S_prof_fname)))
                # TODO(@Adarsh321123): deal with the double counting of swap steps later
                kmed = banditpam.KMedoids(n_medoids=args.num_medoids, algorithm="BanditPAM_orig")
                run_exp(args, kmed, medoids_fname, B_prof_fname, S_prof_fname)
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
    # TODO(@Adarsh321123): see if it is a problem that the profile sometimes has one different built medoid and therefore 5 swaps instead of 4