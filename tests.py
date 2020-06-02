import tests_exp_config
import run_profiles
import data_utils
import naive_pam_v1
import ucb_pam
import sys
import argparse

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def test_exps():
    for exp in tests_exp_config.experiments:
        print(exp)
        args = Namespace()
        args = run_profiles.remap_args(args, exp)
        args.fast_pam1 = True

        if exp[0] == 'naive_v1':
            built_medoids, swapped_medoids, _1, _2, _3, _4 = naive_pam_v1.naive_build_and_swap(args)
        elif exp[0] == 'ucb':
            built_medoids, swapped_medoids, _1, _2, _3, _4 = ucb_pam.UCB_build_and_swap(args)
        else:
            raise Exception('Invalid algorithm specified')

        assert(built_medoids == exp[-2]), "Build method failed for exp " + str(exp)
        assert(swapped_medoids == exp[-1]), "Swap method failed for exp " + str(exp)
