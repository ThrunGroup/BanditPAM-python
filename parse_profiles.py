import sys
import os
import pstats

import snakevizcode

FN_NAME = 'data_utils.py:104(empty_counter)'

def verify_logfiles():
    ucb_logfiles = [os.path.join('profiles', x) for x in os.listdir('profiles') if os.path.isfile(os.path.join('profiles', x)) and x != '.DS_Store' and x[:5] == 'L-ucb']
    for u_lfile in ucb_logfiles:
        n_lfile = u_lfile.replace('ucb', 'naive_v1')
        if not os.path.exists(n_lfile):
            print("Warning: no naive experiment", n_lfile)
        else:
            with open(u_lfile, 'r') as fin1:
                with open(n_lfile, 'r') as fin2:
                    for i in range(2): # Verify that the top two lines, built medoids and swap medoids, are equal
                        if fin1.readline() != fin2.readline():
                            print("ERROR: Results for", u_lfile, "disagree!!")
                    print("OK: Results for", u_lfile, "agree")

def main():
    algos = ['ucb', 'naive_v1']
    dataset = 'MNIST'
    metric = 'L2'

    Ns = [1000, 3000, 10000, 30000, 70000]
    ks = [2, 3, 4, 5, 10, 20, 30]
    seeds = range(10)
    
    profiles = [os.path.join('profiles', x) for x in os.listdir('profiles') if os.path.isfile(os.path.join('profiles', x)) and x != '.DS_Store' and x[:2] == 'p-']
    for profile in profiles:
        p = pstats.Stats(profile)
        for row in snakevizcode.table_rows(p):
            if FN_NAME in row:
                d_calls = row[0][1]
                print(profile, d_calls)

if __name__ == '__main__':
    verify_logfiles()
    main()
