import sys
import os
import pstats

import snakevizcode

FN_NAME = 'data_utils.py:104(empty_counter)'

def main():
    profiles = [os.path.join('profiles', x) for x in os.listdir('profiles') if os.path.isfile(os.path.join('profiles', x)) and x != '.DS_Store']
    for profile in profiles:
        p = pstats.Stats(profile)
        for row in snakevizcode.table_rows(p):
            if FN_NAME in row:
                print(row[0][1], type(row[0][1]))

if __name__ == '__main__':
    main()
