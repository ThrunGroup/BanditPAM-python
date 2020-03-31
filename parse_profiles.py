import sys
import os
import pstats

def main():
    profiles = [os.path.join('profiles', x) for x in os.listdir('profiles')]
    for profile in profiles:
        p = pstats.Stats(profile)
        p.strip_dirs().print_stats('data_utils')

if __name__ == '__main__':
    main()
