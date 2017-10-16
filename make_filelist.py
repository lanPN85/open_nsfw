from __future__ import print_function
from argparse import ArgumentParser

import os


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(dest='ROOT_DIR')
    parser.add_argument(dest='DATA_DIR')
    parser.add_argument('--label', dest='LABEL', default='0')
    return parser.parse_args()


def main(args):
    dir = os.path.join(args.ROOT_DIR,  args.DATA_DIR)
    filelist = os.listdir(dir)

    for fn in filelist:
        path = os.path.join(args.DATA_DIR, fn)
        print('%s %s' % (path, args.LABEL))


if __name__ == '__main__':
    main(parse_arguments())
