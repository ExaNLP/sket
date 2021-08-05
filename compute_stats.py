import json
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--outputs', default='./outputs/concepts/refined/aoec/lung/*.json', type=str, help='SKET results file.')
parser.add_argument('--use_case', default='lung', choices=['colon', 'cervix', 'lung'], help='Considered use-case.')
parser.add_argument('--hospital', default='aoec', choices=['aoec', 'radboud'], help='Considered hospital.')
args = parser.parse_args()


def main():
    # read SKET results
    if '*.json' == args.outputs.split('/')[-1]:  # read files
        # read file paths
        rsfps = glob.glob(args.outputs)
        # set dict
        rs = {}
        for rsfp in rsfps:
            with open(rsfp, 'r') as rsf:
                rs.update(json.load(rsf))
    else:  # read file
        with open(args.outputs, 'r') as rsf:
            rs = json.load(rsf)

    stats = []
    # loop over reports and store size
    for rid, rdata in rs.items():
        stats.append(sum([len(sem_data) for sem_cat, sem_data in rdata.items()]))
    # convert into numpy
    stats = np.array(stats)
    print('size: {}'.format(np.size(stats)))
    print('max: {}'.format(np.max(stats)))
    print('min: {}'.format(np.min(stats)))
    print('mean: {}'.format(np.mean(stats)))
    print('std: {}'.format(np.std(stats)))
    print('tot: {}'.format(np.sum(stats)))


if __name__ == "__main__":
    main()
