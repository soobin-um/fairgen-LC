import os
# NOTE: set GPU thing here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from fid_score import _compute_statistics_of_path
from inception import InceptionV3

def precompute_biased_FID_stats(args):
    assert args.dataset in ['celeba', 'UTKFace', 'FairFace']
    
    if args.dataset in ['celeba']:
        if args.multi:
            attr_type = 'multi'
        else:
            attr_type = 'gender'
    elif args.dataset in ['UTKFace', 'FairFace']:
        attr_type = 'race'
    
    # load sample and specify output path
    sample_path = '../../fid_stats/{}/biased_perc{}_all_{}_samples.npz'.format(args.dataset, args.perc, attr_type)
    output_path = '../../fid_stats/{}/biased_perc{}_all_{}_fid_stats.npz'.format(args.dataset, args.perc, attr_type)

    cuda = True
    dims = 2048
    batch_size = 100

    if not os.path.exists(sample_path):
        raise RuntimeError('Invalid path: %s' % sample_path)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    print("calculate FID stats..", end=" ", flush=True)

    mu, sigma = _compute_statistics_of_path(sample_path, model, batch_size, dims, cuda)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)

    print("finished saving pre-computed statistics to: {}".format(output_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='interested dataset: celeba, UTKFace, or FairFace')
    parser.add_argument('--perc', type=float, default='0.1', 
        help="Flag for percentage of balanced dataset: [0.01, 0.025, 0.05, 0.1, 0.25]")
    parser.add_argument('--multi', type=int, default=0, help="Flag for multi-attribute experiments")
    args = parser.parse_args()
    precompute_biased_FID_stats(args)
