import os
# NOTE: set GPU thing here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from fid_score import _compute_statistics_of_path
from inception import InceptionV3

def precompute_UTKFace_intra_FID_stats():
    # load sample and specify output path
    non_white_sample_path = '../../fid_stats/UTKFace/non_white_samples.npz'
    non_white_output_path = '../../fid_stats/UTKFace/non_white_fid_stats.npz'
    
    white_sample_path = '../../fid_stats/UTKFace/white_samples.npz'
    white_output_path = '../../fid_stats/UTKFace/white_fid_stats.npz'

    cuda = True
    dims = 2048
    batch_size = 100

    if not os.path.exists(non_white_sample_path):
        raise RuntimeError('Invalid path: %s' % non_white_sample_path)
        
    if not os.path.exists(white_sample_path):
        raise RuntimeError('Invalid path: %s' % white_sample_path)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    print("calculate FID stats..", end=" ", flush=True)

    mu_non_white, sigma_non_white = _compute_statistics_of_path(non_white_sample_path, model, batch_size, dims, cuda)
    np.savez_compressed(non_white_output_path, mu=mu_non_white, sigma=sigma_non_white)
    
    mu_white, sigma_white = _compute_statistics_of_path(white_sample_path, model, batch_size, dims, cuda)
    np.savez_compressed(white_output_path, mu=mu_white, sigma=sigma_white)

    print("finished saving pre-computed non_white statistics to: {}".format(non_white_output_path))
    print("finished saving pre-computed white statistics to: {}".format(white_output_path))


if __name__ == '__main__':
    precompute_UTKFace_intra_FID_stats()
