import os
# NOTE: set GPU thing here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from fid_score import _compute_statistics_of_path
from inception import InceptionV3

def precompute_CA_multi_intra_FID_stats():
    # load sample and specify output path
    unblack_female_sample_path = '../../fid_stats/celeba/unblack_female_samples.npz'
    unblack_female_output_path = '../../fid_stats/celeba/unblack_female_fid_stats.npz'
    
    unblack_male_sample_path = '../../fid_stats/celeba/unblack_male_samples.npz'
    unblack_male_output_path = '../../fid_stats/celeba/unblack_male_fid_stats.npz'
    
    black_female_sample_path = '../../fid_stats/celeba/black_female_samples.npz'
    black_female_output_path = '../../fid_stats/celeba/black_female_fid_stats.npz'
    
    black_male_sample_path = '../../fid_stats/celeba/black_male_samples.npz'
    black_male_output_path = '../../fid_stats/celeba/black_male_fid_stats.npz'

    cuda = True
    dims = 2048
    batch_size = 100

    if not os.path.exists(unblack_female_sample_path):
        raise RuntimeError('Invalid path: %s' % unblack_female_sample_path)
        
    if not os.path.exists(unblack_male_sample_path):
        raise RuntimeError('Invalid path: %s' % unblack_male_sample_path)
        
    if not os.path.exists(black_female_sample_path):
        raise RuntimeError('Invalid path: %s' % black_female_sample_path)
        
    if not os.path.exists(black_male_sample_path):
        raise RuntimeError('Invalid path: %s' % black_male_sample_path)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    print("calculate FID stats..", end=" ", flush=True)

    mu_unblack_female, sigma_unblack_female = _compute_statistics_of_path(unblack_female_sample_path, model, batch_size, dims, cuda)
    np.savez_compressed(unblack_female_output_path, mu=mu_unblack_female, sigma=sigma_unblack_female)
    
    mu_unblack_male, sigma_unblack_male = _compute_statistics_of_path(unblack_male_sample_path, model, batch_size, dims, cuda)
    np.savez_compressed(unblack_male_output_path, mu=mu_unblack_male, sigma=sigma_unblack_male)
     
    mu_black_female, sigma_black_female = _compute_statistics_of_path(black_female_sample_path, model, batch_size, dims, cuda)
    np.savez_compressed(black_female_output_path, mu=mu_black_female, sigma=sigma_black_female)

    mu_black_male, sigma_black_male = _compute_statistics_of_path(black_male_sample_path, model, batch_size, dims, cuda)
    np.savez_compressed(black_male_output_path, mu=mu_black_male, sigma=sigma_black_male)
    
    print("finished saving pre-computed unblack-haired female statistics to: {}".format(unblack_female_output_path))
    print("finished saving pre-computed unblack-haired male statistics to: {}".format(unblack_male_output_path))
    print("finished saving pre-computed black-haired female statistics to: {}".format(black_female_output_path))
    print("finished saving pre-computed black-haired male statistics to: {}".format(black_male_output_path))

if __name__ == '__main__':
    precompute_CA_multi_intra_FID_stats()
