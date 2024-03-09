import os
# NOTE: set GPU thing here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from fid_score import _compute_statistics_of_path
from inception import InceptionV3

def precompute_CA_single_intra_FID_stats():
    # load sample and specify output path
    male_sample_path = '../../fid_stats/celeba/male_samples.npz'
    male_output_path = '../../fid_stats/celeba/male_fid_stats.npz'
    
    female_sample_path = '../../fid_stats/celeba/female_samples.npz'
    female_output_path = '../../fid_stats/celeba/female_fid_stats.npz'

    cuda = True
    dims = 2048
    batch_size = 100

    if not os.path.exists(male_sample_path):
        raise RuntimeError('Invalid path: %s' % male_sample_path)
        
    if not os.path.exists(female_sample_path):
        raise RuntimeError('Invalid path: %s' % female_sample_path)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    print("calculate FID stats..", end=" ", flush=True)

    mu_male, sigma_male = _compute_statistics_of_path(male_sample_path, model, batch_size, dims, cuda)
    np.savez_compressed(male_output_path, mu=mu_male, sigma=sigma_male)
    
    mu_female, sigma_female = _compute_statistics_of_path(female_sample_path, model, batch_size, dims, cuda)
    np.savez_compressed(female_output_path, mu=mu_female, sigma=sigma_female)

    print("finished saving pre-computed male statistics to: {}".format(male_output_path))
    print("finished saving pre-computed female statistics to: {}".format(female_output_path))


if __name__ == '__main__':
    precompute_CA_single_intra_FID_stats()
