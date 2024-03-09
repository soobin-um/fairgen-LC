''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import os
import glob
import sys

import functools
import math
import numpy as np
from tqdm import tqdm, trange
import pickle


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import fid_score # modified by sum
import inception_utils
import utils
import losses
from clf_models import ResNet18, BasicBlock


CLF_PATH_celeba = '../src/results/celeba/attr_clf/model_best.pth.tar'
MULTI_CLF_PATH_celeba = '../src/results/celeba/multi_clf/model_best.pth.tar'
CLF_PATH_UTKFace = '../src/results/UTKFace/attr_clf/model_best.pth.tar'
CLF_PATH_FairFace = '../src/results/FairFace/attr_clf/model_best.pth.tar'


def classify_examples(model, sample_path):
    """
    classifies generated samples into appropriate classes 
    """
    model.eval()
    preds = []
    probs = []
    samples = np.load(sample_path)['x']
    n_batches = samples.shape[0] // 1000

    with torch.no_grad():
        # generate 10K samples
        for i in range(n_batches):
            x = samples[i*1000:(i+1)*1000]
            samp = x / 255.  # renormalize to feed into classifier
            samp = torch.from_numpy(samp).to('cuda').float()

            # get classifier predictions
            logits, probas = model(samp)
            _, pred = torch.max(probas, 1)
            probs.append(probas)
            preds.append(pred)
        preds = torch.cat(preds).data.cpu().numpy()
        probs = torch.cat(probs).data.cpu().numpy()

    return preds, probs


def run(config):
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num_fair': 0, 'save_best_num_fid': 0, 'best_IS': 0, 'best_FID': 999999, 'best_fair_d': 999999, 'config': config}

    # Optionally, get the configuration from the state dict. This allows for
    # recovery of the config provided only a state dict and experiment name,
    # and can be convenient for writing less verbose sample shell scripts.
    if config['config_from_name']:
        utils.load_weights(None, None, state_dict, config['weights_root'],
                           config['experiment_name'], config['load_weights'], None,
                           strict=False, load_optim=False)
        # Ignore items which we might want to overwrite from the command line
        for item in state_dict['config']:
            if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
                config[item] = state_dict['config'][item]

    # update config (see train.py for explanation)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = 1
    if config['conditional']:
        config['n_classes'] = 2
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    device = 'cuda'
    config['sample_num_npz'] = 10000
    print(config['ema_start'])
    
    # load classifier
    # multi-attribute
    print('Pre-loading pre-trained multi-attribute classifier...')
    clf_state_dict = torch.load(MULTI_CLF_PATH_celeba)['state_dict']
    clf_classes = 4
    # load attribute classifier here
    clf = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], 
                    num_classes=clf_classes, grayscale=False) 
    clf.load_state_dict(clf_state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = clf.to(device)
    clf.eval()  # turn off batch norm
    
    # classify examples and get probabilties
    n_classes = 4

    # Seed RNG
    # utils.seed_rng(config['seed'])  # config['seed'])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    G = model.Generator(**config).cuda()
    utils.count_parameters(G)

    # Load weights
    print('Loading weights...')
    print('sampling from model with best %s...' % (config['mode']))

    # find best weights for either FID or fair checkpointing
    weights_root = config['weights_root']
    ckpts = glob.glob(os.path.join(weights_root, experiment_name, 'state_dict_best_{}*'.format(config['mode'])))
    best_ckpt = 'best_{}{}'.format(config['mode'],len(ckpts)-1)
    config['load_weights'] = best_ckpt if config['load_weights'] == '' else config['load_weights']

    # load weights to sample from generator
    utils.load_weights(G if not (config['use_ema']) else None, None, state_dict, weights_root, experiment_name, config['load_weights'], G if config['ema'] and config['use_ema'] else None,
        strict=False, load_optim=False)

    # Update batch size setting used for G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'],
                               z_var=config['z_var'])
    
    if config['G_eval_mode']:
        print('Putting G in eval mode..')
        G.eval()

    # Sample function
    sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
    if config['accumulate_stats']:
        print('Accumulating standing stats across %d accumulations...' %
              config['num_standing_accumulations'])
        utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])

    # # # Sample a number of images and save them to an NPZ, for use with TF-Inception
    sample_path = '%s/%s/' % (config['samples_root'], experiment_name)
    print('looking in sample path {}'.format(sample_path))
    if not os.path.exists(sample_path):
        print('creating sample path: {}'.format(sample_path))
        os.mkdir(sample_path)

    # Lists to hold images and labels for images
    print('saving samples from best %s checkpoint' % (config['mode']))
    # sampling 10 sets of 10K samples
    for k in range(10):
        npz_filename = '%s/%s/%s_%s_%s_samples_%s.npz' % (
            config['samples_root'], experiment_name, config['mode'],
            'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train', k)
        if os.path.exists(npz_filename):
            print('samples already exist, skipping...')
            continue
        x, y = [], []
        print('Sampling %d images and saving them to npz...' %
              config['sample_num_npz'])
        for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
            with torch.no_grad():
                images, labels = sample()
            x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
            y += [labels.cpu().numpy()]
        x = np.concatenate(x, 0)[:config['sample_num_npz']]
        y = np.concatenate(y, 0)[:config['sample_num_npz']]
        print('checking labels: {}'.format(y.sum()))
        print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
        npz_filename = '%s/%s/%s_%s_%s_samples_%s.npz' % (
            config['samples_root'], experiment_name, config['mode'],
            'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train', k)
        print('Saving npz to %s...' % npz_filename)
        np.savez(npz_filename, **{'x': x, 'y': y})
        

        # generate samples for each group
        print('For computing intra-FID, sampling 10,000 images for each group and saving them to npz...')
        
        sample_per_group = 10000
        
        z00_samples = []
        z00_labels = []
        z00_count = 0

        z01_samples = []
        z01_labels = []
        z01_count = 0

        z10_samples = []
        z10_labels = []
        z10_count = 0

        z11_samples = []
        z11_labels = []
        z11_count = 0

        while(z00_count < sample_per_group or z01_count < sample_per_group 
              or z10_count < sample_per_group or z11_count < sample_per_group):
            x, y = [], []
            for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
                with torch.no_grad():
                    images, labels = sample()
                x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
                y += [labels.cpu().numpy()]
            x = np.concatenate(x, 0)[:config['sample_num_npz']]
            y = np.concatenate(y, 0)[:config['sample_num_npz']]

            preds = []
            n_batches = x.shape[0] // 1000
            with torch.no_grad():
                for i in range(n_batches):
                    t_samp = x[i*1000:(i+1)*1000]
                    samp = t_samp / 255.  # renormalize to feed into classifier
                    samp = torch.from_numpy(samp).to('cuda').float()

                    # get classifier predictions
                    logits, probas = clf(samp)
                    _, pred = torch.max(probas, 1)
                    preds.append(pred)
                preds = torch.cat(preds).data.cpu().numpy()

            if z00_count < sample_per_group:
                z00_samples.append(x[preds==0])
                z00_labels.append(y[preds==0])
                z00_count += len(x[preds==0])

            elif z01_count < sample_per_group:
                z01_samples.append(x[preds==1])
                z01_labels.append(y[preds==1])
                z01_count += len(x[preds==1])

            elif z10_count < sample_per_group:
                z10_samples.append(x[preds==2])
                z10_labels.append(y[preds==2])
                z10_count += len(x[preds==2])

            elif z11_count < sample_per_group:
                z11_samples.append(x[preds==3])
                z11_labels.append(y[preds==3])
                z11_count += len(x[preds==3])

        z00_samples = np.concatenate(z00_samples)[:sample_per_group]
        z00_labels = np.concatenate(z00_labels)[:sample_per_group]

        z01_samples = np.concatenate(z01_samples)[:sample_per_group]
        z01_labels = np.concatenate(z01_labels)[:sample_per_group]

        z10_samples = np.concatenate(z10_samples)[:sample_per_group]
        z10_labels = np.concatenate(z10_labels)[:sample_per_group]

        z11_samples = np.concatenate(z11_samples)[:sample_per_group]
        z11_labels = np.concatenate(z11_labels)[:sample_per_group]

        print('z00 images shape: %s, Labels shape: %s' % (z00_samples.shape, z00_labels.shape))
        print('z01 images shape: %s, Labels shape: %s' % (z01_samples.shape, z01_labels.shape))
        print('z10 images shape: %s, Labels shape: %s' % (z10_samples.shape, z10_labels.shape))
        print('z11 images shape: %s, Labels shape: %s' % (z11_samples.shape, z11_labels.shape))
        
        z00_npz_filename = '%s/%s/%s_%s_%s_z00_samples_%s.npz' % (
            config['samples_root'], experiment_name, config['mode'],
            'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train', k)
        
        z01_npz_filename = '%s/%s/%s_%s_%s_z01_samples_%s.npz' % (
            config['samples_root'], experiment_name, config['mode'],
            'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train', k)
        
        z10_npz_filename = '%s/%s/%s_%s_%s_z10_samples_%s.npz' % (
            config['samples_root'], experiment_name, config['mode'],
            'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train', k)
        
        z11_npz_filename = '%s/%s/%s_%s_%s_z11_samples_%s.npz' % (
            config['samples_root'], experiment_name, config['mode'],
            'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train', k)

        print('Saving z00 npz to %s...' % z00_npz_filename)
        print('Saving z01 npz to %s...' % z01_npz_filename)
        print('Saving z10 npz to %s...' % z10_npz_filename)
        print('Saving z11 npz to %s...' % z11_npz_filename)
        
        np.savez(z00_npz_filename, **{'x': z00_samples, 'y': z00_labels})
        np.savez(z01_npz_filename, **{'x': z01_samples, 'y': z01_labels})
        np.savez(z10_npz_filename, **{'x': z10_samples, 'y': z10_labels})
        np.savez(z11_npz_filename, **{'x': z11_samples, 'y': z11_labels})
        

    # classify proportions
    metrics = {'l2': 0, 'l1': 0, 'kl': 0}
    l2_db = np.zeros(10)
    l1_db = np.zeros(10)
    kl_db = np.zeros(10)
    FID_db = np.zeros(10) # modified by sum
    z00_FID_db = np.zeros(10) # modified by sum
    z01_FID_db = np.zeros(10) # modified by sum
    z10_FID_db = np.zeros(10) # modified by sum
    z11_FID_db = np.zeros(10) # modified by sum
    

    # output file
    fname = '%s/%s/FD_intra_FID_from_samples_%s_%s_%s.p' % (
            config['samples_root'], experiment_name, config['mode'],
            'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train')
    
    probs_fname = 'clf_probs_%s_%s_%s.npy' % (config['mode'],
            'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train')

    biased_data = '../fid_stats/fid_stats_celeba.npz'
    unbiased_data = '../fid_stats/celeba/unbiased_all_multi_fid_stats.npz'
    z00_data = '../fid_stats/celeba/unblack_female_fid_stats.npz'
    z01_data = '../fid_stats/celeba/unblack_male_fid_stats.npz'
    z10_data = '../fid_stats/celeba/black_female_fid_stats.npz'
    z11_data = '../fid_stats/celeba/black_male_fid_stats.npz'
                
    # number of classes
    probs_db = np.zeros((10, 10000, n_classes))
    for i in range(10):
        # grab appropriate samples
        npz_filename = '{}/{}/{}_{}_{}_samples_{}.npz'.format(
            config['samples_root'], experiment_name, config['mode'], 'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train', i)
        z00_npz_filename = '{}/{}/{}_{}_{}_z00_samples_{}.npz'.format(
            config['samples_root'], experiment_name, config['mode'], 'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train', i)
        z01_npz_filename = '{}/{}/{}_{}_{}_z01_samples_{}.npz'.format(
            config['samples_root'], experiment_name, config['mode'], 'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train', i)
        z10_npz_filename = '{}/{}/{}_{}_{}_z10_samples_{}.npz'.format(
            config['samples_root'], experiment_name, config['mode'], 'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train', i)
        z11_npz_filename = '{}/{}/{}_{}_{}_z11_samples_{}.npz'.format(
            config['samples_root'], experiment_name, config['mode'], 'ema' if config['use_ema'] else '',
            'G_eval' if config['G_eval_mode'] else 'G_train', i)
        preds, probs = classify_examples(clf, npz_filename)
        l2, l1, kl = utils.fairness_discrepancy(preds, clf_classes)
        # modified by sum
        FID = fid_score.calculate_fid_given_paths([unbiased_data, npz_filename], batch_size=100, cuda=True, dims=2048)
        z00_FID = fid_score.calculate_fid_given_paths([z00_data, z00_npz_filename], batch_size=100, cuda=True, dims=2048)
        z01_FID = fid_score.calculate_fid_given_paths([z01_data, z01_npz_filename], batch_size=100, cuda=True, dims=2048)
        z10_FID = fid_score.calculate_fid_given_paths([z10_data, z10_npz_filename], batch_size=100, cuda=True, dims=2048)
        z11_FID = fid_score.calculate_fid_given_paths([z11_data, z11_npz_filename], batch_size=100, cuda=True, dims=2048)
        

        # save metrics
        l2_db[i] = l2
        l1_db[i] = l1
        kl_db[i] = kl
        FID_db[i] = FID # modified by sum
        z00_FID_db[i] = z00_FID # modified by sum
        z01_FID_db[i] = z01_FID # modified by sum
        z10_FID_db[i] = z10_FID # modified by sum
        z11_FID_db[i] = z11_FID # modified by sum
        probs_db[i] = probs
        print('fair_disc and FID for iter {} is: l2:{}, l1:{}, kl:{}, FID: {}, z00_FID: {}, z01_FID: {}, z10_FID: {}, z11_FID: {}'
              .format(i, l2, l1, kl, FID, z00_FID, z01_FID, z10_FID, z11_FID)) # modified by sum
    metrics['l2'] = l2_db
    metrics['l1'] = l1_db
    metrics['kl'] = kl_db
    metrics['FID'] = FID_db # modified by sum
    metrics['z00_FID'] = z00_FID_db # modified by sum
    metrics['z01_FID'] = z01_FID_db # modified by sum
    metrics['z10_FID'] = z10_FID_db # modified by sum
    metrics['z11_FID'] = z11_FID_db # modified by sum
    print('Fairness Discrepancies and FID saved in {}'.format(fname)) # modified by sum
    print(FID_db) # modified by sum
    print(z00_FID_db) # modified by sum
    print(z01_FID_db) # modified by sum
    print(z10_FID_db) # modified by sum
    print(z11_FID_db) # modified by sum
    print(l2_db)
    
    # save all metrics
    with open(fname, 'wb') as fp:
        pickle.dump(metrics, fp)
    np.save(os.path.join(config['samples_root'], experiment_name, probs_fname), probs_db)


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
