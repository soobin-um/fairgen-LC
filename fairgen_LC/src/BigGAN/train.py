""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
import time

from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
import fid_score
from sync_batchnorm import patch_replication_callback

# The main training file. Config is a dictionary specifying the configuration
# of this training run.


def run(config):
    
    start = time.time()

    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    # config['n_classes'] = utils.nclass_dict[config['dataset']]

    # NOTE: setting n_classes to 1 except in conditional case to train as unconditional model
    config['n_classes'] = 1 
    if config['conditional']:
        config['n_classes'] = 2
    print('n classes: {}'.format(config['n_classes']))
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    
    device = torch.device('cuda:%s' %(config['GPU_main']))
    torch.cuda.set_device(device)

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True
    
    if config['reproducible']:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        #     torch.use_deterministic_algorithms(True)

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)
    
    writer = SummaryWriter("%s/runs/%s" % (config['base_root'], experiment_name))

    # Next, build the model
    G = model.Generator(**config).to(device)
    D = model.Discriminator(**config).to(device)
    D_v = model.Discriminator_v(**config).to(device)
    ema_losses = utils.ema_losses(decay=config['ema_losses_decay'], start_itr=config['ema_losses_start'])

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(
            config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init': True,
                                   'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None

    # FP16?
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
        # Consider automatically reducing SN_eps?
    GD = model.G_D(G, D, 0)  # check if labels are 0's if "unconditional"
    GD_v = model.G_D(G, D_v, 0)  # check if labels are 0's if "unconditional"
    print(G)
    print(D)
    print(D_v)
    print('Number of params in G: {} D: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]]))
    
    # Prepare state dict, which holds things like epoch # and itr #
    if not config['multi']:
        state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num_fair': 0, 'save_best_num_fid': 0, 'save_best_num_tradeoff': 0,
                      'save_best_num_intra_tradeoff': 0, 'best_IS': 0, 'best_FID': 999999, 'best_z0_FID': 999999, 'best_z1_FID': 999999,
                      'best_fair_d': 999999, 'best_tradeoff': 999999, 'best_intra_tradeoff': 999999, 'config': config,
                      'ema_losses_D_real': 1000., 'ema_losses_D_fake': 1000.}
    else:
        state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num_fair': 0, 'save_best_num_fid': 0, 'save_best_num_tradeoff': 0,
                      'save_best_num_intra_tradeoff': 0, 'best_IS': 0, 'best_FID': 999999, 'best_z00_FID': 999999, 'best_z01_FID': 999999,
                      'best_z10_FID': 999999, 'best_z11_FID': 999999, 'best_fair_d': 999999, 'best_tradeoff': 999999,
                      'best_intra_tradeoff': 999999, 'config': config, 'ema_losses_D_real': 1000., 'ema_losses_D_fake': 1000.}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(G, D, D_v, state_dict,
                           config['weights_root'], experiment_name,
                           config['load_weights'] if config['load_weights'] else None,
                           G_ema if config['ema'] else None)
        utils.load_ema_losses_from_state_dict(ema_losses, state_dict)

    # If parallel, parallelize the GD module
    if config['parallel']:
        GD = nn.DataParallel(GD, list(range(config['GPU_main'], config['GPU_main'] + config['nGPU'])))
        GD_v = nn.DataParallel(GD_v, list(range(config['GPU_main'], config['GPU_main'] + config['nGPU'])))
        if config['cross_replica']:
            patch_replication_callback(GD)
            patch_replication_callback(GD_v)

    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    test_metrics_fname = '%s/%s_log.json' % (config['logs_root'],
                                              experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = utils.MetricsLogger(test_metrics_fname,
                                   reinitialize=(not config['resume']))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname,
                               reinitialize=(not config['resume']),
                               logstyle=config['logstyle'])
    # Write metadata
    utils.write_metadata(config['logs_root'],
                         experiment_name, config, state_dict)
    # Prepare data; the Discriminator's batch size is all that needs to be passed
    # to the dataloader, as G doesn't require dataloading.
    # Note that at every loader iteration we pass in enough data to complete
    # a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps']
                    * config['num_D_accumulations'])
    D_v_batch_size = (config['v_batch_size'] * config['num_D_v_steps']
                    * config['num_D_v_accumulations'])
    
    loaders = utils.get_data_loaders(0, config, **{**config, 'batch_size': D_batch_size,
        'start_itr': state_dict['itr']})
    
    valid_loaders = utils.get_data_loaders(1, config, **{**config, 'batch_size': D_v_batch_size,
        'start_itr': state_dict['itr']})

    # Prepare inception metrics: FID and IS
    get_inception_metrics = inception_utils.prepare_inception_metrics(
        config['dataset'], config['parallel'], config['no_fid'])

    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'], config['v_batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'], true_prop=config['true_prop'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                         config['n_classes'], device=device,
                                         fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()
    
    # NOTE: "unconditional" GAN
    if not config['conditional']:
        fixed_y.zero_()
        y_.zero_()
    
    # Loaders are loaded, prepare the training function
    if config['which_train_fn'] == 'GAN':
        train = train_fns.GAN_training_function(G, D, GD, D_v, GD_v, z_, y_, 
                                                ema, state_dict, config, writer, ema_losses)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()
    # Prepare Sample function for use with inception metrics
    sample = functools.partial(utils.sample,
                               G=(G_ema if config['ema'] and config['use_ema']
                                   else G),
                               z_=z_, y_=y_, config=config)
    
    print('Beginning training at epoch %d...' % state_dict['epoch'])
    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        # Which progressbar to use? TQDM or my own?
        if config['pbar'] == 'mine':
            pbar = utils.progress(
                loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loaders[0])


        # iterate through the dataloaders
        for i, (x, y) in enumerate(pbar):
            x_v, y_v = next(iter(valid_loaders[0]))
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            D_v.train()
            if config['ema']:
                G_ema.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device)
                x_v, y_v = x_v.to(device).half(), y_v.to(device)
            else:
                x, y = x.to(device), y.to(device)
                x_v, y_v = x_v.to(device), y_v.to(device)
                
            metrics = train(x, y, x_v, y_v)
            writer.flush()
            train_log.log(itr=int(state_dict['itr']), **metrics)

            # Every sv_log_interval, log singular values
            if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
                train_log.log(itr=int(state_dict['itr']),
                              **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

            # If using my progbar, print metrics.
            #if config['pbar'] == 'mine':
            #    print(', '.join(['itr: %d' % state_dict['itr']]
            #                    + ['%s : %+4.3f' % (key, metrics[key])
            #                        for key in metrics]), end=' ')

            # Save weights and copies as configured at specified interval
            if not (state_dict['itr'] % config['save_every']):
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                utils.save_ema_losses_to_state_dict(ema_losses, state_dict)
                train_fns.save_and_sample(G, D, D_v, G_ema, z_, y_, fixed_z, fixed_y,
                                          state_dict, config, experiment_name)
            
            # Test at specified interval
            if (epoch >= config['start_eval']) and (state_dict['itr'] % config['test_every']) == 0:
                utils.save_ema_losses_to_state_dict(ema_losses, state_dict)
                if config['intra_FID']:
                    if config['multi']:
                        if not config['FID_off']:
                            unbiased_all_data_moments = '../fid_stats/celeba/unbiased_all_multi_fid_stats.npz'
                        z00_data_moments = '../fid_stats/celeba/unblack_female_fid_stats.npz'
                        z01_data_moments = '../fid_stats/celeba/unblack_male_fid_stats.npz'
                        z10_data_moments = '../fid_stats/celeba/black_female_fid_stats.npz'
                        z11_data_moments = '../fid_stats/celeba/black_male_fid_stats.npz'

                        # load appropriate moments
                        if not config['FID_off']:
                            print('Loaded all data moments at: {}'.format(unbiased_all_data_moments))
                        print('Loaded data moments for unblack-haired female at: {}'.format(z00_data_moments))
                        print('Loaded data moments for unblack-haired male at: {}'.format(z01_data_moments))
                        print('Loaded data moments for black-haired female at: {}'.format(z10_data_moments))
                        print('Loaded data moments for black-haired male at: {}'.format(z11_data_moments))
                        experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))
                        
                        # eval mode for FID computation
                        if config['G_eval_mode']:
                            print('Switching G to eval mode...')
                            G.eval()
                            if config['ema']:
                                G_ema.eval()
                                
                        utils.sample_inception(
                            G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))
                        utils.sample_inception_intra_multi(
                            G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))

                        # Get saved sample path
                        folder_number = str(epoch)
                        if not config['FID_off']:
                            all_sample_moments = '%s/%s/%s/samples.npz' % (config['samples_root'], experiment_name, folder_number)
                        z00_sample_moments = '%s/%s/%s/z00_samples.npz' % (config['samples_root'], experiment_name, folder_number)
                        z01_sample_moments = '%s/%s/%s/z01_samples.npz' % (config['samples_root'], experiment_name, folder_number)
                        z10_sample_moments = '%s/%s/%s/z10_samples.npz' % (config['samples_root'], experiment_name, folder_number)
                        z11_sample_moments = '%s/%s/%s/z11_samples.npz' % (config['samples_root'], experiment_name, folder_number)

                        # Calculate FID
                        if not config['FID_off']:
                            FID = fid_score.calculate_fid_given_paths([unbiased_all_data_moments, all_sample_moments],
                                                                      batch_size=100, cuda=True, dims=2048)
                        else:
                            FID = 999999
                            
                        z00_FID = fid_score.calculate_fid_given_paths([z00_data_moments, z00_sample_moments],
                                                                     batch_size=100, cuda=True, dims=2048)
                        z01_FID = fid_score.calculate_fid_given_paths([z01_data_moments, z01_sample_moments],
                                                                     batch_size=100, cuda=True, dims=2048)
                        z10_FID = fid_score.calculate_fid_given_paths([z10_data_moments, z10_sample_moments],
                                                                     batch_size=100, cuda=True, dims=2048)
                        z11_FID = fid_score.calculate_fid_given_paths([z11_data_moments, z11_sample_moments],
                                                                     batch_size=100, cuda=True, dims=2048)
                        print("FID calculated")
                        train_fns.update_intra_FID_multi(G, D, D_v, G_ema, state_dict, config, FID, z00_FID, z01_FID, z10_FID, z11_FID,
                                                         experiment_name, test_log, epoch, writer)  # added epoch logging
                        writer.flush()
                    else:    
                        if config['dataset'] == 'CA64':
                            if not config['FID_off']:
                                unbiased_all_data_moments = '../fid_stats/celeba/unbiased_all_gender_fid_stats.npz'
                            z1_data_moments = '../fid_stats/celeba/male_fid_stats.npz'
                            z0_data_moments = '../fid_stats/celeba/female_fid_stats.npz'
                            
                        elif config['dataset'] == 'UTKFace':
                            if not config['FID_off']:
                                unbiased_all_data_moments = '../fid_stats/UTKFace/unbiased_all_race_fid_stats.npz'
                            z1_data_moments = '../fid_stats/UTKFace/non_white_fid_stats.npz'
                            z0_data_moments = '../fid_stats/UTKFace/white_fid_stats.npz'
                            
                        elif config['dataset'] == 'FairFace':
                            if not config['FID_off']:
                                unbiased_all_data_moments = '../fid_stats/FairFace/unbiased_all_race_fid_stats.npz'
                            z1_data_moments = '../fid_stats/FairFace/black_fid_stats.npz'
                            z0_data_moments = '../fid_stats/FairFace/white_fid_stats.npz'

                        # load appropriate moments
                        if not config['FID_off']:
                            print('Loaded all data moments at: {}'.format(unbiased_all_data_moments))
                        print('Loaded data moments for z1 at: {}'.format(z1_data_moments))
                        print('Loaded data moments for z0 at: {}'.format(z0_data_moments))
                        experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))

                        # eval mode for FID computation
                        if config['G_eval_mode']:
                            print('Switching G to eval mode...')
                            G.eval()
                            if config['ema']:
                                G_ema.eval()
                        utils.sample_inception(
                            G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))
                        utils.sample_inception_intra(
                            G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))

                        # Get saved sample path
                        folder_number = str(epoch)
                        if not config['FID_off']:
                            all_sample_moments = '%s/%s/%s/samples.npz' % (config['samples_root'], experiment_name, folder_number)
                        z1_sample_moments = '%s/%s/%s/z1_samples.npz' % (config['samples_root'], experiment_name, folder_number)
                        z0_sample_moments = '%s/%s/%s/z0_samples.npz' % (config['samples_root'], experiment_name, folder_number)

                        # Calculate FID
                        if not config['FID_off']:
                            FID = fid_score.calculate_fid_given_paths([unbiased_all_data_moments, all_sample_moments],
                                                                      batch_size=100, cuda=True, dims=2048)
                        else:
                            FID = 999999
                            
                        z1_FID = fid_score.calculate_fid_given_paths([z1_data_moments, z1_sample_moments],
                                                                     batch_size=100, cuda=True, dims=2048)
                        z0_FID = fid_score.calculate_fid_given_paths([z0_data_moments, z0_sample_moments],
                                                                     batch_size=100, cuda=True, dims=2048)
                        print("FID calculated")
                        train_fns.update_intra_FID(G, D, D_v, G_ema, state_dict, config, FID, z1_FID, z0_FID,
                                                   experiment_name, test_log, epoch, writer)  # added epoch logging
                        
                        writer.flush()
                    
                else:
                    # Find correct inception moments
                    if config['dataset'] == 'CA64':
                        if config['biased_FID']:
                            data_moments = '../fid_stats/celeba/biased_perc{}_all_gender_fid_stats.npz'.format(config['perc'])
                            if config['multi']:
                                data_moments = '../fid_stats/celeba/biased_perc{}_all_multi_fid_stats.npz'.format(config['perc'])
                                fid_type = 'multi'
                            else:
                                fid_type = 'gender'

                        else:
                            data_moments = '../fid_stats/celeba/unbiased_all_gender_fid_stats.npz'
                            if config['multi']:
                                data_moments = '../fid_stats/celeba/unbiased_all_multi_fid_stats.npz'
                                fid_type = 'multi'
                            else:
                                fid_type = 'gender'
                    elif config['dataset'] == 'UTKFace':
                        if config['biased_FID']:
                            data_moments = '../fid_stats/UTKFace/biased_perc{}_all_race_fid_stats.npz'.format(config['perc'])
                        else:
                            data_moments = '../fid_stats/UTKFace/unbiased_all_race_fid_stats.npz'
                        fid_type = 'race'

                    elif config['dataset'] == 'FairFace':
                        if config['biased_FID']:
                            data_moments = '../fid_stats/FairFace/biased_perc{}_all_race_fid_stats.npz'.format(config['perc'])
                        else:
                            data_moments = '../fid_stats/FairFace/unbiased_all_race_fid_stats.npz'
                        fid_type = 'race'

                    # load appropriate moments
                    print('Loaded data moments at: {}'.format(data_moments))
                    experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))

                    # eval mode for FID computation
                    if config['G_eval_mode']:
                        print('Switching G to eval mode...')
                        G.eval()
                        if config['ema']:
                            G_ema.eval()
                    utils.sample_inception(
                        G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))
                    # Get saved sample path
                    folder_number = str(epoch)
                    sample_moments = '%s/%s/%s/samples.npz' % (config['samples_root'], experiment_name, folder_number)
                    # Calculate FID
                    FID = fid_score.calculate_fid_given_paths([data_moments, sample_moments], batch_size=100, cuda=True, dims=2048)
                    print("FID calculated")
                    train_fns.update_FID(G, D, D_v, G_ema, state_dict, config, 
                                         FID, experiment_name, test_log, epoch, writer)  # added epoch logging
                    writer.flush()
                
        # Test every epoch
        if (epoch >= config['start_eval']):
            utils.save_ema_losses_to_state_dict(ema_losses, state_dict)
            if config['intra_FID']:
                if config['multi']:
                    if not config['FID_off']:
                        unbiased_all_data_moments = '../fid_stats/celeba/unbiased_all_multi_fid_stats.npz'
                    z00_data_moments = '../fid_stats/celeba/unblack_female_fid_stats.npz'
                    z01_data_moments = '../fid_stats/celeba/unblack_male_fid_stats.npz'
                    z10_data_moments = '../fid_stats/celeba/black_female_fid_stats.npz'
                    z11_data_moments = '../fid_stats/celeba/black_male_fid_stats.npz'

                    # load appropriate moments
                    if not config['FID_off']:
                        print('Loaded all data moments at: {}'.format(unbiased_all_data_moments))
                    print('Loaded data moments for unblack-haired female at: {}'.format(z00_data_moments))
                    print('Loaded data moments for unblack-haired male at: {}'.format(z01_data_moments))
                    print('Loaded data moments for black-haired female at: {}'.format(z10_data_moments))
                    print('Loaded data moments for black-haired male at: {}'.format(z11_data_moments))
                    experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))

                    # eval mode for FID computation
                    if config['G_eval_mode']:
                        print('Switching G to eval mode...')
                        G.eval()
                        if config['ema']:
                            G_ema.eval()
                    utils.sample_inception(
                        G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))
                    utils.sample_inception_intra_multi(
                        G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))

                    # Get saved sample path
                    folder_number = str(epoch)
                    if not config['FID_off']:
                        all_sample_moments = '%s/%s/%s/samples.npz' % (config['samples_root'], experiment_name, folder_number)
                    z00_sample_moments = '%s/%s/%s/z00_samples.npz' % (config['samples_root'], experiment_name, folder_number)
                    z01_sample_moments = '%s/%s/%s/z01_samples.npz' % (config['samples_root'], experiment_name, folder_number)
                    z10_sample_moments = '%s/%s/%s/z10_samples.npz' % (config['samples_root'], experiment_name, folder_number)
                    z11_sample_moments = '%s/%s/%s/z11_samples.npz' % (config['samples_root'], experiment_name, folder_number)

                    # Calculate FID
                    if not config['FID_off']:
                        FID = fid_score.calculate_fid_given_paths([unbiased_all_data_moments, all_sample_moments],
                                                                  batch_size=100, cuda=True, dims=2048)
                    else:
                        FID = 999999
                        
                    z00_FID = fid_score.calculate_fid_given_paths([z00_data_moments, z00_sample_moments],
                                                                 batch_size=100, cuda=True, dims=2048)
                    z01_FID = fid_score.calculate_fid_given_paths([z01_data_moments, z01_sample_moments],
                                                                 batch_size=100, cuda=True, dims=2048)
                    z10_FID = fid_score.calculate_fid_given_paths([z10_data_moments, z10_sample_moments],
                                                                 batch_size=100, cuda=True, dims=2048)
                    z11_FID = fid_score.calculate_fid_given_paths([z11_data_moments, z11_sample_moments],
                                                                 batch_size=100, cuda=True, dims=2048)
                    print("FID calculated")
                    train_fns.update_intra_FID_multi(G, D, D_v, G_ema, state_dict, config, FID, z00_FID, z01_FID, z10_FID, z11_FID,
                                                     experiment_name, test_log, epoch, writer)  # added epoch logging
                    writer.flush()
                    
                else:
                    if config['dataset'] == 'CA64':
                        if not config['FID_off']:
                            unbiased_all_data_moments = '../fid_stats/celeba/unbiased_all_gender_fid_stats.npz'
                        z1_data_moments = '../fid_stats/celeba/male_fid_stats.npz'
                        z0_data_moments = '../fid_stats/celeba/female_fid_stats.npz'
                        
                    elif config['dataset'] == 'UTKFace':
                        if not config['FID_off']:
                            unbiased_all_data_moments = '../fid_stats/UTKFace/unbiased_all_race_fid_stats.npz'
                        z1_data_moments = '../fid_stats/UTKFace/non_white_fid_stats.npz'
                        z0_data_moments = '../fid_stats/UTKFace/white_fid_stats.npz'

                    elif config['dataset'] == 'FairFace':
                        if not config['FID_off']:
                            unbiased_all_data_moments = '../fid_stats/FairFace/unbiased_all_race_fid_stats.npz'
                        z1_data_moments = '../fid_stats/FairFace/black_fid_stats.npz'
                        z0_data_moments = '../fid_stats/FairFace/white_fid_stats.npz'

                    # load appropriate moments
                    if not config['FID_off']:
                        print('Loaded all data moments at: {}'.format(unbiased_all_data_moments))
                    print('Loaded data moments for z1 at: {}'.format(z1_data_moments))
                    print('Loaded data moments for z0 at: {}'.format(z0_data_moments))
                    experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))

                    # eval mode for FID computation
                    if config['G_eval_mode']:
                        print('Switching G to eval mode...')
                        G.eval()
                        if config['ema']:
                            G_ema.eval()
                    utils.sample_inception(
                        G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))
                    utils.sample_inception_intra(
                        G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))

                    # Get saved sample path
                    folder_number = str(epoch)
                    if not config['FID_off']:
                        all_sample_moments = '%s/%s/%s/samples.npz' % (config['samples_root'], experiment_name, folder_number)
                    z1_sample_moments = '%s/%s/%s/z1_samples.npz' % (config['samples_root'], experiment_name, folder_number)
                    z0_sample_moments = '%s/%s/%s/z0_samples.npz' % (config['samples_root'], experiment_name, folder_number)

                    # Calculate FID
                    if not config['FID_off']:
                        FID = fid_score.calculate_fid_given_paths([unbiased_all_data_moments, all_sample_moments],
                                                                  batch_size=100, cuda=True, dims=2048)
                    else:
                        FID = 999999
                    z1_FID = fid_score.calculate_fid_given_paths([z1_data_moments, z1_sample_moments],
                                                                 batch_size=100, cuda=True, dims=2048)
                    z0_FID = fid_score.calculate_fid_given_paths([z0_data_moments, z0_sample_moments],
                                                                 batch_size=100, cuda=True, dims=2048)
                    print("FID calculated")
                    train_fns.update_intra_FID(G, D, D_v, G_ema, state_dict, config, FID, z1_FID, z0_FID,
                                               experiment_name, test_log, epoch, writer)  # added epoch logging
                    writer.flush()

            else: # Calculate overall FID
                # Find correct inception moments
                if config['dataset'] == 'CA64':
                    if config['biased_FID']:
                        data_moments = '../fid_stats/celeba/biased_perc{}_all_gender_fid_stats.npz'.format(config['perc'])
                        if config['multi']:
                            data_moments = '../fid_stats/celeba/biased_perc{}_all_multi_fid_stats.npz'.format(config['perc'])
                            fid_type = 'multi'
                        else:
                            fid_type = 'gender'

                    else:
                        data_moments = '../fid_stats/celeba/unbiased_all_gender_fid_stats.npz'
                        if config['multi']:
                            data_moments = '../fid_stats/celeba/unbiased_all_multi_fid_stats.npz'
                            fid_type = 'multi'
                        else:
                            fid_type = 'gender'
                elif config['dataset'] == 'UTKFace':
                    if config['biased_FID']:
                        data_moments = '../fid_stats/UTKFace/biased_perc{}_all_race_fid_stats.npz'.format(config['perc'])
                    else:
                        data_moments = '../fid_stats/UTKFace/unbiased_all_race_fid_stats.npz'
                    fid_type = 'race'

                elif config['dataset'] == 'FairFace':
                    if config['biased_FID']:
                        data_moments = '../fid_stats/FairFace/biased_perc{}_all_race_fid_stats.npz'.format(config['perc'])
                    else:
                        data_moments = '../fid_stats/FairFace/unbiased_all_race_fid_stats.npz'
                    fid_type = 'race'

                # load appropriate moments
                print('Loaded data moments at: {}'.format(data_moments))
                experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))

                # eval mode for FID computation
                if config['G_eval_mode']:
                    print('Switching G to eval mode...')
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                utils.sample_inception(
                    G_ema if config['ema'] and config['use_ema'] else G, config, str(epoch))
                # Get saved sample path
                folder_number = str(epoch)
                sample_moments = '%s/%s/%s/samples.npz' % (config['samples_root'], experiment_name, folder_number)
                # Calculate FID
                FID = fid_score.calculate_fid_given_paths([data_moments, sample_moments], batch_size=100, cuda=True, dims=2048)
                print("FID calculated")
                train_fns.update_FID(G, D, D_v, G_ema, state_dict, config, FID, experiment_name, test_log, epoch, writer)  # added epoch logging
                writer.flush()
            
        # Increment epoch counter at end of epoch
        print('Completed epoch {}, time: {}'.format(epoch, time.time()-start))
        state_dict['epoch'] += 1
    writer.close()


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
