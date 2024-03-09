''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import utils
import losses

from torch.utils.tensorboard import SummaryWriter
from clf_models import ResNet18, BasicBlock

# NOTE: this is only for the binary attribute classifier!
CLF_PATH_celeba = '../src/results/celeba/attr_clf/model_best.pth.tar'
MULTI_CLF_PATH_celeba = '../src/results/celeba/multi_clf/model_best.pth.tar'
CLF_PATH_UTKFace = '../src/results/UTKFace/attr_clf/model_best.pth.tar'
CLF_PATH_FairFace = '../src/results/FairFace/attr_clf/model_best.pth.tar'


# Dummy training function for debugging
def dummy_training_function():
    def train(x, y, ratio):
        return {}
    return train


def select_loss(config):
    if config['loss_type'] == 'hinge_dis_linear_gen':
        return losses.loss_hinge_dis, losses.loss_linear_gen
    elif config['loss_type'] == 'hinge_dis_hinge_gen':
        return losses.loss_hinge_dis, losses.loss_hinge_gen
    elif config['loss_type'] == 'dcgan':
        return losses.loss_dcgan_dis, losses.loss_dcgan_gen
    elif 'wgan' in config['loss_type']:
        return losses.loss_wgan_dis, losses.loss_wgan_gen
    elif config['loss_type'] == 'ls':
        return losses.loss_ls_dis, losses.loss_ls_gen
    elif config['loss_type'] == 'kl':
        return losses.loss_kl_dis, losses.loss_kl_gen
    elif config['loss_type'] == 'kl_gen':
        return losses.loss_hinge_dis, losses.loss_kl_gen
    elif config['loss_type'] == 'kl_dis':
        return losses.loss_kl_dis, losses.loss_hinge_gen
    elif config['loss_type'] == 'kl_grad':
        return losses.loss_kl_grad_dis, losses.loss_kl_grad_gen
    elif config['loss_type'] == 'f_kl':
        return losses.loss_f_kl_dis, losses.loss_f_kl_gen
    elif config['loss_type'] == 'chi2':
        return losses.loss_chi_dis, losses.loss_chi_gen
    elif config['loss_type'] == 'dv':
        return losses.loss_dv_dis, losses.loss_dv_gen
    else:
        raise ValueError('loss not defined')
        
def select_v_loss(config):
    if config['v_loss_type'] == 'hinge_dis_linear_gen':
        return losses.loss_hinge_dis, losses.loss_linear_gen
    elif config['v_loss_type'] == 'hinge_dis_hinge_gen':
        return losses.loss_hinge_dis, losses.loss_hinge_gen
    elif config['v_loss_type'] == 'dcgan':
        return losses.loss_dcgan_dis, losses.loss_dcgan_gen
    elif 'wgan' in config['v_loss_type']:
        return losses.loss_wgan_dis, losses.loss_wgan_gen
    elif config['v_loss_type'] == 'ls':
        return losses.loss_ls_dis, losses.loss_ls_gen
    elif config['v_loss_type'] == 'kl':
        return losses.loss_kl_dis, losses.loss_kl_gen
    elif config['v_loss_type'] == 'kl_gen':
        return losses.loss_hinge_dis, losses.loss_kl_gen
    elif config['v_loss_type'] == 'kl_dis':
        return losses.loss_kl_dis, losses.loss_hinge_gen
    elif config['v_loss_type'] == 'kl_grad':
        return losses.loss_kl_grad_dis, losses.loss_kl_grad_gen
    elif config['v_loss_type'] == 'f_kl':
        return losses.loss_f_kl_dis, losses.loss_f_kl_gen
    elif config['v_loss_type'] == 'chi2':
        return losses.loss_chi_dis, losses.loss_chi_gen
    elif config['v_loss_type'] == 'dv':
        return losses.loss_dv_dis, losses.loss_dv_gen
    else:
        raise ValueError('loss not defined')
        
def compute_gradient_penalty(GD, z, gy, x, dy, config):
    """Calculates the gradient penalty loss for WGAN GP"""
    
    # Generate fake_samples from GD module
    _, G_z = GD(z, gy, train_G=False, return_G_z=True, split_D=config['split_D'])
    
    # Random weight term for interpolation between real and fake samples
    alpha = torch.as_tensor(np.random.random((x.size(0), 1, 1, 1)), dtype=torch.float, device=x.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * x + ((1 - alpha) * G_z)).requires_grad_(True)
    _, d_interpolates = GD(z, gy, interpolates, dy, train_G=False, split_D=config['split_D'])
    fake = torch.autograd.Variable(torch.ones([x.shape[0], 1], dtype=torch.float, device=x.device).fill_(1.0),
                                   requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
        
def GAN_training_function(G, D, GD, D_v, GD_v, z_, y_, ema, state_dict, config, writer, ema_losses=None):
    discriminator_loss, generator_loss = select_loss(config)
    discriminator_v_loss, generator_v_loss = select_v_loss(config)
    
    def train(x, y, x_v, y_v):
        G.optim.zero_grad()
        D.optim.zero_grad()
        D_v.optim.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        
        x_v = torch.split(x_v, config['v_batch_size'])
        y_v = torch.split(y_v, config['v_batch_size'])
        
        D_counter = 0
        D_v_counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(D_v, False)
            utils.toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for accumulation_index in range(config['num_D_accumulations']):
                z_.sample_()
                # only feed in 0's for y if "unconditional"
                if not config['conditional']:
                    y_.zero_()
                    y_counter = torch.zeros_like(y[D_counter]).to(y_.device).long()
                else:
                    y_.sample_()
                    y_counter = y[D_counter]
                D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']], x[D_counter], y_counter, train_G=False,
                    split_D=config['split_D'])

                if 'wgan' in config['loss_type']:
                    D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
                    if config['loss_type'] == 'wgan_gp':
                        gradient_penalty = compute_gradient_penalty(GD, z_[:config['batch_size']],
                                                                    y_[:config['batch_size']], x[D_counter], y_counter, config)
                        D_loss = (1 - config['lambda_']) * (D_loss_real + D_loss_fake + config['lambda_gp'] * gradient_penalty) / \
                                 float(config['num_D_accumulations'])
                    else:
                        D_loss = (1 - config['lambda_']) * (D_loss_real + D_loss_fake) / \
                                 float(config['num_D_accumulations'])
                
                else:
                    D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
                    D_loss = (1 - config['lambda_']) * (D_loss_real + D_loss_fake) / \
                              float(config['num_D_accumulations'])

                D_loss.backward()
                D_counter += 1
                
            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()
            
            # To check D*(x)
            with torch.no_grad():
                z_.sample_()
                if not config['conditional']:
                    y_.zero_()
                    y_counter = torch.zeros_like(y[0]).to(y_.device).long()
                else:
                    y_.sample_()
                    y_counter = y[0]
                D_fake_print, D_real_print = GD(z_[:config['batch_size']],
                                                y_[:config['batch_size']], x[0], y_counter, train_G=False,
                                                split_D=config['split_D'])
                
        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(D_v, True)
            utils.toggle_grad(G, False)

        for step_index in range(config['num_D_v_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D_v.optim.zero_grad()
            for accumulation_index in range(config['num_D_v_accumulations']):
                z_.sample_()
                # only feed in 0's for y if "unconditional"
                if not config['conditional']:
                    y_.zero_()
                    y_counter = torch.zeros_like(y_v[D_v_counter]).to(y_.device).long()
                else:
                    y_.sample_()
                    y_counter = y_v[D_v_counter]
                D_v_fake, D_v_real = GD_v(z_[:config['v_batch_size']], y_[:config['v_batch_size']], x_v[D_v_counter], y_counter, train_G=False,
                    split_D=config['split_D'])

                if 'hinge' in config['v_loss_type']:
                    D_v_loss_real, D_v_loss_fake = discriminator_v_loss(D_v_fake, D_v_real, ema_losses, state_dict['itr'])
                    # consistency along time dimension
                    if config['LC'] > 0 and state_dict['itr'] > ema_losses.start_itr:
                        if config['adaptive_LC']:
                            adaptive_weight = 1 / (2 * (config['LC_mu'] + 0.5 * (ema_losses.D_real - ema_losses.D_fake) ) )
                            D_v_loss_LC = losses.lecam_reg(D_v_real, D_v_fake, ema_losses, config) * adaptive_weight
                        else:
                            D_v_loss_LC = losses.lecam_reg(D_v_real, D_v_fake, ema_losses, config) * config['LC']
                    else:
                        D_v_loss_LC = torch.tensor(0.)
                    D_v_loss = config['lambda_'] * (D_v_loss_real + D_v_loss_fake + D_v_loss_LC) / \
                               float(config['num_D_v_accumulations'])
                    
                elif 'wgan' in config['v_loss_type']:
                    D_v_loss_real, D_v_loss_fake = discriminator_v_loss(D_v_fake, D_v_real, ema_losses, state_dict['itr'])
                    if config['v_loss_type'] == 'wgan_gp':
                        gradient_penalty_v = compute_gradient_penalty(GD_v, z_[:config['v_batch_size']],
                                                                      y_[:config['v_batch_size']], x_v[D_v_counter], y_counter, config)
                        D_v_loss = config['lambda_'] * (D_v_loss_real + D_v_loss_fake + config['lambda_gp'] * gradient_penalty_v) / \
                                   float(config['num_D_v_accumulations'])
                    else:
                        if config['LC'] > 0 and state_dict['itr'] > ema_losses.start_itr:
                            if config['adaptive_LC']:
                                adaptive_weight = 1 / (2 * (config['LC_mu'] + ema_losses.D_real))
                                D_v_loss_LC = losses.lecam_reg(D_v_real, D_v_fake, ema_losses, config) * adaptive_weight
                            else:
                                D_v_loss_LC = losses.lecam_reg(D_v_real, D_v_fake, ema_losses, config) * config['LC']
                        else:
                            D_v_loss_LC = torch.tensor(0.)
                        D_v_loss = config['lambda_'] * (D_v_loss_real + D_v_loss_fake + D_v_loss_LC) / \
                                   float(config['num_D_v_accumulations'])
                
                else:
                    D_v_loss_real, D_v_loss_fake = discriminator_v_loss(D_v_fake, D_v_real)
                    D_v_loss = config['lambda_'] * (D_v_loss_real + D_v_loss_fake) / \
                               float(config['num_D_v_accumulations'])

                D_v_loss.backward()
                D_v_counter += 1

            # Optionally apply ortho reg in D
            if config['D_v_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D_v')
                utils.ortho(D_v, config['D_v_ortho'])

            D_v.optim.step()
            
            # To check D_fair*(x)
            with torch.no_grad():
                z_.sample_()
                if not config['conditional']:
                    y_.zero_()
                    y_counter = torch.zeros_like(y_v[0]).to(y_.device).long()
                else:
                    y_.sample_()
                    y_counter = y_v[0]
                D_v_fake_print, D_v_real_print = GD_v(z_[:config['v_batch_size']], y_[:config['v_batch_size']],
                                                      x_v[0], y_counter, train_G=False, split_D=config['split_D'])
           
        ###############################################################################################
        ###############################################################################################
        
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(D_v, False)
            utils.toggle_grad(G, True)
            
        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_G_accumulations']):
            z_.sample_()
            y_.sample_()
            # NOTE: setting all labels to 0 to train as unconditional model
            if not config['conditional']:
                y_.zero_()
            D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
            D_v_fake = GD_v(z_, y_, train_G=True, split_D=config['split_D'])
            # we don't need to do anything for the generator loss
            G_loss = ( (1 - config['lambda_']) * generator_loss(D_fake)  + config['lambda_'] * generator_v_loss(
                D_v_fake) ) / float(config['num_G_accumulations'])
            G_loss.backward()

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            # Debug print to indicate we're using ortho reg in G
            print('using modified ortho reg in G')
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])
            
        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item()),
               'D_v_loss_real': float(D_v_loss_real.item()),
               'D_v_loss_fake': float(D_v_loss_fake.item()),
               'D_real': float(D_real_print.mean().item()),
               'D_fake': float(D_fake_print.mean().item()),
               'D_v_real': float(D_v_real_print.mean().item()),
               'D_v_fake': float(D_v_fake_print.mean().item())}
        # Return G's loss and the components of D's loss.
        
        # leave logs via tensorboard
        writer.add_scalar("Loss/G_loss", G_loss.item(), state_dict['itr'])
        writer.add_scalar("Loss/D_loss_real", D_loss_real.item(), state_dict['itr'])
        writer.add_scalar("Loss/D_loss_fake", D_loss_fake.item(), state_dict['itr'])
        writer.add_scalar("Loss/D_v_loss_real", D_v_loss_real.item(), state_dict['itr'])
        writer.add_scalar("Loss/D_v_loss_fake", D_v_loss_fake.item(), state_dict['itr'])
        writer.add_scalar("Output/D_real", D_real_print.mean().item(), state_dict['itr'])
        writer.add_scalar("Output/D_fake", D_fake_print.mean().item(), state_dict['itr'])
        writer.add_scalar("Output/D_v_real", D_v_real_print.mean().item(), state_dict['itr'])
        writer.add_scalar("Output/D_v_fake", D_v_fake_print.mean().item(), state_dict['itr'])
        
        return out
    return train


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''


def save_and_sample(G, D, D_v, G_ema, z_, y_, fixed_z, fixed_y,
                    state_dict, config, experiment_name):
    utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                       experiment_name, None, G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name,
                           'copy%d' % state_dict['save_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_num'] = (
            state_dict['save_num'] + 1) % config['num_save_copies']

    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    # Accumulate standing statistics?
    if config['accumulate_stats']:
    # NOTE: setting all labels to 0 to train as unconditional model
        if not config['conditional']:
            y_.zero_()
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])

    # Save a random sample sheet with fixed z and y
    with torch.no_grad():
        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(
                which_G, (fixed_z, which_G.shared(fixed_y)),
                list(range(config['GPU_main'], config['GPU_main'] + config['nGPU'])))
        else:
            fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    utils.sample_sheet(which_G,
                       classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                       num_classes=config['n_classes'],
                       samples_per_class=10, parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       config=config,
                       z_=z_)
    # Also save interp sheets
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        utils.interp_sheet(which_G,
                           num_per_sheet=16,
                           num_midpoints=8,
                           num_classes=config['n_classes'],
                           parallel=config['parallel'],
                           samples_root=config['samples_root'],
                           experiment_name=experiment_name,
                           folder_number=state_dict['itr'],
                           config=config,
                           sheet_number=0,
                           fix_z=fix_z, fix_y=fix_y, device='cuda')

''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''
def update_FID(G, D, D_v, G_ema, state_dict, config, FID, experiment_name, test_log, epoch, writer):
    if not config['multi']:
        print('Pre-loading pre-trained single-attribute classifier...')
        if config['dataset'] == 'CA64':
            clf_state_dict = torch.load(CLF_PATH_celeba)['state_dict']
        elif config['dataset'] == 'UTKFace':
            clf_state_dict = torch.load(CLF_PATH_UTKFace)['state_dict']
        elif config['dataset'] == 'FairFace':
            clf_state_dict = torch.load(CLF_PATH_FairFace)['state_dict']
        clf_classes = 2
    else:
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

    # obtain classifier predictions for samples
    preds = classify_examples(clf, config)  # (10K,)
    fair_d, l1_fair_d, kl_fair_d = utils.fairness_discrepancy(preds, clf_classes)
    # when comparing, just look at L2 for now!
    print('Fairness discrepancy metric is: {}'.format(fair_d))

    print('Itr %d: PYTORCH UNOFFICIAL FID is %5.4f' %
          (state_dict['itr'], FID))
    
    # compute a heuristic tradeoff
    tradeoff = FID + 100 * fair_d
    best_tradeoff_flag = 0
    
    # If improved over previous best metric, save appropriate copy

    # save model by both best FID (inaccurate) and fairness discrepancy
    if fair_d < state_dict['best_fair_d']:
        print('%s improved over previous best fair_d, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name, 'best_fair%d' % state_dict['save_best_num_fair'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_fair'] = (
            state_dict['save_best_num_fair'] + 1) % config['num_best_copies']

    if FID < state_dict['best_FID']:
        print('%s improved over previous best FID, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name, 'best_fid%d' % state_dict['save_best_num_fid'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_fid'] = (
            state_dict['save_best_num_fid'] + 1) % config['num_best_copies']

    if tradeoff < state_dict['best_tradeoff']:
        best_tradeoff_flag = 1
        print('Fairness and FID tradeoff improved over previous best tradeoff, saving checkpoint...')
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name, 'best_tradeoff%d' % state_dict['save_best_num_tradeoff'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_tradeoff'] = (
            state_dict['save_best_num_tradeoff'] + 1) % config['num_best_copies']
        
    # update best fairness discrepancy and FID score
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    state_dict['best_fair_d'] = min(state_dict['best_fair_d'], fair_d)
    state_dict['best_tradeoff'] = min(state_dict['best_tradeoff'], tradeoff)
    
    # Log results to file
    test_log.log(epoch=int(epoch), itr=int(state_dict['itr']), IS_mean=float(0), IS_std=float(0), FID=float(FID), FAIR=float(fair_d), 
                 L1_FAIR=float(l1_fair_d), KL_FAIR=float(kl_fair_d), TO=float(tradeoff), BEST_TO=int(best_tradeoff_flag))
    
    # leave logs via tensorboard
    writer.add_scalar("Performance/FID", FID, state_dict['itr'])
    writer.add_scalar("Performance/fair_d", fair_d, state_dict['itr'])
    writer.add_scalar("Performance/tradeoff", tradeoff, state_dict['itr'])

    
def update_intra_FID(G, D, D_v, G_ema, state_dict, config, FID, z1_FID, z0_FID, experiment_name, test_log, epoch, writer):
    print('Pre-loading pre-trained single-attribute classifier...')
    if config['dataset'] == 'CA64':
        clf_state_dict = torch.load(CLF_PATH_celeba)['state_dict']
    elif config['dataset'] == 'UTKFace':
        clf_state_dict = torch.load(CLF_PATH_UTKFace)['state_dict']
    elif config['dataset'] == 'FairFace':
        clf_state_dict = torch.load(CLF_PATH_FairFace)['state_dict']
    clf_classes = 2
    # load attribute classifier here
    clf = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], 
                    num_classes=clf_classes, grayscale=False) 
    clf.load_state_dict(clf_state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = clf.to(device)
    clf.eval()  # turn off batch norm

    # obtain classifier predictions for samples
    preds = classify_examples(clf, config)  # (10K,)
    fair_d, l1_fair_d, kl_fair_d = utils.fairness_discrepancy(preds, clf_classes)
    # when comparing, just look at L2 for now!
    print('Fairness discrepancy metric is: {}'.format(fair_d))

    print('Itr %d: PYTORCH UNOFFICIAL all groups, z1 and z0 FIDs are %5.4f, %5.4f and %5.4f' %
          (state_dict['itr'], FID, z1_FID, z0_FID))
    
    # compute a heuristic tradeoff
    tradeoff = FID + 100 * fair_d
    intra_tradeoff = z1_FID + z0_FID + 100 * fair_d
    
    best_fair_d_flag = 0
    best_fid_flag = 0
    best_z1_flag = 0
    best_z0_flag = 0
    best_tradeoff_flag = 0
    best_intra_tradeoff_flag = 0
    
    # If improved over previous best metric, save appropriate copy

    # save model by both best FID (inaccurate) and fairness discrepancy
    if fair_d < state_dict['best_fair_d']:
        best_fair_d_flag = 1
        print('%s improved over previous best fair_d, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name, 'best_fair%d' % state_dict['save_best_num_fair'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_fair'] = (
            state_dict['save_best_num_fair'] + 1) % config['num_best_copies']
        
    if FID < state_dict['best_FID']:
        best_fid_flag = 1
        print('%s improved over previous best FID, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name, 'best_fid%d' % state_dict['save_best_num_fid'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_fid'] = (
            state_dict['save_best_num_fid'] + 1) % config['num_best_copies']

    if z1_FID < state_dict['best_z1_FID']:
        best_z1_flag = 1
        print('%s improved over previous best z1 FID' %
              config['which_best'])
        
    if z0_FID < state_dict['best_z0_FID']:
        best_z0_flag = 1
        print('%s improved over previous best z0 FID' %
              config['which_best'])
        
    if tradeoff < state_dict['best_tradeoff']:
        best_tradeoff_flag = 1
        print('Fairness and FID tradeoff improved over previous best tradeoff, saving checkpoint...')
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name, 'best_tradeoff%d' % state_dict['save_best_num_tradeoff'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_tradeoff'] = (
            state_dict['save_best_num_tradeoff'] + 1) % config['num_best_copies']
        
    if intra_tradeoff < state_dict['best_intra_tradeoff']:
        best_intra_tradeoff_flag = 1
        print('Fairness and intra-FID tradeoff improved over previous best tradeoff, saving checkpoint...')
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name, 'best_intra_tradeoff%d' % state_dict['save_best_num_intra_tradeoff'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_intra_tradeoff'] = (
            state_dict['save_best_num_intra_tradeoff'] + 1) % config['num_best_copies']
        
    # update best fairness discrepancy and FID score
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    state_dict['best_z1_FID'] = min(state_dict['best_z1_FID'], z1_FID)
    state_dict['best_z0_FID'] = min(state_dict['best_z0_FID'], z0_FID)
    state_dict['best_fair_d'] = min(state_dict['best_fair_d'], fair_d)
    state_dict['best_tradeoff'] = min(state_dict['best_tradeoff'], tradeoff)
    state_dict['best_intra_tradeoff'] = min(state_dict['best_intra_tradeoff'], intra_tradeoff)
    
    # Log results to file
    test_log.log(epoch=int(epoch), itr=int(state_dict['itr']), IS_mean=float(0), IS_std=float(0), FID=float(FID), Z1_FID=float(z1_FID),
                 Z0_FID=float(z0_FID), FAIR=float(fair_d), L1_FAIR=float(l1_fair_d), KL_FAIR=float(kl_fair_d),
                 TO=float(tradeoff), INTRA_TO=float(intra_tradeoff), BEST_FID=int(best_fid_flag), BEST_Z1=int(best_z1_flag),
                 BEST_Z0=int(best_z0_flag), BEST_FAIR=int(best_fair_d_flag), BEST_TO=int(best_tradeoff_flag),
                 BEST_INTRA_TO=int(best_intra_tradeoff_flag))
    
    # leave logs via tensorboard
    writer.add_scalar("Performance/FID", FID, state_dict['itr'])
    writer.add_scalar("Performance/z1_FID", z1_FID, state_dict['itr'])
    writer.add_scalar("Performance/z0_FID", z0_FID, state_dict['itr'])
    writer.add_scalar("Performance/fair_d", fair_d, state_dict['itr'])
    writer.add_scalar("Performance/tradeoff", tradeoff, state_dict['itr'])
    writer.add_scalar("Performance/intra_tradeoff", intra_tradeoff, state_dict['itr'])
    
def update_intra_FID_multi(G, D, D_v, G_ema, state_dict, config, FID, z00_FID, z01_FID, z10_FID, z11_FID,
                           experiment_name, test_log, epoch, writer):
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

    # obtain classifier predictions for samples
    preds = classify_examples(clf, config)  # (10K,)
    fair_d, l1_fair_d, kl_fair_d = utils.fairness_discrepancy(preds, clf_classes)
    # when comparing, just look at L2 for now!
    print('Fairness discrepancy metric is: {}'.format(fair_d))

    print('Itr %d: PYTORCH UNOFFICIAL all groups, (z00, z01, z10, z11) intra-FIDs are %5.4f and (%5.4f, %5.4f, %5.4f, %5.4f)' %
          (state_dict['itr'], FID, z00_FID, z01_FID, z10_FID, z11_FID))
    
    # compute a heuristic tradeoff
    tradeoff = FID + 100 * fair_d
    intra_tradeoff = z00_FID + z01_FID + z10_FID + z11_FID + 100 * fair_d
    
    best_fair_d_flag = 0
    best_fid_flag = 0
    best_z00_flag = 0
    best_z01_flag = 0
    best_z10_flag = 0
    best_z11_flag = 0
    best_tradeoff_flag = 0
    best_intra_tradeoff_flag = 0
    
    # If improved over previous best metric, save appropriate copy

    # save model by both best FID (inaccurate) and fairness discrepancy
    if fair_d < state_dict['best_fair_d']:
        best_fair_d_flag = 1
        print('%s improved over previous best fair_d, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name, 'best_fair%d' % state_dict['save_best_num_fair'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_fair'] = (
            state_dict['save_best_num_fair'] + 1) % config['num_best_copies']
        
    if FID < state_dict['best_FID']:
        best_fid_flag = 1
        print('%s improved over previous best FID, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name, 'best_fid%d' % state_dict['save_best_num_fid'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_fid'] = (
            state_dict['save_best_num_fid'] + 1) % config['num_best_copies']

    if z00_FID < state_dict['best_z00_FID']:
        best_z00_flag = 1
        print('%s improved over previous best z00 FID' %
              config['which_best'])
        
    if z01_FID < state_dict['best_z01_FID']:
        best_z01_flag = 1
        print('%s improved over previous best z01 FID' %
              config['which_best'])
        
    if z10_FID < state_dict['best_z10_FID']:
        best_z10_flag = 1
        print('%s improved over previous best z10 FID' %
              config['which_best'])
        
    if z11_FID < state_dict['best_z11_FID']:
        best_z11_flag = 1
        print('%s improved over previous best z11 FID' %
              config['which_best'])

    if tradeoff < state_dict['best_tradeoff']:
        best_tradeoff_flag = 1
        print('Fairness and FID tradeoff improved over previous best tradeoff, saving checkpoint...')
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name, 'best_tradeoff%d' % state_dict['save_best_num_tradeoff'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_tradeoff'] = (
            state_dict['save_best_num_tradeoff'] + 1) % config['num_best_copies']
        
    if intra_tradeoff < state_dict['best_intra_tradeoff']:
        best_intra_tradeoff_flag = 1
        print('Fairness and intra-FID tradeoff improved over previous best tradeoff, saving checkpoint...')
        utils.save_weights(G, D, D_v, state_dict, config['weights_root'],
                           experiment_name, 'best_intra_tradeoff%d' % state_dict['save_best_num_intra_tradeoff'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num_intra_tradeoff'] = (
            state_dict['save_best_num_intra_tradeoff'] + 1) % config['num_best_copies']
        
    # update best fairness discrepancy and FID score
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    state_dict['best_z00_FID'] = min(state_dict['best_z00_FID'], z00_FID)
    state_dict['best_z01_FID'] = min(state_dict['best_z01_FID'], z01_FID)
    state_dict['best_z10_FID'] = min(state_dict['best_z10_FID'], z10_FID)
    state_dict['best_z11_FID'] = min(state_dict['best_z11_FID'], z11_FID)
    state_dict['best_fair_d'] = min(state_dict['best_fair_d'], fair_d)
    state_dict['best_tradeoff'] = min(state_dict['best_tradeoff'], tradeoff)
    state_dict['best_intra_tradeoff'] = min(state_dict['best_intra_tradeoff'], intra_tradeoff)
    
    # Log results to file
    test_log.log(epoch=int(epoch), itr=int(state_dict['itr']), IS_mean=float(0), IS_std=float(0), FID=float(FID), Z00_FID=float(z00_FID),
                 Z01_FID=float(z01_FID), Z10_FID=float(z10_FID), Z11_FID=float(z11_FID), FAIR=float(fair_d), L1_FAIR=float(l1_fair_d),
                 KL_FAIR=float(kl_fair_d), TO=float(tradeoff), INTRA_TO=float(intra_tradeoff), BEST_FID=int(best_fid_flag), 
                 BEST_Z00=int(best_z00_flag), BEST_Z01=int(best_z01_flag), BEST_Z10=int(best_z10_flag), BEST_Z11=int(best_z11_flag),
                 BEST_FAIR=int(best_fair_d_flag), BEST_TO=int(best_tradeoff_flag), BEST_INTRA_TO=int(best_intra_tradeoff_flag))
    
    # leave logs via tensorboard
    writer.add_scalar("Performance/FID", FID, state_dict['itr'])
    writer.add_scalar("Performance/z00_FID", z00_FID, state_dict['itr'])
    writer.add_scalar("Performance/z01_FID", z01_FID, state_dict['itr'])
    writer.add_scalar("Performance/z10_FID", z10_FID, state_dict['itr'])
    writer.add_scalar("Performance/z11_FID", z11_FID, state_dict['itr'])
    writer.add_scalar("Performance/fair_d", fair_d, state_dict['itr'])
    writer.add_scalar("Performance/tradeoff", tradeoff, state_dict['itr'])
    writer.add_scalar("Performance/intra_tradeoff", intra_tradeoff, state_dict['itr'])


def classify_examples(model, config):
    """
    classifies generated samples into appropriate classes 
    """
    model.eval()
    preds = []
    samples = np.load(config['sample_path'])['x']
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
            preds.append(pred)
        preds = torch.cat(preds).data.cpu().numpy()

    return preds