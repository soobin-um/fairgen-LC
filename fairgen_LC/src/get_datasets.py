"""
Save all the balanced/unbalanced datasets we will be using for our generative model
"""
import os
import sys
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from dataset_splits import (
    build_90_10_unbalanced_datasets_clf_celeba,
    build_80_20_unbalanced_datasets_clf_celeba,
    build_multi_datasets_clf_celeba,
    build_90_10_unbalanced_datasets_UTKFace,
    build_90_10_unbalanced_datasets_FairFace
)

import seaborn as sns


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='celeba, UTKFace, or FairFace')
    parser.add_argument('--perc', type=float, default=1.0,
                        help='size of balanced dataset [0.1, 0.25, 0.5, 1.0]')
    parser.add_argument('--bias', type=str, default='90_10',
                        help='type of bias [90_10, 80_20, multi]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=777,
                        help='seed for reproducibility [default: 777]')
    parser.add_argument('--reproducible', action='store_true', default=False,
                        help='Use settings for encouraging reproducibility? (default: %(default)s)')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.reproducible:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        #     torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if args.cuda else 'cpu')
    
    assert args.dataset in ['celeba', 'UTKFace', 'FairFace']

    # grab appropriate dataset splits
    if args.dataset == 'celeba':
        assert args.perc in [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        if args.bias == '90_10':
            balanced_train_dataset, unbalanced_train_dataset = build_90_10_unbalanced_datasets_clf_celeba(
                args.dataset, 'train', args.perc)
            bias = '90_10_perc{}'.format(args.perc)
        elif args.bias == '80_20':
            balanced_train_dataset, unbalanced_train_dataset = build_80_20_unbalanced_datasets_clf_celeba(
                args.dataset, 'train', args.perc)
            bias = '80_20_perc{}'.format(args.perc)
        elif args.bias == 'multi':
            balanced_train_dataset, unbalanced_train_dataset = build_multi_datasets_clf_celeba(
                args.dataset, 'train', args.perc)
            bias = 'multi_perc{}'.format(args.perc)
        else:
            raise NotImplementedError
            
    elif args.dataset == 'UTKFace':
        assert args.perc in [0.1, 0.25]
        if args.bias == '90_10':
            balanced_train_dataset, unbalanced_train_dataset = build_90_10_unbalanced_datasets_UTKFace(
                args.dataset, 'train', args.perc)
            bias = '90_10_perc{}'.format(args.perc)
        else:
            raise NotImplementedError
            
    elif args.dataset == 'FairFace':
        assert args.perc in [0.1, 0.25]
        if args.bias == '90_10':
            balanced_train_dataset, unbalanced_train_dataset = build_90_10_unbalanced_datasets_FairFace(
                args.dataset, 'train', args.perc)
            bias = '90_10_perc{}'.format(args.perc)
        else:
            raise NotImplementedError

    # save outputs in correct directory
    args.out_dir = '../data/{0}/{0}_{1}'.format(args.dataset, bias)
    print('outputs will be saved to: {}'.format(args.out_dir))
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # run through unbalanced dataset
    print('saving unbalanced dataset')
    # save values
    unbalanced_train_loader = torch.utils.data.DataLoader(unbalanced_train_dataset.dataset, batch_size=100, shuffle=False)
    train_data = []
    train_labels = []
    attrs = []

    with torch.no_grad():
        # only iterating through unbalanced dataset!
        for data, attr, target in unbalanced_train_loader:
            data = data.float() / 255.
            attr, target = attr.long(), target.long()

            # save data, density ratios, and labels
            train_data.append(data)
            train_labels.append(target)
            attrs.append(attr)
        train_data = torch.cat(train_data)
        train_labels = torch.cat(train_labels)
        attrs = torch.cat(attrs)
    train_data = (train_data * 255).to(torch.uint8)

    # save files
    # NOTE: we are returning the true attr labels so that we can look at the density ratios across classes later
    torch.save(attrs.data.cpu(), os.path.join(args.out_dir, '{}_unbalanced_train_attr_labels.pt'.format(args.dataset)))
    torch.save(train_data.data.cpu(), os.path.join(args.out_dir, '{}_unbalanced_train_data.pt'.format(args.dataset)))
    torch.save(train_labels.data.cpu(), os.path.join(args.out_dir, '{}_unbalanced_train_labels.pt'.format(args.dataset)))

    print('saving balanced dataset')
    # save balanced dataset
    balanced_train_loader = torch.utils.data.DataLoader(balanced_train_dataset.dataset, batch_size=100, shuffle=False)
    train_data = []
    train_labels = []

    # save density ratios and labels
    with torch.no_grad():
        # only iterating through unbalanced dataset!
        for data,attr,target in balanced_train_loader:
            data = data.float() / 255.
            target = target.long()

            # save data, density ratios, and labels
            train_data.append(data)
            train_labels.append(target)
        train_data = torch.cat(train_data)
        train_labels = torch.cat(train_labels)
    train_data = (train_data * 255).to(torch.uint8)
    torch.save(train_data.data.cpu(), os.path.join(args.out_dir, '{}_balanced_train_data.pt'.format(args.dataset)))
    torch.save(train_labels.data.cpu(), os.path.join(args.out_dir,'{}_balanced_train_labels.pt'.format(args.dataset)))
