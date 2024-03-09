# A Fair Generative Model Using LeCam divergence (AAAI 2023 Oral)

Soobin Um and Changho Suh

This repository contains the official code for the paper "A Fair Generative Model Using LeCam divergence" (AAAI 2023 Oral).

If you want to run the proposed algorithm, use the codes in `fairgen_LC` folder. Or if you are interested in the baselines, use the codes contained in `baselines`.

## 1) Data setup
### 1-1) CelebA:

(a) Download the CelebA dataset here (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) into the `data/` directory (if elsewhere, note the path for step b). Of the download links provided, choose `Align&Cropped Images` and download `Img/img_align_celeba/` folder, `Anno/list_attr_celeba.txt`, and `Eval/list_eval_partition.txt` to `data/`.

(b) Preprocess the CelebA dataset for faster training:
```
python3 preprocess_celeba.py --data_dir=/path/to/downloaded/dataset/celeba/ --out_dir=../data/celeba --partition=train
```

You should run this script for `--partition=[train, val, test]` to cache all the necessary data. The preprocessed files will then be saved in `data/`.

To split the data for multiple attributes, check `notebooks/multi-attribute data and unbiased FID splits.ipynb`.

### 1-2) UTKFace:

(a) Download the UTKFace dataset here (https://susanqq.github.io/UTKFace/) into the `data/UTKFace/` directory (if elsewhere, note the path for step b). Of the download links provided, choose `Align&Cropped Faces` and download `UTKFace.tar.gz` file, and decompress it into `data/UTKFace/` directory.

(b) Preprocess the UTKFace dataset for faster training:
```
python3 preprocess_UTKFace.py --data_dir=/path/to/downloaded/dataset/UTKFace/ --out_dir=../data/UTKFace
```

Contrary to CelebA dataset, this run will automatically generate train/val/test splits in the designated directory.

### 1-3) FairFace:

(a) Download the FairFace dataset here (https://github.com/dchen236/FairFace) into the `data/FairFace` directory (if elsewhere, note the path for step b). Of the download links provided, download `Padding=0.25` together with labels `Train` and `Validation`. Unzip the compressed file into `data/FairFace` directory and place the two label files in the same folder.

(b) Preprocess the FairFace dataset for faster training:
```
python3 preprocess_FairFace.py --data_dir=/path/to/downloaded/dataset/FairFace/ --out_dir=../data/FairFace --split_test 1
```

As in the UTKFace preprocessing, this run will automatically generate train/val/test splits in the designated directory.


## 2) Pre-train attribute classifier
### 2-1) For CelebA-single (female-vs-male):
```
python3 train_attribute_clf.py celeba ./results/celeba/attr_clf
```

### 2-2) For CelebA-multi (non-black-hair vs black-hair), add the `--multi=True` flag.
```
python3 train_attribute_clf.py celeba ./results/celeba/multi_clf -- multi=True
```

### 2-3) For UTKFace (white-vs-non-white):
```
python3 train_attribute_clf.py UTKFace ./results/UTKFace/attr_clf
```

### 2-4) For FairFace (white-vs-black):
```
python3 train_attribute_clf.py FairFace ./results/FairFace/attr_clf
```

Then, the trained attribute classifier will be saved in the designated folder and will be used for downstream evaluation for generative model training. Note the path where these classifiers are saved, as they will be needed for GAN training + evaluation.


## 3) Construct training and reference datasets (for proposed approach)
We obtain training and reference datasets by splitting the full datasets that we preprocessed in step 1).
The training and reference datasets should be constructed so as to respect the appropriate `bias` and `perc (reference set size relative to training set)` setting, which can be adjusted in the script below:
```
python3 get_datasets.py [celeba, UTKFace, FairFace] --perc=[0.01, 0.025, 0.05, 0.1, 0.25] --bias=[90_10, multi]
```
Note that `--perc=[0.01, 0.025, 0.05]` and `--bias=multi` settings are available only for CelebA dataset.

## 3) Pre-train density ratio classifier (for baselines)

The density ratio classifier should be trained for the appropriate `bias` and `perc` setting, which can be adjusted in the script below:
```
python3 get_density_ratios.py [celeba, UTKFace, FairFace] [ResNet18, CNN5, CNN3] --perc=[0.1, 0.25, 0.5, 1.0] --bias=[90_10, 80_20, multi]
```
By employing `--ncf` argument, you can control the complexity of CNN classifiers. You can incorporate temperature scaling via `--cal`. The temperature scaling code is from: https://github.com/gpleiss/temperature_scaling.

Note that the best density ratio classifier will be saved in its corresponding directory under `./data/[celeba, UTKFace, FairFace]`.


## 4) Pre-compute statistics for intra FID:

### 4-1) CelebA
For CelebA-single, we provide pre-computed statistics for intra FID in the source directory.

(a) `./female_fid_stats.npz` contains activations from female group of CelebA dataset.

(b) `./male_fid_stats.npz` contains activations from male group of CelebA dataset.

Place the above files to `./[fairgen_LC, baselines]/fid_stats/celeba` (or `./baselines/fid_stats/celeba`, if you want to run baseline schemes).

These pre-computed FID statistics are for model checkpointing (during GAN training) and downstream evaluation of sample quality during training on CelebA-single scenario.

For CelebA-multi, please do the following steps:

(a) To prepare sample data for FID statistic calculation, check `./[fairgen_LC baselines]/notebooks/intra-FID splits_celeba`.

(b) Execute `./[fairgen_LC, baselines]/src/[BigGAN, KL-BigGAN]/precompute_celeba_multi_intra_fid_stats.py`

Then, the unbiased FID statistics for UTKFace will be saved in `./[fairgen_LC, baselines]/fid_stats/celeba`. This pre-computed FID statistics are for model checkpointing (during GAN training) and downstream evaluation of sample quality during training on CelebA-multi scenario.

### 4-2) UTKFace

(a) To prepare sample data for unbiased FID statistic calculation, check `./[fairgen_LC, baselines]/notebooks/intra-FID splits_UTKFace.ipynb`.

(b) Execute `./[fairgen_LC, baselines]/src/[BigGAN, KL-BigGAN]/precompute_UTKFace_intra_fid_stats.py`

Then, the unbiased FID statistics for UTKFace will be saved in `./[fairgen_LC, baselines]/fid_stats/UTKFace`. This pre-computed FID statistics are for model checkpointing (during GAN training) and downstream evaluation of sample quality during training on UTKFace dataset.

### 4-3) FairFace

(a) To prepare sample data for unbiased FID statistic calculation, check `./[fairgen_LC, baselines]/notebooks/intra-FID splits_FairFace.ipynb`.

(b) Execute `./[fairgen_LC, baselines]/src/[BigGAN, KL-BigGAN]/precompute_FairFace_intra_fid_stats.py`

Then, the unbiased FID statistics for FairFace will be saved in `./[fairgen_LC, baselines]/fid_stats/FairFace`. This pre-computed FID statistics are for model checkpointing (during GAN training) and downstream evaluation of sample quality during training on FairFace dataset.


## 5) Train generative model (BigGAN)
Sample scripts to train the model can be found in `./[fairgen_LC, baselines]/scripts/`:

For instance, you can run the proposed approach on CelebA-single with 10% reference set as
`bash run_celeba_single_0.1_fairgen_LC`

We also provide script files for the three baselines executable as
`bash run_celeba_perc0.1_Baseline_1.sh`
`bash run_celeba_perc0.1_Baseline_2.sh`
`bash run_celeba_perc0.1_Choi_et_al.sh`

You can run on other datasets by replacing `--dataset` into `UTKFace` or `FairFace` in the script file. The reference set size is adjusted by `--perc`. For the multi-attribute setting, set `--bias multi` and append ` --multi 1`.

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{um2023fair,
  title={A fair generative model using lecam divergence},
  author={Um, Soobin and Suh, Changho},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={8},
  pages={10034--10042},
  year={2023}
}
```