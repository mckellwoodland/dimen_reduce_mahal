"""
Calculates OOD detection results using the Maximum Softmax Probability.

NumPy logits must be saved as .npy files.
"""

# Imports
import argparse
import os
import utils
import time
import torch
import tqdm

import nibabel as nib
import numpy as np
import pandas as pd

from scipy import special, stats

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-te', '--test_pth', type=str, required=True, help='Path to folder containing testing logits.')
required.add_argument('-o', '--out_file', type=str, default=None, help='txt filename to write output to.')
required.add_argument('-p', '--per_seg_pth', type=str, required=True, help='Path to CSV files with segmentation performance results.')

optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-c','--calibration', type=str, default=None, help='Whether to use temperature scaling [ts] or energy [energy].\
                                                                          Defaults to None.')
optional.add_argument('-cm', '--correlation_metric', type=str, default=None, help='Name of column in csv file pointed to by correlation_path that contains the variable to calculate the coefficient on.\
                                                                                   Defaults to None')
optional.add_argument('-cp', '--correlation_pth', type=str, default=None, help='A path to a csv file to calculate the Pearson correlation coefficient with. \
                                                                                Filenames must be under file_col parameter.\
                                                                                Defaults to None.')
optional.add_argument('-e', '--embed_type', type=str, default='numpy', help='Whether the embeddings were saved as NumPy (.npy) [numpy] or NifTI (.nii.gz.) [nifti] embeddings. \
                                                                             Defaults to numpy.')
optional.add_argument('-fc', '--file_col', type=str, default='filename', help='Name of the column in the segmentation performance CSV file that contains the filenames.\
                                                                               Defaults to filename.')
optional.add_argument('-nc', '--name_col', type=str, default='DICE_Score', help='Name of the column in the segmentation performance CSV file that contains the dice scores.\
                                                                                Defaults to DICE_Score.')
optional.add_argument('-t', '--thres', type=float, default=0.95, help='Dice similarity coefficient threshold to determine which images are out-of-distribution. \
                                                                       Defaults to 0.95. \
                                                                       A value of None will result in a median threshold.')
optional.add_argument('-tem', '--temperature', type=float, default=None, help='Temperature for temperature scaling and energy scoring.\
                                                                              Defaults to None.\
                                                                              A value of None will result in a temperature search.')
args = parser.parse_args()

if not args.calibration:
    T = [1]
elif args.temperature:
    T = [args.temperature]
else:
    T = [1,2,3,4,5,10,100,1000]

if __name__=="__main__":
    df = pd.read_csv(args.per_seg_pth)
    dsc_mean = df[args.name_col].mean()
    hd_mean = df['HD'].mean()
    sd_mean = df['SD'].mean()
    for t in T:
        AUROC, AUPRC, FPR90, TIME, STAT, PVAL = [],[],[],[], [], []
        DSCs, HDs, SDs, numOODs = [], [], [], []
        if args.correlation_pth:
            df_c = pd.read_csv(args.correlation_pth)
            CORRs, CORR_ps = [], []
        for seed in range(1,6):
            np.random.seed(seed)
            test_f, dists = [],[]
            total_time = 0
            for img_f in tqdm.tqdm(os.listdir(args.test_pth)):
                test_f.append(img_f.split('.')[0])
                if args.embed_type == 'nifti':
                    logit_fg = nib.load(os.path.join(args.test_pth,img_f)).get_fdata()
                elif args.embed_type == 'numpy':
                    logit_fg = np.load(os.path.join(args.test_pth,img_f))
                start = time.time()
                logit_fg_exp = np.expand_dims(logit_fg, axis=0)
                logit_bg = -logit_fg_exp
                logits = np.vstack([logit_fg_exp,logit_bg])
                if args.calibration == 'energy':
                    logits = torch.from_numpy(logits)
                    uncertainty = -(t*torch.logsumexp(logits / t, dim=0)).numpy()
                    dists.append(abs(np.mean(uncertainty)))
                else:
                    logits /= t
                    sft = special.softmax(logits,axis=0)
                    uncertainty = np.max(sft,axis=0)
                    dists.append(1-np.mean(uncertainty))
                total_time += time.time() - start
            df_test = pd.DataFrame({f'{args.file_col}':test_f,
                                        'dists':dists})
            comb = df_test.merge(df,how='inner',on=f'{args.file_col}')
            
            # Split test into ID/OOD by the threshold.
            df_ID = comb[comb[args.name_col] >= args.thres]
            df_OOD = comb[comb[args.name_col] < args.thres]
            thres90 = df_OOD['dists'].quantile(q=0.10)
            auroc, auprc, fpr90 = utils.evaluate(df_ID.dists, df_OOD.dists)
            statistic, pvalue = stats.pearsonr(comb.dists, comb[f'{args.name_col}'])
            AUROC.append(auroc)
            AUPRC.append(auprc)
            FPR90.append(fpr90)
            TIME.append(total_time)
            STAT.append(statistic)
            PVAL.append(pvalue)
            DSCs.append(comb[comb['dists'] <= thres90][args.name_col].mean() - dsc_mean)
            HDs.append(comb[comb['dists'] <= thres90]['HD'].mean() - hd_mean)
            SDs.append(comb[comb['dists'] <= thres90]['SD'].mean() - sd_mean)
            numOODs.append(len(comb[comb['dists'] > thres90]))
            if args.correlation_pth:
                if args.correlation_metric in df_test.columns:
                    df_test = df_test.drop(args.correlation_metric,axis=1)
                df_comb = df_test.merge(df_c,how='inner',on=args.file_col)
                stat, p = stats.pearsonr(df_comb.dists, df_comb[args.correlation_metric])
                CORRs.append(stat)
                CORR_ps.append(p)
        with open(args.out_file, 'a') as f:
            f.write(f"----{t}----\n")
            f.write(f"AUROC: {round(np.mean(AUROC),2)} ({round(np.std(AUROC),2)}) {AUROC}\n")
            f.write(f"AUPRC: {round(np.mean(AUPRC),2)} ({round(np.std(AUPRC),2)}) {AUPRC}\n")
            f.write(f"FPR90: {round(np.mean(FPR90),2)} ({round(np.std(FPR90),2)}) {FPR90}\n")
            f.write(f"TIME:  {round(np.mean(TIME) ,2)} ({round(np.std(TIME) ,2)}) {TIME} \n")
            f.write(f"Pearson: {round(np.mean(STAT),2)} ({round(np.std(STAT),2)}) {STAT} {PVAL}\n")
            if args.correlation_pth:
                f.write(f"Pearson with {args.correlation_metric}: {round(np.mean(CORRs),2)} ({round(np.std(CORRs),2)})\n{CORRs}\n{CORR_ps}\n")
            f.write(f"DSC: {round(np.mean(DSCs),2)} ({round(np.std(DSCs),2)})\n")
            f.write(f"HD: {round(np.mean(HDs),2)} ({round(np.std(HDs),2)})\n")
            f.write(f"SD: {round(np.mean(SDs),2)} ({round(np.std(SDs),2)})\n")
            f.write(f"#OOD: {round(np.mean(numOODs),2)} ({round(np.std(numOODs),2)})\n")
