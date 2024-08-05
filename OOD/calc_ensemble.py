"""
Calculates OOD detection results using the MC Dropout or Ensembling.
"""

# Imports
import argparse
import os
import nibabel as nib
import numpy as np
import pandas as pd
import time
import utils

from scipy import stats

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-te', '--test_pth', type=str, required=True, help='Path to folder containing testing outputs.')
required.add_argument('-o', '--out_file', type=str, default=None, help='txt filename to write output to.')
required.add_argument('-p', '--per_seg_pth', type=str, required=True, help='Path to CSV files with segmentation performance results.')

optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-cm', '--correlation_metric', type=str, default=None, help='Name of column in csv file pointed to by correlation_path that contains the variable to calculate the coefficient on.\
                                                                                   Defaults to None')
optional.add_argument('-cp', '--correlation_pth', type=str, default=None, help='A path to a csv file to calculate the Pearson correlation coefficient with. \
                                                                                Filenames must be under file_col parameter.\
                                                                                Defaults to None.')
optional.add_argument('-fc', '--file_col', type=str, default='filename', help='Name of the column in the segmentation performance CSV file that contains the filenames.\
                                                                               Defaults to filename.')
optional.add_argument('-nc', '--name_col', type=str, default='DICE_Score', help='Name of the column in the segmentation performance CSV file that contains the dice scores.\
                                                                                Defaults to DICE_Score.')
optional.add_argument('-t', '--thres', type=float, default=0.95, help='Dice similarity coefficient threshold to determine which images are out-of-distribution. \
                                                                       Defaults to 0.95. \
                                                                       A value of None will result in a median threshold.')
args = parser.parse_args()

if __name__=="__main__":
    AUROC, AUPRC, FPR90, TIME, STAT, PVAL = [],[],[],[], [], []
    DSCs, HDs, SDs, numOODs = [], [], [], []
    if args.correlation_pth:
        df_c = pd.read_csv(args.correlation_pth)
        df_c[args.file_col] = df_c[args.file_col].apply(lambda x: x.split('.')[0] + '_output1')
        CORRs, CORR_ps = [], []
    for seed in range(1,6):
        np.random.seed(seed)
        
        df = pd.read_csv(args.per_seg_pth)
        df[args.file_col] = df[args.file_col].apply(lambda x: x.split('.')[0] + '_output1')
        dsc_mean = df[args.name_col].mean()
        hd_mean = df['HD'].mean()
        sd_mean = df['SD'].mean()
        files = [file for file in os.listdir(args.test_pth) if ('output1.' in file) and ('pt' not in file)]

        stdvs = []
        total_time = 0
        for file in files:
            preds = []
            for num in range(1,6):
                filename = file.split('.')[0][:-1] + str(num) + '.nii.gz'
                img_nib = nib.load(os.path.join(args.test_pth,filename))
                img = img_nib.get_fdata()
                preds.append(img)
            start = time.time()
            img_std = np.std(preds, axis=0)
            stdvs.append(np.mean(img_std))
            total_time += time.time() - start
        files = [file.split('.')[0] for file in files]
        
        in_dir = []
        if not args.thres:
            thres = df[args.name_col].median()
        else:
            thres = args.thres
        for filename in df[args.file_col]:
            if float(df[df[args.file_col] == filename][args.name_col]) >= thres:
                in_dir.append(filename)

        in_files, in_stdvs = [], []
        out_files, out_stdvs = [], []
        for i, file in enumerate(files):
            if file in in_dir:
                in_files.append(file)
                in_stdvs.append(stdvs[i])
            else:
                out_files.append(file)
                out_stdvs.append(stdvs[i])

        df_in = pd.DataFrame({f'{args.file_col}':in_files, f'dists':in_stdvs})
        df_out = pd.DataFrame({f'{args.file_col}':out_files, f'dists':out_stdvs})
        df_test = pd.concat([df_in,df_out])
        thres90 = df_out['dists'].quantile(q=0.10)
        comb = df_test.merge(df,how='inner',on=args.file_col)

        auroc, auprc, fpr90 = utils.evaluate(df_in.dists, df_out.dists)
        statistic, pvalue = stats.pearsonr(comb.dists, comb[args.name_col])
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
