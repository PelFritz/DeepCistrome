import pandas as pd
import numpy as np
import seaborn as sns
from utils import one_hot_encode
from pyfaidx import Fasta
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from pybedtools import BedTool
import os
pd.options.display.width=0
sns.set_style("ticks")
genome = Fasta(filename='data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa', as_raw=True, read_ahead=10000,
               sequence_always_upper=True)
overlap_matrix = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed', sep='\t', nrows=2, index_col=0)
tf_families = [x.replace('_tnt', '') for x in overlap_matrix.columns.tolist()]


def overlap_moa_peaks_with_moa_peaks(moa_peaks, chrom_num, frac=0.7):
    BedTool.from_dataframe(moa_peaks).saveas(f'data/moa_{chrom_num}.bed')
    moa_bed = f'data/moa_{chrom_num}.bed'
    os.system(
        f"bedtools intersect -a {moa_bed} -b data/peaks/*.bed -C -filenames -F {frac} > data/moa_{chrom_num}_overlap.bed")


def prepare_overlap_mat(chrom_num):
    overlap_mat = pd.read_csv(filepath_or_buffer=f'data/moa_{chrom_num}_overlap.bed', sep='\t', header=None,
                              dtype={0: str, 1: int, 2: int, 3: str, 4: int})
    overlap_mat[3] = overlap_mat[3].str.replace('data/peaks/', '').str.replace('.bed', '').str.replace('_tnt', '')
    overlap_mat[4] = [1 if v > 0 else v for v in overlap_mat[4]]
    overlap_mat[5] = [f"{v[0]}:{v[1]}:{v[2]}" for v in overlap_mat.values]
    overlap_mat = overlap_mat.pivot(columns=3, index=5, values=4)
    return overlap_mat


def prepare_data(chroms, window_size):
    peaks_df = pd.read_csv(filepath_or_buffer='data/MOA/q_0.05/Ara_M5_peaks.narrowPeak', sep='\t', header=None)
    peaks_df = peaks_df[peaks_df[0].isin(chroms)]
    x = []
    moa_peaks, peak_ids = [], []
    for peak in peaks_df.values:
        chrom, summit = peak[0], int(peak[1]) + int(peak[9])
        start, end = 0 if summit-125 < 0 else summit-125, summit+125
        seq = one_hot_encode(genome[chrom.replace('Chr', '')][start:end])
        if seq.shape[0] == window_size:
            x.append(seq)
            moa_peaks.append([peak[0].replace('Chr', ''), start, end])
            peak_ids.append(f"{peak[0].replace('Chr', '')}:{start}:{end}")
    moa_peaks = pd.DataFrame(moa_peaks)
    overlap_moa_peaks_with_moa_peaks(moa_peaks=moa_peaks, chrom_num=chroms[0].replace('Chr', ''))
    overlap_df = prepare_overlap_mat(chrom_num=chroms[0].replace('Chr', ''))
    overlap_df = overlap_df.loc[peak_ids, overlap_df.columns.tolist()]
    return np.array(x), overlap_df

def compute_sensitivity(true_df, pred_df, columns):
    true_df = true_df.loc[:, columns]
    pred_df = pred_df.loc[:, columns]
    true_pred_inters = true_df.values + pred_df.values
    true_pred_inters = pd.DataFrame(true_pred_inters, columns=columns)
    true_pred_inters.replace(to_replace={1:0, 2:1}, inplace=True)
    return true_pred_inters

chromosomes = ['Chr1', 'Chr2', 'Chr3', 'Chr4', 'Chr5']
predictions_model, predictions_di, predictions_si, moa_dap_overlap = [], [], [], []
for chrom_name in chromosomes:
    enc_seqs, true_overlap = prepare_data(chroms=[chrom_name], window_size=250)
    assert enc_seqs.shape[0] == true_overlap.shape[0]
    moa_dap_overlap.append(true_overlap)
    dap_model = load_model(f"saved_models/model_chrom_{chrom_name.replace('Chr', '')}_model.h5")
    preds_model = dap_model.predict(enc_seqs)
    preds_model = preds_model > 0.5
    preds_model = preds_model.astype(int)
    predictions_model.append(preds_model)

    dap_di_model = load_model(f"saved_models/model_chrom_{chrom_name.replace('Chr', '')}_di.h5")
    preds_di_model = dap_di_model.predict(enc_seqs)
    preds_di_model = preds_di_model > 0.5
    preds_di_model = preds_di_model.astype(int)
    predictions_di.append(preds_di_model)

    dap_si_model = load_model(f"saved_models/model_chrom_{chrom_name.replace('Chr', '')}_si.h5")
    preds_si_model = dap_si_model.predict(enc_seqs)
    preds_si_model = preds_si_model > 0.5
    preds_si_model = preds_si_model.astype(int)
    predictions_si.append(preds_si_model)

predictions_model = pd.DataFrame(np.concatenate(predictions_model, axis=0), columns=tf_families)
predictions_model['bound'] = predictions_model.sum(axis=1)
predictions_di = pd.DataFrame(np.concatenate(predictions_di, axis=0), columns=tf_families)
predictions_di['bound'] = predictions_di.sum(axis=1)
predictions_si = pd.DataFrame(np.concatenate(predictions_si, axis=0), columns=tf_families)
predictions_si['bound'] = predictions_si.sum(axis=1)

data = pd.DataFrame(data={'model': ['model', 'd-baseline', 's-baseline'],
                          'accuracy':[len(np.where(predictions_model['bound'] > 0)[0])/predictions_model.shape[0],
                                      len(np.where(predictions_di['bound'] > 0)[0])/predictions_di.shape[0],
                                      len(np.where(predictions_si['bound'] > 0)[0])/predictions_si.shape[0],] })
print(f'Number of MOA peaks: {predictions_model.shape[0]}')
print(data.head())
fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(4, 8))
data.sort_values(by='model', inplace=True)
sns.barplot(x='model', y='accuracy', hue='model', data=data, ax=ax1, palette=['#00215E', '#FC4100', '#FFC55A'],
            hue_order=['model', 'd-baseline', 's-baseline'], order=['model', 'd-baseline', 's-baseline'])
sns.despine()
ax1.yaxis.grid(True, alpha=0.3)
ax1.xaxis.grid(True, alpha=0.3)
ax1.set_axisbelow(True)
ax1.set_ylim(0, 1)
fig1.tight_layout()
plt.savefig(f"results/Figures/performance_DAP_model_on_MOA.svg", bbox_inches='tight',
            dpi=300, format='svg')
plt.show()

moa_dap_overlap = pd.concat(moa_dap_overlap, axis=0)
pred_true_inters = compute_sensitivity(true_df=moa_dap_overlap, pred_df=predictions_model,
                                       columns=moa_dap_overlap.columns.tolist())
fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
data2_pred = pd.DataFrame(data={'tf family': tf_families,
                                'number of binding': predictions_model.sum(axis=0).values[:-1]})
data2_pred['Case'] = ['Predicted binding']*data2_pred.shape[0]

data2_true = pd.DataFrame(data={'tf family': moa_dap_overlap.columns.tolist(),
                                'number of binding': moa_dap_overlap.sum(axis=0)})
data2_true['Case'] = ['Dap-MOA overlap']*data2_true.shape[0]

data2_true_pred_int = pd.DataFrame(data={'tf family': pred_true_inters.columns.tolist(),
                                         'number of binding': pred_true_inters.sum(axis=0)})
data2_true_pred_int['Case'] = ['Correct predictions']*data2_true_pred_int.shape[0]

data2 = pd.concat([data2_pred, data2_true, data2_true_pred_int], axis=0)

data2.sort_values(by='number of binding', inplace=True)
sns.barplot(data=data2, x='tf family', y='number of binding', hue='Case',
            hue_order=['Dap-MOA overlap', 'Predicted binding', 'Correct predictions'],
            palette=['#5F8670', '#FF9800', '#B80000'])
sns.despine()
ax2.yaxis.grid(True, alpha=0.3)
ax2.xaxis.grid(True, alpha=0.3)
ax2.set_axisbelow(True)
plt.xticks(rotation=90)
fig2.tight_layout()
plt.savefig(f"results/Figures/DAP_model_on_MOA_per_family.svg", bbox_inches='tight',
            dpi=300, format='svg')
plt.show()
