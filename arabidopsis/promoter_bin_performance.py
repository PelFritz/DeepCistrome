import pyranges as pr
import pandas as pd
import seaborn as sns
from pyfaidx import Fasta
from utils import one_hot_encode
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
sns.set_theme(style="whitegrid")

if not os.path.exists('data/promoter_bins'):
    os.mkdir('data/promoter_bins')


def prepare_coords(annot, idx):
    annot = pr.read_gtf(annot, as_df=True)
    annot = annot[annot['Feature'] == 'gene']
    annot = annot[['Chromosome', 'Start', 'End', 'Strand']]
    annot = annot[annot['Chromosome'].isin(['1', '2', '3', '4', '5'])]
    coords = []
    for chrom, gene_start, gene_end, strand in annot.values:
        if strand == '+':
            start = max(0, gene_start-4000)
            coords.append([chrom, start + (idx * 250), start + ((idx + 1) * 250)])
        else:
            start = gene_end+4000
            coords.append([chrom, start - ((idx+1) * 250), start - (idx * 250)])
    coords = pd.DataFrame(coords)
    coords.drop_duplicates(subset=[0, 1, 2], inplace=True, keep='first')
    coords.to_csv(path_or_buf=f"data/promoter_bins/Bin_{idx+1}.bed", sep="\t", index=False, header=False)


def prepare_data(genome, bin_tab):
    genome = Fasta(genome, as_raw=True, read_ahead=10000, sequence_always_upper=True)
    seqs, y = [], []
    for coord in bin_tab.index.tolist():
        chrom, start, end = coord.split(':')
        seq = one_hot_encode(genome[chrom][int(start):int(end)])
        if seq.shape[0] == 250:
            seqs.append(seq)
            y.append(bin_tab.loc[coord, :].tolist())
    return np.array(seqs), np.array(y)


# Actual Binding for the Genes predicted above
for idx in range(4500//250):
    if not os.path.exists(f"data/promoter_bins/Bin_{idx+1}.bed") and not os.path.exists(f"data/promoter_bins/bin_{idx+1}_overlap.bed"):
        prepare_coords(annot='data/annotation/Arabidopsis_thaliana.TAIR10.59.gtf',
                       idx=idx)
        os.system(f"bedtools intersect -a data/promoter_bins/Bin_{idx+1}.bed -b data/peaks/*.bed -C -filenames -F 0.7 > data/promoter_bins/bin_{idx+1}_overlap.bed")
    print(f'Done with Bin {idx+1}')

bin_idx, num_tfs_true, num_tfs_pred = [], [], []
for overlap_file in os.listdir("data/promoter_bins/"):
    if overlap_file.endswith("_overlap.bed"):
        bin_df = pd.read_csv(filepath_or_buffer=f"data/promoter_bins/{overlap_file}", sep="\t", header=None,
                             dtype={0: str, 1: int, 2: int, 3: str, 4: int})

        bin_df[3] = bin_df[3].str.replace('data/peaks/', '').str.replace('.bed', '')
        bin_df[4] = [1 if v > 0 else v for v in bin_df[4]]
        bin_df[5] = [f"{v[0]}:{v[1]}:{v[2]}" for v in bin_df.values]
        bin_df = bin_df.pivot(columns=3, index=5, values=4)
        bin_df = bin_df[bin_df.sum(axis=1) > 0]
        bin_df['chrom'] = [int(x.split(':')[0]) for x in bin_df.index]

        bin_pred_sum, bin_true_sum = 0, 0
        for chrom_name in [1, 2, 3, 4, 5]:
            enc_seqs, targets = prepare_data(genome='data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa',
                                             bin_tab=bin_df[bin_df['chrom'] == chrom_name].drop(columns='chrom'))
            model = load_model(f"saved_models/model_chrom_{chrom_name}_model.h5")
            preds = model.predict(enc_seqs)
            preds = preds > 0.5
            preds = preds.astype(int)

            true_sum = np.sum(targets.sum(axis=1))
            pred_sum = np.count_nonzero(targets+preds == 2)
            bin_pred_sum += pred_sum
            bin_true_sum += true_sum

        bin_idx.append(int(overlap_file.split('_')[1]))
        num_tfs_true.append(bin_true_sum)
        num_tfs_pred.append(bin_pred_sum)
        print(bin_pred_sum/bin_true_sum)


tbs_data_true = pd.DataFrame(data={'Bin': bin_idx, 'num_tbs': num_tfs_true})
tbs_data_true = tbs_data_true.groupby(by='Bin').sum().reset_index()
tbs_data_true.sort_values(by='Bin', ascending=True, inplace=True)

tbs_data_pred = pd.DataFrame(data={'Bin': bin_idx, 'num_tbs': num_tfs_pred})
tbs_data_pred = tbs_data_pred.groupby(by='Bin').sum().reset_index()
tbs_data_pred.sort_values(by='Bin', ascending=True, inplace=True)
perc_pred_cor = tbs_data_pred['num_tbs'].values / tbs_data_true['num_tbs'].values
tbs_data_pred['num_tbs'] = tbs_data_pred['num_tbs']/tbs_data_pred['num_tbs'].sum()


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.barplot(x='Bin', y='num_tbs', data=tbs_data_pred, ax=ax, estimator='sum', errorbar=None, color='silver')
sns.pointplot(data=tbs_data_pred, x='Bin', y='num_tbs', errorbar=None, ax=ax, estimator='sum', color='cornflowerblue')
ax.set_ylim(0.05, 0.08)
ax.yaxis.grid(True, alpha=0.3)
ax.xaxis.grid(True, alpha=0.3)
ax.plot(15.5, 0.075, "*", markersize=10, color="teal")

# Annotate bars with percentage of actual binding even predicted correctly
for p, perc in zip(ax.patches, perc_pred_cor):
    ax.annotate(round(perc*100, 1), xy=(p.get_x()+p.get_width()/2, p.get_height()+0.001), ha='center', va='bottom',
                color='teal')
ax.set_xticklabels(['- 4000', '', '- 3500', '', '- 3000', '', '- 2500', '', '- 2000', '', '- 1500', '', '- 1000', '',
                    '- 500', '', '', '500'])
fig.tight_layout()
plt.savefig(f"results/Figures/performance_per_prom_bin.svg", bbox_inches='tight',
            dpi=300, format='svg')
plt.show()
