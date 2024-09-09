import pandas as pd
from pyfaidx import Fasta
import numpy as np
from utils import one_hot_encode
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import os


def create_dataset(genome, coords):
    genome = Fasta(genome, as_raw=True, read_ahead=10000, sequence_always_upper=True)
    x = []
    for chrom, start, end in coords:
        seq = one_hot_encode(genome[chrom][int(start):int(end)])
        if seq.shape[0] == 250:
            x.append(seq)
    return np.array(x)


def create_coords(peak_file, chrom):
    df = pd.read_csv(filepath_or_buffer=f"data/peaks/{peak_file}", sep='\t', header=None, dtype={0: str, 1: int, 2: int})
    df = df[df[0] == chrom]

    coords = []
    for c, s, e in df.values:
        midpoint = (s+e)//2
        coords.append([c, max(0, midpoint-125), midpoint+125])
    return coords


overlap_mat = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed', sep='\t', index_col=0, nrows=3)
tfs = overlap_mat.columns.tolist()

tf_name, pred_perc = [], []
for peaks in os.listdir('data/peaks'):
    print(peaks)
    pred_sum, true_sum = 0, 0
    for chrom_name in ['1', '2', '3', '4', '5']:
        coords_list = create_coords(peak_file=peaks, chrom=chrom_name)
        model = load_model(f'saved_models/model_chrom_{chrom_name}_model.h5')
        enc_seqs = create_dataset(genome='data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa', coords=coords_list)
        print(f"Chrom: {chrom_name}, size: {len(coords_list)}, TF: {peaks.split('.bed')[0]}")
        predictions = model.predict(enc_seqs)
        correct_preds = np.where(predictions[:, tfs.index(peaks.split('.bed')[0])] > 0.5)[0]
        pred_sum += len(correct_preds)
        true_sum += enc_seqs.shape[0]

    tf_name.append(peaks.split('.bed')[0])
    pred_perc.append(pred_sum / true_sum)

data = pd.DataFrame({'TF': tf_name, 'Predicted Percentage': pred_perc})
data.sort_values(by=['Predicted Percentage'], ascending=False, inplace=True)
data.to_csv(path_or_buf='results/predicetd_percentages_per_family.csv', index=False)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
sns.barplot(x='TF', y='Predicted Percentage', data=data, ax=ax, errorbar=None, color='black')
ax.yaxis.grid(True, alpha=0.3)
ax.xaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.axhline(y=0.5, color='silver', linestyle='--')
plt.xticks(rotation=90)
fig.tight_layout()
plt.savefig(f"results/Figures/performance_on_peaks_per_tf.svg", bbox_inches='tight',
            dpi=300, format='svg')
plt.show()


