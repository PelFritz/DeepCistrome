import pandas as pd
import numpy as np
import seaborn as sns
from utils import one_hot_encode
from pyfaidx import Fasta
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from pybedtools import BedTool
pd.options.display.width=0
sns.set_context(context="paper", rc={"font.size":12,"axes.titlesize":12,"axes.labelsize":12})

overlap_mat = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed', sep='\t', nrows=2, index_col=0)
tf_families = overlap_mat.columns.tolist()
use_cols = ['snp.chr', 'snp.position', 'snp.ref', 'snp.alt', 'maf',
           'snp.annotations.0.effect', 'snp.annotations.0.impact',
           'study.phenotype.name']
sign_ass_araGWAScatalog = 'data/gwas_hits/aragwas_associations_significant_permutation.csv'
sign_ass_araGWAScatalog = pd.read_csv(sign_ass_araGWAScatalog, usecols=use_cols)
sign_ass_araGWAScatalog['chrom'] = sign_ass_araGWAScatalog['snp.chr'].str.replace('chr', '')
sign_ass_araGWAScatalog = sign_ass_araGWAScatalog[sign_ass_araGWAScatalog['chrom'].isin(['1', '2', '3', '4', '5'])]
sign_ass_araGWAScatalog.dropna(how='any', axis=0, inplace=True)
sign_ass_araGWAScatalog['start'] = sign_ass_araGWAScatalog['snp.position']
sign_ass_araGWAScatalog['start'] = sign_ass_araGWAScatalog['start'].astype('int')
sign_ass_araGWAScatalog['end'] = sign_ass_araGWAScatalog['snp.position']
sign_ass_araGWAScatalog['end'] = sign_ass_araGWAScatalog['end'].astype('int')
sign_ass_araGWAScatalog = sign_ass_araGWAScatalog[['chrom', 'start', 'end', 'snp.ref', 'snp.alt',
                                                   'maf', 'snp.annotations.0.effect','snp.annotations.0.impact',
                                                   'study.phenotype.name']]
print(sign_ass_araGWAScatalog.head())

def prepare_moa_peaks():
    peaks_df = pd.read_csv(filepath_or_buffer='data/MOA/q_0.05/Ara_M5_peaks.narrowPeak', sep='\t', header=None)
    moa_peaks = []
    for peak in peaks_df.values:
        chrom, summit = peak[0], int(peak[1]) + int(peak[9])
        start, end = 0 if summit-125 < 0 else summit-125, summit+125
        moa_peaks.append([peak[0].replace('Chr', ''), start, end])
    moa_peaks = pd.DataFrame(moa_peaks)
    return moa_peaks


def create_dataset(chrom, genome, peak_coords, window_size=1000):
    onehot_dict = {'A': [1, 0, 0, 0],
                   'C': [0, 1, 0, 0],
                   'G': [0, 0, 1, 0],
                   'T': [0, 0, 0, 1]}
    genome = Fasta(genome, as_raw=True, read_ahead=10000, sequence_always_upper=True)
    peak_coords[[0, 9]] = peak_coords[[0, 9]].astype('str')
    peak_coords = peak_coords[peak_coords[9] == chrom]

    seqs, snp_seqs, snp_meta = [], [], []
    for coord in peak_coords.values:
        chrom, start, end = coord[9], coord[10], coord[11]
        snp_ref, snp_alt, snp_pos = coord[3], coord[4], coord[1]-1
        enc_seq = one_hot_encode(genome[chrom][start:end])
        enc_seq_snp = one_hot_encode(genome[chrom][start:end])
        enc_seq_snp[snp_pos - start, :] = onehot_dict[snp_alt]
        if enc_seq.shape[0] == window_size:
            seqs.append(enc_seq)
            snp_seqs.append(enc_seq_snp)
            snp_meta.append([f"{chrom}.{snp_pos+1}.{snp_ref}_{snp_alt}", coord[6], coord[7]])

    return np.array(seqs), np.array(snp_seqs), np.array(snp_meta)

moa_peaks_coords_df = prepare_moa_peaks()
moa_peaks_coords_df.to_csv(path_or_buf='data/moa_peaks_coords.bed', index=False, header=False, sep='\t')

inters_results = BedTool.from_dataframe(sign_ass_araGWAScatalog).intersect('data/moa_peaks_coords.bed', wa=True, wb=True)
inters_results = inters_results.to_dataframe(disable_auto_names=True, header=None)
print(inters_results.head())

mutation_effect_dfs, mutation_effect_meta = [], []
for chrom in ['1', '2', '3', '4', '5']:
    orig_seq, mut_seq, snp_metas = create_dataset(chrom=chrom,
                                                  genome='data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa',
                                                  peak_coords=inters_results,
                                                  window_size=250)
    model = load_model(f'saved_models/model_chrom_{chrom}_model.h5')
    print(orig_seq.shape, mut_seq.shape)

    # Predictions actual sequence
    preds_actual = model.predict(orig_seq)
    preds_actual = preds_actual > 0.5
    preds_actual = preds_actual.astype('int')

    # Predictions SNP mutated sequence
    preds_mutated = model.predict(mut_seq)
    preds_mutated = preds_mutated > 0.5
    preds_mutated = preds_mutated.astype('int')

    mutation_effect = preds_actual - preds_mutated
    mutation_effect = pd.DataFrame(mutation_effect, columns=tf_families)
    mutation_effect_dfs.append(mutation_effect)

    snp_meta_df = pd.DataFrame(snp_metas, columns=['SNP.ID', 'Annotation', 'Impact'])
    mutation_effect_meta.append(snp_meta_df)

mutation_effect_dfs = pd.concat(mutation_effect_dfs, axis=0)
mutation_effect_meta = pd.concat(mutation_effect_meta, axis=0)
mutation_effect_meta['binding change'] = mutation_effect_dfs.abs().sum(axis=1)
mutation_effect_meta['binding change'] = ['yes' if i != 0 else 'no' for i in mutation_effect_meta['binding change']]
print(mutation_effect_dfs.head())
print(mutation_effect_meta.head())
fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(15, 4))
sns.countplot(mutation_effect_meta, ax=ax[0], x='binding change', stat='percent', color='#001F3F')
ax[0].yaxis.grid(True, alpha=0.3)
ax[0].xaxis.grid(True, alpha=0.3)
ax[0].set_axisbelow(True)
ax[0].set_ylim(0, 100)

sns.countplot(mutation_effect_meta[mutation_effect_meta['binding change'] == 'yes'],
              ax=ax[1], x='Annotation', stat='percent', hue='Impact',
              palette=['#640D5F', '#FF6600', '#EE66A6'])
ax[1].yaxis.grid(True, alpha=0.3)
ax[1].xaxis.grid(True, alpha=0.3)
ax[1].set_axisbelow(True)
ax[1].set_ylim(0, 100)
fig.tight_layout()
plt.savefig(f"results/Figures/SNPs_MOA_peaks.svg", bbox_inches='tight', dpi=300, format='svg')
plt.show()