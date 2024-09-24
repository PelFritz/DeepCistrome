import pandas as pd
import numpy as np
import pyranges as pr
from pybedtools import BedTool
from pyfaidx import Fasta
from utils import one_hot_encode
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.width=0
sns.set_context(context="paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})
promoter_size = 1000
overlap_mat = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed', sep='\t', nrows=2, index_col=0)
tf_families = overlap_mat.columns.tolist()
gwas_hits = pd.read_csv(filepath_or_buffer='data/gwas_hits/aragwas_associations_maf_filtered_annotation_filtered_significant_permutation.csv',
                        usecols=['snp.chr', 'snp.position', 'snp.ref', 'snp.alt', 'maf',
                                 'snp.annotations.0.effect', 'snp.annotations.0.impact',
                                 'study.phenotype.name'],
                        dtype={'study.phenotype.name': str, 'snp.position': int, 'maf': float})
gwas_hits['chrom'] = gwas_hits['snp.chr'].str.replace('chr', '')
gwas_hits['start'] = gwas_hits['snp.position']
gwas_hits['end'] = gwas_hits['snp.position']
gwas_hits = gwas_hits[['chrom', 'start', 'end', 'snp.ref', 'snp.alt', 'maf', 'snp.annotations.0.effect',
                       'snp.annotations.0.impact', 'study.phenotype.name']]
print(gwas_hits.head())
print(gwas_hits['chrom'].unique())
def get_promoter_coords(promoter_size=1000):
    gene_models = pr.read_gtf('data/annotation/Arabidopsis_thaliana.TAIR10.59.gtf', as_df=True)
    gene_models = gene_models[gene_models['Feature'] == 'gene']
    gene_models = gene_models[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
    gene_models = gene_models[gene_models['Chromosome'].isin(['1', '2', '3', '4', '5'])]

    prom_coords = []
    for chrom, start, end, strand, gene_id in gene_models.values:
        if strand == '+':
            prom_coords.append([chrom, max(0, start - promoter_size), start])
        else:
            prom_coords.append([chrom, end, end + promoter_size])
    prom_coords = pd.DataFrame(prom_coords)
    return prom_coords

prom_coords_df = get_promoter_coords(promoter_size=promoter_size)
prom_coords_df.to_csv(path_or_buf='data/promoter_coords.bed', index=False, header=False, sep='\t')

inters_results = BedTool.from_dataframe(gwas_hits).intersect('data/promoter_coords.bed', wa=True, wb=True)
inters_results = inters_results.to_dataframe(disable_auto_names=True, header=None)
print(inters_results.head())
print(gwas_hits.shape)
print(inters_results.shape)


def create_dataset(chrom, genome, promoter_coords, promoter_size=1000):
    onehot_dict = {'A': [1, 0, 0, 0],
                   'C': [0, 1, 0, 0],
                   'G': [0, 0, 1, 0],
                   'T': [0, 0, 0, 1]}
    genome = Fasta(genome, as_raw=True, read_ahead=10000, sequence_always_upper=True)
    promoter_coords[[0, 9]] = promoter_coords[[0, 9]].astype('str')
    promoter_coords = promoter_coords[promoter_coords[9] == chrom]

    seqs, snp_seqs, snp_ids = [], [], []
    for coord in promoter_coords.values:
        chrom, start, end = coord[9], coord[10], coord[11]
        snp_ref, snp_alt, snp_pos = coord[3], coord[4], coord[1]-1
        enc_seq = one_hot_encode(genome[chrom][start:end])
        enc_seq_snp = one_hot_encode(genome[chrom][start:end])
        enc_seq_snp[snp_pos - start, :] = onehot_dict[snp_alt]
        if enc_seq.shape[0] == promoter_size:
            seqs.append(enc_seq)
            snp_seqs.append(enc_seq_snp)
            snp_ids.append(f"{chrom}.{snp_pos+1}.{snp_ref}_{snp_alt}")

    return np.array(seqs), np.array(snp_seqs), snp_ids

fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(10, 10), sharex='col')
for row_idx, chrom in enumerate(['1', '2', '3', '4', '5']):
    orig_seq, mut_seq, snp_identifiers = create_dataset(chrom=chrom,
                                                        genome='data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa',
                                                        promoter_coords=inters_results,
                                                        promoter_size=promoter_size)
    model = load_model(f'saved_models/model_chrom_{chrom}_model.h5')
    print(orig_seq.shape, mut_seq.shape)

    predictions_original, predictions_mutated = [], []
    for idx in range(promoter_size // 250):
        preds = np.expand_dims(model.predict(orig_seq[:, 250 * idx:250 * (idx + 1), :]), axis=0)
        preds = preds > 0.5
        preds = preds.astype('int')
        predictions_original.append(preds)
    predictions_original = np.concatenate(predictions_original, axis=0)
    predictions_original = predictions_original.sum(axis=0)

    for idx in range(promoter_size // 250):
        preds = np.expand_dims(model.predict(mut_seq[:, 250 * idx:250 * (idx + 1), :]), axis=0)
        preds = preds > 0.5
        preds = preds.astype('int')
        predictions_mutated.append(preds)
    predictions_mutated = np.concatenate(predictions_mutated, axis=0)
    predictions_mutated = predictions_mutated.sum(axis=0)

    mutation_effect = predictions_original - predictions_mutated
    mutation_effect = pd.DataFrame(mutation_effect, columns=tf_families, index=snp_identifiers)
    print(mutation_effect.head(10))
    sns.heatmap(mutation_effect, cmap='coolwarm', ax=ax[row_idx])
    #for i in range(mutation_effect.shape[1] + 1):
    #    ax[row_idx].axvline(i, color='white', lw=1)
    print(predictions_original.shape, predictions_mutated.shape)

fig.tight_layout()
plt.savefig(f"results/Figures/promoter_snp_effects_on_model_preds.png", bbox_inches='tight',
            dpi=300, format='png')
plt.show()