import pandas as pd
import numpy as np
import pyranges as pr
from pybedtools import BedTool
from pyfaidx import Fasta
from utils import one_hot_encode
from tensorflow.keras.models import load_model
from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from deeplift.visualization.viz_sequence import plot_weights_given_ax
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
pd.options.display.width=0
sns.set_context(context="paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})


def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example, seed=1234):
    assert len(list_containing_input_modes_for_an_example) == 1
    onehot_seq = list_containing_input_modes_for_an_example[0]
    rng = np.random.RandomState(seed)
    to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(50)])

    return [to_return]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    for l in range(len(mult)):
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape) == 2
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None, :, :] - bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference * mult[l]
            projected_hypothetical_contribs[:, :, i] = np.sum(hypothetical_contribs, axis=-1)
        to_return.append(np.mean(projected_hypothetical_contribs, axis=0))
    return to_return


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
        chrom, start, end, gene_id, pheno = coord[9], coord[10], coord[11], coord[12], coord[8]
        snp_ref, snp_alt, snp_pos = coord[3], coord[4], coord[1]-1
        enc_seq = one_hot_encode(genome[chrom][start:end])
        enc_seq_snp = one_hot_encode(genome[chrom][start:end])
        enc_seq_snp[snp_pos - start, :] = onehot_dict[snp_alt]
        if enc_seq.shape[0] == promoter_size:
            seqs.append(enc_seq)
            snp_seqs.append(enc_seq_snp)
            snp_ids.append(f"{chrom}.{snp_pos+1}.{snp_ref}_{snp_alt}.{gene_id}.{pheno}")

    return np.array(seqs), np.array(snp_seqs), snp_ids


chroms = ['2', '5']
phenos = ['Mo98', 'clim-bio11']
tfs = ['bZIP_tnt', 'BES1_tnt']

for chr_name, pheno, tf in zip(chroms, phenos, tfs):
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
    print(gwas_hits['study.phenotype.name'].unique())
    gwas_hits = gwas_hits[gwas_hits['study.phenotype.name'] == pheno]
    def get_promoter_coords(promoter_size=1000):
        gene_models = pr.read_gtf('data/annotation/Arabidopsis_thaliana.TAIR10.59.gtf', as_df=True)
        gene_models = gene_models[gene_models['Feature'] == 'gene']
        gene_models = gene_models[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
        gene_models = gene_models[gene_models['Chromosome'].isin(['1', '2', '3', '4', '5'])]

        prom_coords = []
        for chrom, start, end, strand, gene_id in gene_models.values:
            if strand == '+':
                prom_coords.append([chrom, max(0, start - promoter_size), start, gene_id])
            else:
                prom_coords.append([chrom, end, end + promoter_size, gene_id])
        prom_coords = pd.DataFrame(prom_coords)
        return prom_coords

    prom_coords_df = get_promoter_coords(promoter_size=promoter_size)
    prom_coords_df.to_csv(path_or_buf='data/promoter_coords.bed', index=False, header=False, sep='\t')

    inters_results = BedTool.from_dataframe(gwas_hits).intersect('data/promoter_coords.bed', wa=True, wb=True)
    inters_results = inters_results.to_dataframe(disable_auto_names=True, header=None)
    print(inters_results.head())
    print(gwas_hits.shape)
    print(inters_results.shape)


    chrom = chr_name
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
    print(mutation_effect.head(50))
    print(mutation_effect.shape)

    tf_family_to_investigate = tf
    tf_family_to_investigate_idx = tf_families.index(tf_family_to_investigate)
    idx_with_snp_effect = np.where(mutation_effect.values[:, tf_family_to_investigate_idx] != 0)[0]
    effect_type = mutation_effect.values[idx_with_snp_effect, tf_family_to_investigate_idx]
    print(effect_type)
    seqs_to_investigate_orig, seqs_to_investigate_mut = orig_seq[idx_with_snp_effect], mut_seq[idx_with_snp_effect]
    genes_to_investigate = [mutation_effect.index.tolist()[i].split('.')[-2] for i in idx_with_snp_effect]
    changes_to_investigate = [mutation_effect.index.tolist()[i].split('.')[-3] for i in idx_with_snp_effect]
    snp_ids = [mutation_effect.index.tolist()[i] for i in idx_with_snp_effect]
    original_seq_with_effect, mutated_seq_with_effect, coords_with_mutation, save_names = [], [], [], []
    probs_orig, probs_mut = [], []
    for idx in range(promoter_size // 250):
        preds_orig_probs = model.predict(seqs_to_investigate_orig[:, 250 * idx:250 * (idx + 1), :])
        preds_orig = preds_orig_probs.copy()
        preds_orig = preds_orig > 0.5
        preds_orig = preds_orig.astype('int')

        # Mutated seqs
        preds_mut_probs = model.predict(seqs_to_investigate_mut[:, 250 * idx:250 * (idx + 1), :])
        preds_mut = preds_mut_probs.copy()
        preds_mut = preds_mut > 0.5
        preds_mut = preds_mut.astype('int')

        for seq_idx in range(len(idx_with_snp_effect)):
            if preds_orig[seq_idx, tf_family_to_investigate_idx] != preds_mut[seq_idx, tf_family_to_investigate_idx]:
                print(f"Mutation at: {idx}: {250 * idx}:{250 * (idx + 1)} - Gene ID {genes_to_investigate[seq_idx]}")
                original_seq_with_effect.append(seqs_to_investigate_orig[seq_idx, 250 * idx:250 * (idx + 1), :])
                probs_orig.append(preds_orig_probs[seq_idx, tf_family_to_investigate_idx])
                mutated_seq_with_effect.append(seqs_to_investigate_mut[seq_idx, 250 * idx:250 * (idx + 1), :])
                probs_mut.append(preds_mut_probs[seq_idx, tf_family_to_investigate_idx])
                mut_pos = np.where(np.sum(abs(seqs_to_investigate_orig[seq_idx, 250 * idx:250 * (idx + 1), :] -
                                              seqs_to_investigate_mut[seq_idx, 250 * idx:250 * (idx + 1), :]), axis=1))[0][0]
                coords_with_mutation.append(f"Mutation: Gene ID {genes_to_investigate[seq_idx]}: Ref_ALT: {changes_to_investigate[seq_idx]}: Pos: {mut_pos}")
                save_names.append(snp_ids[seq_idx])

    original_seq_with_effect = np.array(original_seq_with_effect)
    mutated_seq_with_effect = np.array(mutated_seq_with_effect)
    print(original_seq_with_effect.shape, mutated_seq_with_effect.shape, len(coords_with_mutation))
    print(coords_with_mutation)

    # Compute shap importance
    shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
    shap.explainers.deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers.deep.deep_tf.linearity_1d(0)
    dinuc_shuff_explainer = shap.DeepExplainer(
        (model.input, model.layers[-2].output[:, tf_family_to_investigate_idx]),
        data=dinuc_shuffle_several_times,
        combine_mult_and_diffref=combine_mult_and_diffref)

    # Original Seq
    hypothetical_scores_orig = dinuc_shuff_explainer.shap_values(original_seq_with_effect)
    actual_scores_orig = hypothetical_scores_orig * original_seq_with_effect
    # Mutated Seq
    hypothetical_scores_mutate = dinuc_shuff_explainer.shap_values(mutated_seq_with_effect)
    actual_scores_mutate = hypothetical_scores_mutate * mutated_seq_with_effect

    print(actual_scores_mutate.shape, actual_scores_orig.shape)
    for idx in range(len(coords_with_mutation)):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 4))
        plot_weights_given_ax(ax=ax[0], array=actual_scores_orig[idx],
                                      height_padding_factor=0.2, length_padding=1.0, subticks_frequency=25,
                                      highlight={})
        ax[0].set_title(f"Original: Predicted Prob: {round(float(probs_orig[idx]), 3)}")
        plot_weights_given_ax(ax=ax[1], array=actual_scores_mutate[idx],
                              height_padding_factor=0.2, length_padding=1.0, subticks_frequency=25,
                              highlight={})
        ax[1].set_title(f"Mutated: {coords_with_mutation[idx]}: Predicted Prob: {round(float(probs_mut[idx]), 3)}")
        fig.tight_layout()
        plt.savefig(f"results/Figures/SNP_{snp_ids[idx]}_{tf}.svg", bbox_inches='tight',
                    dpi=300, format='svg')
        plt.show()
