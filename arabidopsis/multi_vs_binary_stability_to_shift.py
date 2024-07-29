import pandas as pd
import numpy as np
from pyfaidx import Fasta
from arabidopsis.utils import one_hot_encode
from deeplift.dinuc_shuffle import dinuc_shuffle
from tensorflow.keras.models import load_model
import shap.explainers.deep.deep_tf
import shap
import tensorflow as tf
import os
import matplotlib.pyplot as plt
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
selected_families = ['C2C2dof_tnt', 'AP2EREBP_tnt', 'HB_tnt', 'MYB_tnt', 'bHLH_tnt', 'bZIP_tnt']
genome_path = 'Arabidopsis_thaliana.TAIR10.dna.toplevel.fa'

tf_families_ara = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed',
                              index_col=0, sep='\t', nrows=3).columns.tolist()


def prepare_dataset(family_name, chrom_name, genome, sample_size=5000, shift=0):
    genome = Fasta(filename=f'data/genome/{genome}', as_raw=True, read_ahead=10000,
                   sequence_always_upper=True)
    peaks = pd.read_csv(filepath_or_buffer=f'data/peaks/{family_name}.bed', sep='\t', header=None,
                        dtype={0: str, 1: int, 2: int})
    peaks = peaks[peaks[0] == chrom_name]
    peaks = peaks.sample(n=sample_size, random_state=42)
    seqs = []
    for chrom, start, end in peaks.values:
        midpoint = (start + end)//2
        midpoint += shift
        seq = one_hot_encode(genome[chrom][midpoint-125:midpoint+125])
        if seq.shape[0] == 250:
            seqs.append(seq)
    seqs = np.array(seqs)
    return seqs


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


def compute_shap_scores(seqs, model, tf_idx):
    shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
    shap.explainers.deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers.deep.deep_tf.linearity_1d(0)
    dinuc_shuff_explainer = shap.DeepExplainer(
        (model.input, model.layers[-2].output[:, tf_idx]),
        data=dinuc_shuffle_several_times,
        combine_mult_and_diffref=combine_mult_and_diffref)
    hypothetical_scores = dinuc_shuff_explainer.shap_values(seqs)
    actual_scores = hypothetical_scores * seqs
    actual_scores = actual_scores.sum(axis=-1)
    return actual_scores


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


palettes = ['#26355D', '#AF47D2', '#FF8F00']
selected_chrom = '1'
fig1, ax1 = plt.subplots(nrows=6, ncols=3, figsize=(14, 8))
for shift_idx, bp_shift in enumerate([0, -50, 50]):

    coords = [(0, shift_idx), (1, shift_idx), (2, shift_idx), (3, shift_idx), (4, shift_idx), (5, shift_idx)]
    for family, fam_idx in zip(selected_families, range(len(selected_families))):
        print(family, bp_shift)
        enc_seqs = prepare_dataset(family_name=family, genome=genome_path,
                                   chrom_name=selected_chrom,
                                   sample_size=15000, shift=bp_shift)
        # Predictions binary classifier with genome negatives
        binary_gn = load_model(f'saved_models/{family}_chrom_{selected_chrom}_gn.h5')
        preds_binary_gn = binary_gn.predict(enc_seqs)
        preds_binary_gn = preds_binary_gn.ravel() > 0.5
        preds_binary_gn = preds_binary_gn.astype("int")

        # Predictions binary classifier with di-nucleotide shuffled sequence negatives
        binary_dsn = load_model(f'saved_models/{family}_chrom_{selected_chrom}.h5')
        preds_binary_dsn = binary_dsn.predict(enc_seqs)
        preds_binary_dsn = preds_binary_dsn.ravel() > 0.5
        preds_binary_dsn = preds_binary_dsn.astype("int")

        # Predictions multi-label classifier
        multi_label = load_model(f'saved_models/model_chrom_{selected_chrom}_model.h5')
        preds_multi = multi_label.predict(enc_seqs)[:, tf_families_ara.index(family)]
        preds_multi = preds_multi.ravel() > 0.5
        preds_multi = preds_multi.astype("int")

        idx_pred_correct_by_both = np.where(preds_binary_gn+preds_binary_dsn+preds_multi == 3)[0]
        sel_enc_seqs = enc_seqs[idx_pred_correct_by_both][:500]
        print(sel_enc_seqs.shape)

        # computing importance scores
        binary_gn_scores = compute_shap_scores(seqs=sel_enc_seqs, model=binary_gn, tf_idx=0)
        binary_dsn_scores = compute_shap_scores(seqs=sel_enc_seqs, model=binary_dsn, tf_idx=0)
        multi_scores = compute_shap_scores(seqs=sel_enc_seqs, model=multi_label,
                                           tf_idx=tf_families_ara.index(family))

        print(f'Done with {family}')
        print(binary_gn_scores.shape)
        print(binary_dsn_scores.shape)
        print(multi_scores.shape)

        # plot
        binary_gn_scores = normalize(binary_gn_scores.mean(axis=0))
        binary_dsn_scores = normalize(binary_dsn_scores.mean(axis=0))
        multi_scores = normalize(multi_scores.mean(axis=0))

        print(binary_gn_scores.shape, binary_dsn_scores.shape, multi_scores.shape)

        ax1[coords[fam_idx]].plot(np.arange(250), binary_dsn_scores, color=palettes[2])
        ax1[coords[fam_idx]].plot(np.arange(250), binary_gn_scores, color=palettes[1])
        ax1[coords[fam_idx]].plot(np.arange(250), multi_scores, color=palettes[0])
        ax1[coords[fam_idx]].yaxis.grid(True, alpha=0.3)
        ax1[coords[fam_idx]].xaxis.grid(True, alpha=0.3)
        ax1[coords[fam_idx]].set_axisbelow(True)
        ax1[coords[fam_idx]].set_title(family)
        ax1[coords[fam_idx]].spines[['right', 'top']].set_visible(False)

fig1.tight_layout()
plt.savefig(f"results/Figures/binary_vs_multilabel_map_shift_stab.svg", bbox_inches='tight', dpi=300, format='svg')
plt.show()

