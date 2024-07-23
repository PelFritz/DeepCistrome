import pandas as pd
import matplotlib.pyplot as plt
from pyfaidx import Fasta
from arabidopsis.utils import one_hot_encode
import numpy as np
from tensorflow.keras.models import load_model
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors

tf_families_ara = pd.read_csv(filepath_or_buffer='arabidopsis/data/overlap_matrix.bed',
                              index_col=0, sep='\t', nrows=3).columns.tolist()
tf_families_zm = pd.read_csv(filepath_or_buffer='zmays/data/overlap_matrix.bed',
                             index_col=0, sep='\t', nrows=3).columns.tolist()
print(tf_families_ara)
print(tf_families_zm)
zm_ara_tf_family_pairs = [['C2C2-DOF', 'C2C2dof_tnt', 0],
                          ['C3H', 'C3H_tnt', 1],
                          ['EREB', 'AP2EREBP_tnt', 2],
                          ['HB', 'HB_tnt', 3],
                          ['MYB', 'MYB_tnt', 4],
                          ['NAC', 'NAC_tnt', 5],
                          ['bHLH', 'bHLH_tnt', 6],
                          ['bZIP', 'bZIP_tnt', 7]]

fig, ax = plt.subplots(nrows=8, ncols=2, figsize=(8, 16), sharey='row', sharex='col')
fig.add_subplot(111, frameon=False)
ax[0, 0].set_title('Zea mays', style='italic')
ax[0, 1].set_title('Arabidopsis thaliana', style='italic')

def prepare_dataset(family_name, chrom_name, genome, species):
    genome = Fasta(filename=f'{species}/data/genome/{genome}', as_raw=True,
                   read_ahead=10000, sequence_always_upper=True)
    peaks = pd.read_csv(filepath_or_buffer=f'{species}/data/peaks/{family_name}.bed', sep='\t', header=None,
                        dtype={0: str, 1: int, 2: int})
    peaks = peaks[peaks[0] == chrom_name]
    print(peaks.head())
    seqs = []
    for chrom, start, end in peaks.values:
        midpoint = (start + end)//2
        seq = one_hot_encode(genome[chrom][midpoint-125:midpoint+125])
        if seq.shape[0] == 250:
            seqs.append(seq)
    seqs = np.array(seqs)
    return seqs

species_name = ['zmays', 'arabidopsis']
genomes = ['Zea_mays.B73_RefGen_v4.dna.toplevel.fa', 'Arabidopsis_thaliana.TAIR10.dna.toplevel.fa']

for idx in range(len(species_name)):

    for meta_info in zm_ara_tf_family_pairs:
        zm_name, ara_name, row_idx = meta_info[0], meta_info[1], meta_info[2]
        preds_zm_all, preds_ara_all = [], []
        for chromosome in ['1']:  # Limiting to chromosome 1 to reduce file output size
            print(f'Handling Chromosome: {chromosome}')
            enc_seqs = prepare_dataset(family_name=meta_info[idx], chrom_name=chromosome, genome=genomes[idx],
                                       species=species_name[idx])
            if species_name[idx] == 'zmays':
                print('Predicting ZM')
                zm_model = load_model(f'zmays/saved_models/model_chrom_{chromosome}_model.h5')
                pred_prob_zm = zm_model.predict(enc_seqs)[:, tf_families_zm.index(zm_name)]
                pred_prob_ara = []
                for a_chrom in ['1', '2', '3', '4', '5']:
                    ara_model = load_model(f'arabidopsis/saved_models/model_chrom_{a_chrom}_model.h5')
                    pred_prob_ara_chrom = ara_model.predict(enc_seqs)[:, tf_families_ara.index(ara_name)]
                    pred_prob_ara.append(pred_prob_ara_chrom)
                pred_prob_ara = np.array(pred_prob_ara).transpose().max(axis=1)
                assert len(pred_prob_zm) == len(pred_prob_ara)
                preds_zm_all.extend(pred_prob_zm)
                preds_ara_all.extend(pred_prob_ara)
            else:
                print('Predicting ARA')
                ara_model = load_model(f'arabidopsis/saved_models/model_chrom_{chromosome}_model.h5')
                pred_prob_ara = ara_model.predict(enc_seqs)[:, tf_families_ara.index(ara_name)]
                pred_prob_zm = []
                for z_chrom in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                    zm_model = load_model(f'zmays/saved_models/model_chrom_{z_chrom}_model.h5')
                    pred_prob_zm_chrom = zm_model.predict(enc_seqs)[:, tf_families_zm.index(zm_name)]
                    pred_prob_zm.append(pred_prob_zm_chrom)
                pred_prob_zm = np.array(pred_prob_zm).transpose().max(axis=1)
                assert len(pred_prob_zm) == len(pred_prob_ara)
                preds_zm_all.extend(pred_prob_zm)
                preds_ara_all.extend(pred_prob_ara)
        xy = np.vstack((preds_zm_all, preds_ara_all))
        z = gaussian_kde(xy)(xy)
        ax[row_idx, idx].scatter(y=preds_ara_all, x=preds_zm_all, s=2, c=z)
        #ax[row_idx, idx].hexbin(y=preds_ara_all, x=preds_zm_all, gridsize=(10, 10))
        #ax[row_idx, idx].hist2d(y=preds_ara_all, x=preds_zm_all, bins=30, cmap='viridis',
        #                        norm=mcolors.PowerNorm(gamma=0.8), range=[[0, 1],[0, 1]])
        ax[row_idx, idx].set_axisbelow(True)
        ax[row_idx, idx].vlines(0.5, 0, 1, linestyles='dashed', color='k')
        ax[row_idx, idx].hlines(0.5, 0, 1, linestyles='dashed', color='k')
        ax[row_idx, 1].text(1.05, 0.5, zm_name, rotation=-90, ha='left', va='center')
        print(f'Done with {zm_name}')


plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("predicted probability by Zea")
plt.ylabel("predicted probability by Arabidopsis")
fig.tight_layout()

plt.savefig(f"arabidopsis/results/Figures/tf_cross_predictions_sc.pdf", bbox_inches='tight', dpi=300, format='pdf')
plt.savefig(f"arabidopsis/results/Figures/tf_cross_predictions_sc.png", bbox_inches='tight', dpi=300, format='png')

plt.savefig(f"zmays/results/Figures/tf_cross_predictions_sc.png", bbox_inches='tight', dpi=300, format='png')
plt.savefig(f"zmays/results/Figures/tf_cross_predictions_sc.png", bbox_inches='tight', dpi=300, format='png')
plt.show()
