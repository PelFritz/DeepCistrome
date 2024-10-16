import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.width = 0
sns.set_context(context="paper", rc={"font.size":5,"axes.titlesize":8,"axes.labelsize":8})
overlap_mat = pd.read_csv(filepath_or_buffer='data/overlap_matrix.bed', sep='\t', nrows=2, index_col=0)
tf_families = overlap_mat.columns.tolist()
data = pd.read_csv(filepath_or_buffer='data/snp_effect_on_cis_regions.csv', index_col=0)
phenotypes = [i.split('.')[-1] for i in data.index.tolist()]
phenotypes = ['climate related' if x.startswith('cli') else x for x in phenotypes]
phenotype_name = ['avrB', 'avrPphB', 'avrRpm1', 'avrRpt2',
                  'M216T665','M130T666', 'M172T666',
                  'MRLpTRS75',
                  'GR21', 'GR21 cold', 'GR21 warm',
                  'Cd111', 'Mo98', 'Na23', 'B11', 'Li7',
                  'FT10', 'FT16', 'FT22', 'FRI',
                  'Emco5', 'Emoy*', 'LY ', 'LES',
                  'LFS GH', 'MT GH',
                  'LD', 'SD', '0W GH FT', 'DTFplantingSummerLocSweden2009',
                  'YEL ', 'Inter-specific pollination of Arabidopsis thaliana and Malcolmia littorea',
                  'delta 13C', 'delta 13C 261',
                  'YieldPlantingSummer2009',
                  'Size sweden 2009 (1st experiment)', 'Size sweden 2009 (2nd experiment)', 'SizeMainEffect2009',
                  'ScalingExponent',
                  'Germ in dark',
                  'TRS125',
                  'LN22', 'Co59',
                  '0W GH LN', '2W',
                  'DTF sweden 2009 (2nd experiment)',
                  'SizeLocSweden2009']
phenotypes_grp = ['bacteria resistance', 'bacteria resistance', 'bacteria resistance', 'bacteria resistance',
                  'metabolite content', 'metabolite content', 'metabolite content',
                  'root morphology',
                  'seed dormancy', 'seed dormancy', 'seed dormancy',
                  'ion concentration', 'ion concentration', 'ion concentration', 'ion concentration', 'ion concentration',
                  'days to flowering', 'days to flowering', 'days to flowering', 'days to flowering',
                  'protist disease resistance', 'protist disease resistance', 'leaf necrosis', 'leaf necrosis',
                  'reproductive growth time', 'reproductive growth time',
                  'days to flowering', 'days to flowering', 'days to flowering', 'days to flowering',
                  'leaf chlorosis', 'hybrid incompatibility',
                  'stomatal process', 'stomatal process',
                  'seed weight',
                  '', '', '',
                  'shoot system growth and development',
                  'germination ability in dark',
                  'root morphology',
                  'leaf number', 'ion concentration', 'days to flowering', 'days to flowering',
                  'days to flowering',
                  '']
phenotype_to_group = {k:v for k, v in zip(phenotype_name, phenotypes_grp)}
phenotypes = [phenotype_to_group[i] if i in phenotype_to_group.keys() else i for i in phenotypes]
data['phenotype'] = phenotypes
data['chrom'] = [x.split('.')[0] for x in data.index.tolist()]
data = data[data['phenotype'] != 'climate related']
data = data[data['chrom'] != '4']
data.sort_values(by='phenotype', inplace=True)
print(data.head())
print(data.shape)
size_of_all_overlapping_snps = data.shape[0]
data = data[data[tf_families].abs().sum(axis=1) != 0]
size_of_all_overlapping_snps_with_effect = data.shape[0]

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex='col',
                       gridspec_kw={'height_ratios': [2, 2, 2, 1]})
for idx, chrom in zip([0, 1, 2, 3], ['1', '2', '3', '5']):
    data_chrom = data.copy()
    data_chrom = data_chrom[data_chrom['chrom'] == chrom]
    sns.heatmap(data=data_chrom[tf_families], ax=ax[idx], cmap='coolwarm',
                yticklabels=data_chrom['phenotype'])
    ax[idx].set_title(f"Chr{chrom}")
fig.tight_layout()
plt.savefig(f"results/Figures/snp_effects_on_cis_regions_subset.svg", bbox_inches='tight', dpi=300, format='svg')
plt.show()

fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
sns.countplot(data=data, ax=ax2, color='k', y='phenotype', stat='percent',
              order=data['phenotype'].value_counts(ascending=True).index)
ax2.spines[['right', 'top']].set_visible(False)
ax2.set_xlim(0, 100)
fig2.tight_layout()
plt.savefig(f"results/Figures/snp_effects_on_cis_regions_count_plot_subset.svg", bbox_inches='tight', dpi=300, format='svg')
plt.savefig(f"results/Figures/snp_effects_on_cis_regions_count_plot_subset.png", bbox_inches='tight', dpi=300,
            format='png')

plt.show()
print(size_of_all_overlapping_snps_with_effect/size_of_all_overlapping_snps)
