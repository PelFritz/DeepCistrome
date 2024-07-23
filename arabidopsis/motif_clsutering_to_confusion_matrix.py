import h5py
import pandas as pd
from scipy.cluster.hierarchy import linkage
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.width = 0
modisco_motifs = h5py.File(name='data/predictive_motifs.h5', mode='r')['motifs']
tf_fam = pd.read_csv(filepath_or_buffer='data/tfs.csv', sep='\t').values.ravel()

comparison = pd.read_csv(filepath_or_buffer='data/computed_motif_distances.csv')
print(comparison.head())
linkage = linkage(comparison['distance'], optimal_ordering=True, method='single')
confusion_matrix = pd.read_csv(filepath_or_buffer='data/diagonal_masked_cm.csv', index_col=0)
confusion_matrix = confusion_matrix.loc[tf_fam, tf_fam]
print(confusion_matrix.head())
sns.clustermap(confusion_matrix, col_linkage=linkage, row_linkage=linkage, cmap='Reds')
plt.savefig(f"results/Figures/masked_confusion_mat.svg", bbox_inches='tight',
            dpi=300, format='svg')
plt.show()