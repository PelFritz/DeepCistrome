import pandas as pd
import altair as alt
pd.options.display.width=0

mercator_output = pd.read_csv(filepath_or_buffer="data/mercator4_output.txt", sep="\t")
mercator_output.dropna(inplace=True, how="any")
mercator_output['IDENTIFIER'] = mercator_output.IDENTIFIER.str.upper()
mercator_output['gene_id'] = [x.replace("'", '').split('.')[0] for x in mercator_output['IDENTIFIER']]
mercator_output['mercator_bin'] = [x.replace("'", '').split('.')[0] for x in mercator_output['NAME']]
mercator_output = mercator_output[['gene_id', 'mercator_bin']]
mercator_output = mercator_output[mercator_output['mercator_bin'] != 'No Mercator4 annotation']

predicted_cluters = pd.read_csv(filepath_or_buffer="data/prom_term_predictions.csv")
predicted_cluters.rename({'Unnamed: 0': 'gene_id'}, axis='columns', inplace=True)
predicted_cluters = predicted_cluters[['gene_id', 'cluster', 'cluster_col']]

data = mercator_output.merge(predicted_cluters, how='inner', on='gene_id')
data.drop_duplicates(subset=['gene_id', 'mercator_bin', 'cluster'], keep='first', inplace=True)
data = data[data['cluster'] != 0]
print(data['cluster'].unique())
data.replace({0: 'n', 1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
              6: 'VI', 7: 'VII', 8: 'VIII', 9: 'XI', 10: 'X',
              11: 'XI', 12: 'XII', 13: 'VIII', 14: 'XIV'}, inplace=True)
data['cluster_mercator_total'] = data.groupby('cluster')['mercator_bin'].transform('count')
data['Percentage'] = data.groupby(['cluster', 'mercator_bin']).transform('count')['cluster_col'] / data['cluster_mercator_total']
#data.sort_values(by=['cluster'], inplace=True)
print(data.head())
#print(data.groupby(['cluster', 'mercator_bin']).transform('count')['cluster_col'] / data['cluster_mercator_total'])
#chart = alt.Chart(data=data).mark_point(filled=True).encode(x='cluster', y='mercator_bin',
#                                                            size='count(cluster)', color='cluster_col')
chart = alt.Chart(data=data).mark_point(filled=True).encode(
    y='mercator_bin', x='cluster', color=alt.Color('Percentage:Q').scale(scheme='reds'), size='Percentage:Q')
chart.save('results/Figures/enrichment_analysis.png')