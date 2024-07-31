import pandas as pd
import altair as alt
pd.options.display.width=0

mercator_data = pd.read_csv(filepath_or_buffer="data/mercator4_output.txt", sep="\t")
mercator_data.dropna(inplace=True, how="any")
mercator_data['IDENTIFIER'] = mercator_data.IDENTIFIER.str.upper()
mercator_data['gene_id'] = [x.replace("'", '').split('.')[0] for x in mercator_data['IDENTIFIER']]
mercator_data['mercator_main_bin'] = [x.replace("'", '').split('.')[0] for x in mercator_data['NAME']]
mercator_data = mercator_data[mercator_data['mercator_main_bin'] != 'No Mercator4 annotation']

for bio_process in mercator_data['mercator_main_bin'].unique():
    print(bio_process)
    for idx in [0, 1, 2]:
        print(f'Processing level: {idx}')
        mercator_output = mercator_data.copy()
        if idx > 0:
            mercator_output = mercator_output[mercator_output['mercator_main_bin'] == bio_process]
            mercator_output['mercator_bin'] = [x.replace("'", '').split('.')[idx] for x in mercator_output['NAME']]
            mercator_output = mercator_output[['gene_id', 'mercator_bin']]

            predicted_cluters = pd.read_csv(filepath_or_buffer="data/prom_term_predictions.csv")
            predicted_cluters.rename({'Unnamed: 0': 'gene_id'}, axis='columns', inplace=True)
            predicted_cluters = predicted_cluters[['gene_id', 'cluster', 'cluster_col']]

            data = mercator_output.merge(predicted_cluters, how='inner', on='gene_id')
            data.drop_duplicates(subset=['gene_id', 'mercator_bin', 'cluster'], keep='first', inplace=True)
            data = data[data['cluster'] != 0]
            print(data['cluster'].unique())
            data.replace({0: 'n', 1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
                          6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X',
                          11: 'XI', 12: 'XII', 13: 'XIII', 14: 'XIV'}, inplace=True)
            data['cluster_mercator_total'] = data.groupby('cluster')['gene_id'].transform('nunique')
            data['mercator_total'] = data.groupby(['cluster', 'mercator_bin'])['mercator_bin'].transform('count')
            data['Percentage'] = data['mercator_total'] / data['cluster_mercator_total']
        else:
            mercator_output['mercator_bin'] = mercator_output['mercator_main_bin']

            mercator_output = mercator_output[['gene_id', 'mercator_bin', 'mercator_main_bin']]

            predicted_cluters = pd.read_csv(filepath_or_buffer="data/prom_term_predictions.csv")
            predicted_cluters.rename({'Unnamed: 0': 'gene_id'}, axis='columns', inplace=True)
            predicted_cluters = predicted_cluters[['gene_id', 'cluster', 'cluster_col']]

            data = mercator_output.merge(predicted_cluters, how='inner', on='gene_id')
            data.drop_duplicates(subset=['gene_id', 'mercator_bin', 'cluster'], keep='first', inplace=True)
            data = data[data['cluster'] != 0]
            print(data['cluster'].unique())
            data.replace({0: 'n', 1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
                          6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X',
                          11: 'XI', 12: 'XII', 13: 'XIII', 14: 'XIV'}, inplace=True)
            data['cluster_mercator_total'] = data.groupby('cluster')['gene_id'].transform('nunique')
            data['mercator_total'] = data.groupby(['cluster', 'mercator_bin'])['mercator_bin'].transform('count')
            data['Percentage'] = data['mercator_total'] / data['cluster_mercator_total']
            data = data[data['mercator_main_bin'] == bio_process]

        print(data.head())
        print(mercator_output.head())
        chart = alt.Chart(data=data).mark_point(filled=True).encode(
            y='mercator_bin', x=alt.X('cluster', sort=['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV']),
            color=alt.Color('Percentage:Q').scale(scheme='reds'), size='Percentage:Q')
        chart.save(f"results/Figures/{'_'.join(bio_process.split(' '))}_enrichment_analysis_level_{idx}.svg")
