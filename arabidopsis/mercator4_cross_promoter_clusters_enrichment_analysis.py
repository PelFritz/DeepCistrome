import pandas as pd
import altair as alt
pd.options.display.width=0

mercator_data = pd.read_csv(filepath_or_buffer="data/mercator4_output.txt", sep="\t")
mercator_data.dropna(inplace=True, how="any")
mercator_data['IDENTIFIER'] = mercator_data.IDENTIFIER.str.upper()
mercator_data['gene_id'] = [x.replace("'", '').split('.')[0] for x in mercator_data['IDENTIFIER']]
mercator_data['mercator_main_bin'] = [x.replace("'", '').split('.')[0] for x in mercator_data['NAME']]
mercator_data = mercator_data[mercator_data['mercator_main_bin'] != 'No Mercator4 annotation']

for level in [1, 2]:
    network_data = []
    for mercator_grp in mercator_data.groupby(['mercator_main_bin']):
        mercator_grp_df = mercator_grp[1]
        mercator_grp_df['mercator_main_bin_sub_bin'] = [x.replace("'", '').split('.')[level]
                                                        for x in mercator_grp_df['NAME']]
        mercator_grp_df = mercator_grp_df[['gene_id', 'mercator_main_bin_sub_bin', 'mercator_main_bin']]
        predicted_clusters = pd.read_csv(filepath_or_buffer="data/prom_term_predictions.csv")
        predicted_clusters.rename({'Unnamed: 0': 'gene_id'}, axis='columns', inplace=True)
        predicted_clusters = predicted_clusters[['gene_id', 'cluster', 'cluster_col']]
        data = mercator_grp_df.merge(predicted_clusters, how='inner', on='gene_id')
        data = data[data['cluster'] != 0]
        cluster_code = {0: 'n', 1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
                        6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X',
                        11: 'XI', 12: 'XII', 13: 'XIII', 14: 'XIV'}
        data.replace(cluster_code, inplace=True)
        for grp_sub_bin in data.groupby(['mercator_main_bin_sub_bin']):
            enrich_dict = grp_sub_bin[1]['cluster'].value_counts().to_dict()
            for v in cluster_code.values():
                if v not in enrich_dict.keys():
                    enrich_dict[v] = 0
            enrich_dict['mercator_sub_bin'] = grp_sub_bin[0][0]
            enrich_dict['mercator_main_bin'] = mercator_grp[0][0]
            network_data.append(enrich_dict)

    network_data = pd.DataFrame.from_records(data=network_data)
    network_data.drop(columns=['n'], inplace=True)
    network_data_cat_cols = network_data[['mercator_main_bin', 'mercator_sub_bin']]
    network_data_num_cols = network_data.drop(columns=['mercator_main_bin', 'mercator_sub_bin'])

    network_data_num_perc = pd.concat([network_data_cat_cols,
                                       network_data_num_cols.div(network_data_num_cols.sum(axis=1), axis=0)], axis=1)
    network_data_num_count = pd.concat([network_data_cat_cols, network_data_num_cols], axis=1)

    for main_bin in network_data_num_count['mercator_main_bin'].unique():
        data_num_perc = network_data_num_perc[network_data_num_perc['mercator_main_bin'] == main_bin]
        data_num_count = network_data_num_count[network_data_num_count['mercator_main_bin'] == main_bin]
        data_num_perc = pd.melt(data_num_perc,
                                id_vars='mercator_sub_bin',
                                value_vars=list(data_num_perc.columns[2:]),
                                var_name='cluster',
                                value_name='Percentage')
        data_num_count = pd.melt(data_num_count,
                                id_vars='mercator_sub_bin',
                                value_vars=list(data_num_count.columns[2:]),
                                var_name='cluster',
                                value_name='Count')
        data_num_count = data_num_count.merge(data_num_perc, how='inner', on=['mercator_sub_bin', 'cluster'])
        data_num_count.sort_values(by='mercator_sub_bin', ascending=True, inplace=True)
        print(data_num_count.head(50))
        chart = alt.Chart(data=data_num_count).mark_point(filled=True).encode(
            y='mercator_sub_bin', x=alt.X('cluster',
                                      sort=['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII',
                                            'XIII', 'XIV']),
            color=alt.Color('Percentage:Q').scale(scheme='purplered'), size='Percentage:Q')
        chart.save(f"results/Figures/{'_'.join(main_bin.split(' '))}_enrichment_analysis_level_{level}_mercator_normalised.svg")