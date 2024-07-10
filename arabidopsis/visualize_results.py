import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.width = 0

palette = ['#FC4100', '#FFC55A', '#00215E']
def boxplot_performance():
    data = []
    for log_file in os.listdir('results'):
        if log_file.endswith('.log'):
            log_df = pd.read_csv(f'results/{log_file}')
            log_df.sort_values(by='val_loss', ascending=True, inplace=True)
            log_df['case'] = [log_file.split('_')[-1].split('.')[0]]*log_df.shape[0]
            best_val_aupr = log_df[['val_auPR', 'case']].values[0, :].tolist()
            best_val_aupr.append('auPR')

            best_val_auroc = log_df[['val_auROC', 'case']].values[0, :].tolist()
            best_val_auroc.append('auROC')

            best_val_weighted_aupr = log_df[['val_weighted_auPR', 'case']].values[0, :].tolist()
            best_val_weighted_aupr.append('weighted auPR')
            data.append(best_val_aupr)
            data.append(best_val_auroc)
            data.append(best_val_weighted_aupr)

    data = pd.DataFrame(data, columns=['score', 'model type', 'metric'])
    data['region'] = ['genome']*data.shape[0]
    data.replace(to_replace={'di': 'baseline-di', 'si': 'baseline-si'}, inplace=True)

    data_prom = pd.read_csv(filepath_or_buffer='results/prom_evals.csv')
    data_prom.replace(to_replace={'di': 'baseline-di', 'si': 'baseline-si', 'promoters': 'model'}, inplace=True)
    data_prom['region'] = ['promoter']*data_prom.shape[0]
    data = pd.concat([data_prom, data])

    hue_order = sorted(data['model type'].unique())
    order = sorted(data['region'].unique())
    for metric in ['auPR', 'auROC', 'weighted auPR']:
        data_met = data[data['metric'] == metric]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 8))
        g = sns.barplot(data=data_met, y='score', x='region', hue='model type', ax=ax, capsize=0.3,
                        palette=palette, hue_order=hue_order,
                        edgecolor='white', order=order)
        sns.stripplot(y='score', x='region', hue='model type', data=data_met, dodge=True, ax=g, edgecolor='k',
                      linewidth=0.6, palette=palette, alpha=0.5,
                      hue_order=hue_order)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, 1.0)
        plt.savefig(f"results/Figures/performance_multilabel_{metric}.svg", bbox_inches='tight',
                    dpi=300, format='svg')
        plt.show()


boxplot_performance()
