import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.width = 0


def boxplot_performance():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 8))
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
    evals_on_prom = pd.read_csv('results/prom_evals.csv')
    data = pd.concat([data, evals_on_prom], axis=0)
    data.replace(to_replace={'di': 'baseline-di', 'si': 'baseline-si'}, inplace=True)
    hue_order = sorted(data['model type'].unique())
    g = sns.barplot(data=data, y='score', x='metric', hue='model type', ax=ax, capsize=0.3,
                    palette=['#808000', '#9897A9', '#726EFF', '#ff7926'], hue_order=hue_order,
                    edgecolor='white')
    sns.stripplot(y='score', x='metric', hue='model type', data=data, dodge=True, ax=g, edgecolor='k',
                  linewidth=0.6, palette=['#808000', '#9897A9', '#726EFF', '#ff7926'], alpha=0.5,
                  hue_order=hue_order)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.savefig(f"results/Figures/performance_multilabel.svg", bbox_inches='tight', dpi=300, format='svg')
    plt.show()


boxplot_performance()

