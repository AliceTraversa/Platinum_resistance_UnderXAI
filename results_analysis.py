import pandas as pd
import scipy.stats as stats
import itertools
from statsmodels.stats.multitest import multipletests

def load_all_metrics(file_list):
    dfs = [pd.read_csv(f) for f in file_list]
    return pd.concat(dfs, ignore_index=True)

def friedman_and_wilcoxon(df, metric):
    results = []
    for cutoff in df['cutoff'].unique():
        for feature in df['features'].unique():
            sub = df[(df['cutoff'] == cutoff) & (df['features'] == feature) & (df['metric'] == metric)]
            pivot = sub.pivot_table(index='fold', columns='model', values='value')

            if pivot.shape[1] < 3:
                continue

            stat, p = stats.friedmanchisquare(*[pivot[col] for col in pivot.columns])
            print(f"[Friedman] cutoff={cutoff}, feature={feature}, metric={metric}, p={p:.4f}")

            pairs = list(itertools.combinations(pivot.columns, 2))
            raw_pvals = []
            comparisons = []

            for m1, m2 in pairs:
                try:
                    _, pval = stats.wilcoxon(pivot[m1], pivot[m2])
                    raw_pvals.append(pval)
                    comparisons.append((cutoff, feature, m1, m2, pval))
                except:
                    continue

            if raw_pvals:
                _, pvals_corr, _, _ = multipletests(raw_pvals, method='holm')
                for (cutoff, feature, m1, m2, pval), pcorr in zip(comparisons, pvals_corr):
                    results.append({
                        'cutoff': cutoff,
                        'features': feature,
                        'model1': m1,
                        'model2': m2,
                        'metric': metric,
                        'p_value': pval,
                        'p_value_corrected': pcorr,
                        'significant': pcorr < 0.05
                    })

    return pd.DataFrame(results)

def full_vs_reduced(df, metric):
    results = []
    for cutoff in df['cutoff'].unique():
        for model in df['model'].unique():
            full_vals = df[(df['cutoff'] == cutoff) & (df['features'] == 'full') & (df['model'] == model) & (df['metric'] == metric)]
            reduced_vals = df[(df['cutoff'] == cutoff) & (df['features'] == 'reduced') & (df['model'] == model) & (df['metric'] == metric)]

            if len(full_vals) == len(reduced_vals) and len(full_vals) > 0:
                try:
                    _, pval = stats.wilcoxon(full_vals['value'], reduced_vals['value'])
                    results.append({
                        'cutoff': cutoff,
                        'model': model,
                        'metric': metric,
                        'p_value': pval,
                        'significant': pval < 0.05
                    })
                except:
                    continue
    return pd.DataFrame(results)

def compare_models(df, metric, model_a, model_b, cutoff='12m', feature='reduced'):
    a_vals = df[(df['cutoff'] == cutoff) & (df['features'] == feature) &
                (df['model'] == model_a) & (df['metric'] == metric)]
    b_vals = df[(df['cutoff'] == cutoff) & (df['features'] == feature) &
                (df['model'] == model_b) & (df['metric'] == metric)]

    if len(a_vals) == len(b_vals) and len(a_vals) > 0:
        try:
            _, pval = stats.wilcoxon(a_vals['value'], b_vals['value'])
            return pd.DataFrame([{
                'cutoff': cutoff,
                'features': feature,
                'model1': model_a,
                'model2': model_b,
                'metric': metric,
                'p_value': pval,
                'significant': pval < 0.05
            }])
        except:
            pass
    return pd.DataFrame([])

def run_statistical_analysis():
    files = [
        "fold_metrics_6m_full.csv",
        "fold_metrics_6m_reduced.csv",
        "fold_metrics_12m_full.csv",
        "fold_metrics_12m_reduced.csv"
    ]

    df = load_all_metrics(files)
    metrics = ['f1', 'auc']
    long_df = df.melt(id_vars=['cutoff', 'features', 'model', 'fold'],
                      value_vars=metrics, var_name='metric', value_name='value')

    model_tests_all = pd.concat([friedman_and_wilcoxon(long_df, metric=m) for m in metrics])
    model_tests_all.to_csv("model_comparison_tests.csv", index=False)

    feature_tests_all = pd.concat([full_vs_reduced(long_df, metric=m) for m in metrics])
    feature_tests_all.to_csv("feature_set_comparison.csv", index=False)

    comparisons = [
        ('Gradient Boosting', 'XGBoost'),
        ('Random Forest', 'Gradient Boosting'),
        ('Logistic Regression', 'SVM'),
    ]

    model_comparisons = []
    for m1, m2 in comparisons:
        for metric in metrics:
            df_cmp = compare_models(long_df, metric, m1, m2, cutoff='12m', feature='reduced')
            model_comparisons.append(df_cmp)

    pd.concat(model_comparisons).to_csv("pairwise_model_comparisons.csv", index=False)
    print("All statistical tests saved.")

if __name__ == "__main__":
    run_statistical_analysis()
