"""Charts for the model comparison. Reads results/study3.json, results/study5.json and results/offline_metrics.json."""
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {
    'greedy': 'greedy (soft count)', 'random': 'random feasible', 'global_only': 'global, soft count',
    'global_learned': 'global, online rate', 'logit_greedy': 'logistic + greedy', 'logit_global': 'logistic + global',
    'logit_staged': 'logistic staged + global', 'gbm_global': 'boosted trees + global',
    'thompson_global': 'Thompson + global', 'oracle_greedy': 'ORACLE + greedy (diagnostic)',
    'oracle_global': 'ORACLE + global (diagnostic)', 'gbm_value': 'boosted trees + value asks', 'value_ask_only': 'value-ranked asks only (greedy)',
    'logit_value': 'logistic + value asks'}


def paired(rows, mode, ref, key):
    base = {(r['variant'], r['seed']): r for r in rows if r['mode'] == ref}
    diff = [r[key] - base[(r['variant'], r['seed'])][key] for r in rows if r['mode'] == mode]
    return statistics.mean(diff), 1.96 * statistics.stdev(diff) / len(diff) ** 0.5


def dots(ax, rows, modes, key, title, xlabel):
    modes = [m for m in modes if m != 'greedy']
    for i, m in enumerate(modes):
        mean, ci = paired(rows, m, 'greedy', key)
        color = '#b04a4a' if m.startswith('oracle') else '#2f6f9f'
        ax.errorbar(mean, i, xerr=ci, fmt='o', color=color, capsize=3)
    ax.axvline(0, color='#555', lw=1)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([LABELS[m] for m in modes], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(axis='x', alpha=0.3)


rows3 = json.load(open('results/study3.json'))['rows']
modes3 = ['greedy', 'random', 'global_only', 'global_learned', 'logit_greedy', 'logit_global', 'logit_staged',
          'gbm_global', 'thompson_global', 'oracle_greedy', 'oracle_global']
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
dots(axes[0, 0], rows3, modes3, 'mutual', 'Study 3: mutual acceptances per episode, difference vs greedy\n(360 paired episodes, 95% interval)', 'more mutual acceptances than greedy →')
dots(axes[0, 1], rows3, modes3, 'msmi', 'Study 3: MSMI per episode, difference vs greedy', 'more MSMI than greedy →')

off = json.load(open('results/offline_metrics.json'))
coef = off['logit_coefficients']
names = list(coef)[:10][::-1]
axes[1, 0].barh(names, [coef[n] for n in names], color=['#2f6f9f' if coef[n] > 0 else '#b04a4a' for n in names])
axes[1, 0].axvline(0, color='#555', lw=1)
axes[1, 0].set_title('Logistic model: what moves the chance of a yes\n(standardised coefficients, top 10)', fontsize=10)
axes[1, 0].tick_params(labelsize=8)

cal = off['calibration_logit']
axes[1, 1].plot([c[0] for c in cal], [c[1] for c in cal], 'o-', color='#2f6f9f', label='logistic, held-out worlds')
axes[1, 1].plot([0.2, 0.5], [0.2, 0.5], '--', color='#777', label='perfectly calibrated')
axes[1, 1].set_xlabel('predicted chance of a yes', fontsize=8)
axes[1, 1].set_ylabel('observed yes rate', fontsize=8)
axes[1, 1].set_title('Calibration (10 equal-count bins)\nheld-out AUC = %.3f' % off['yes']['logistic regression']['auc'], fontsize=10)
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
Path('results').mkdir(exist_ok=True)
plt.savefig('results/model_comparison.png', dpi=130)
print('saved results/model_comparison.png')

if Path('results/study5.json').exists():
    rows4 = json.load(open('results/study5.json'))['rows']
    modes4 = [m for m in ('greedy', 'value_ask_only', 'logit_global', 'gbm_global', 'logit_value', 'gbm_value')
              if any(r['mode'] == m for r in rows4)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    dots(axes[0], rows4, modes4, 'mutual', 'Study 5 (pre-declared confirmation, 600 paired episodes): mutual acceptances per episode vs greedy', 'more mutual acceptances than greedy →')
    dots(axes[1], rows4, modes4, 'msmi', 'Study 5 (pre-declared confirmation, 600 paired episodes): MSMI per episode vs greedy', 'more MSMI than greedy →')
    plt.tight_layout()
    plt.savefig('results/confirmation.png', dpi=130)
    print('saved results/confirmation.png')
