"""Regenerate the README figure from outputs/oof_all_models.csv.

    python make_figures.py

No W&B, no model files, no raw data — the committed OOF predictions are enough.
The fold assignment is recoverable because StratifiedKFold.split(X, y) consumes
only `y` and the seed; X is used for its length alone. So the exact five folds
every model trained against can be rebuilt here, and each model's per-fold score
recomputed from its out-of-fold column.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED, N_FOLDS = 42, 5
HERE = os.path.dirname(os.path.abspath(__file__))
OOF_CSV = os.path.join(HERE, 'outputs', 'oof_all_models.csv')
FIG_DIR = os.path.join(HERE, 'figures')

# --- palette (validated: all-pairs CVD dE 9.2, normal-vision 24.0, light) ----
SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
INK_MUTED = '#898781'
GRID      = '#e1e0d9'
AXIS      = '#c3c2b7'
SERIES    = {'oof_xgb': '#2a78d6', 'oof_lgb': '#eb6834', 'oof_cat': '#1baf7a'}
LABEL     = {'oof_xgb': 'XGBoost', 'oof_lgb': 'LightGBM',
             'oof_cat': 'CatBoost', 'oof_blend': 'Blend'}

# The known per-fold numbers printed by the notebook, used to prove the fold
# reconstruction below is the same split the models actually trained on.
EXPECTED = {
    'oof_lgb': [0.79125, 0.80034, 0.79150, 0.79549, 0.79107],
    'oof_cat': [0.79063, 0.79880, 0.78884, 0.79560, 0.79002],
}


def fold_scores(df):
    y = df['TARGET'].to_numpy()
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=SEED).split(np.zeros(len(y)), y))
    out = {}
    for col in ['oof_xgb', 'oof_lgb', 'oof_cat', 'oof_blend']:
        out[col] = [roc_auc_score(y[va], df[col].to_numpy()[va])
                    for _, va in folds]
    return out


def verify(scores):
    for col, want in EXPECTED.items():
        got = scores[col]
        ok = np.allclose(got, want, atol=1e-5)
        print(f'  {LABEL[col]:9s} folds {[f"{v:.5f}" for v in got]}  '
              f'{"MATCH" if ok else "MISMATCH — expected " + str(want)}')
        assert ok, f'{col}: fold reconstruction does not match the training run'


def plot(df, scores):
    rows = ['oof_xgb', 'oof_lgb', 'oof_cat', 'oof_blend']
    y_all = df['TARGET'].to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 3.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, col in enumerate(rows):
        ypos = len(rows) - 1 - i
        # the blend is the aggregate, not a peer series — ink, not a 4th hue
        colour = SERIES.get(col, INK)
        folds = scores[col]

        ax.plot([min(folds), max(folds)], [ypos, ypos],
                color=colour, alpha=0.30, lw=2, solid_capstyle='round', zorder=1)
        ax.scatter(folds, [ypos] * len(folds), s=70, color=colour,
                   edgecolors=SURFACE, linewidths=2, zorder=3)

        oof = roc_auc_score(y_all, df[col].to_numpy())
        ax.scatter([oof], [ypos], marker='D', s=95, color=colour,
                   edgecolors=SURFACE, linewidths=2, zorder=4)
        # direct label — the relief rule for the sub-3:1 aqua slot
        ax.annotate(f'{oof:.5f}', (oof, ypos), xytext=(0, 13),
                    textcoords='offset points', ha='center',
                    color=INK, fontsize=9.5, fontweight='600')

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[c] for c in reversed(rows)],
                       fontsize=11, color=INK)
    ax.set_xlabel('ROC-AUC', fontsize=10, color=INK_MUTED)
    ax.set_ylim(-0.6, len(rows) - 0.25)

    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.tick_params(axis='y', labelcolor=INK)

    ax.set_title('Per-fold ROC-AUC  ·  dots are the 5 folds, diamond is the '
                 'pooled out-of-fold score',
                 fontsize=10.5, color=INK_MUTED, loc='left', pad=14)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, 'fold_spread.png')
    fig.savefig(path, facecolor=SURFACE, bbox_inches='tight')
    print(f'\nwrote {path}')

    spread = max(max(scores[c]) - min(scores[c]) for c in rows[:3])
    best = max(roc_auc_score(y_all, df[c].to_numpy()) for c in rows[:3])
    worst = min(roc_auc_score(y_all, df[c].to_numpy()) for c in rows[:3])
    print(f'widest fold spread {spread:.4f}  vs  best-worst model gap '
          f'{best - worst:.4f}   ({spread / (best - worst):.1f}x)')


if __name__ == '__main__':
    df = pd.read_csv(OOF_CSV)
    print(f'{len(df):,} rows, positive rate {df.TARGET.mean():.4f}')
    print('verifying fold reconstruction against the notebook\'s printed scores:')
    scores = fold_scores(df)
    verify(scores)
    plot(df, scores)
