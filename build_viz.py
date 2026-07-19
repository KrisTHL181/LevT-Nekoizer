#!/usr/bin/env python3
"""Build training progress visualization from prog.csv."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# ── Data ────────────────────────────────────────────────────────────────
df_full = pd.read_csv(sys.argv[1])

# Validation points only (keep all — only 10)
val_df = df_full[df_full['val_loss'].notna()].copy()

# Downsample training data: keep every 10th point for display (~2k pts/line)
# Also always keep the first, last, and validation steps
STRIDE = 10
keep_mask = (df_full['step'] % STRIDE == 1) | (df_full['step'] == df_full['step'].iloc[-1])
val_steps = set(val_df['step'])
keep_mask |= df_full['step'].isin(val_steps)
df = df_full[keep_mask].copy()
print(f"Downsampled: {len(df_full):,} → {len(df):,} rows")

# EMA(1000) for loss curves (computed on full data)
for col in ['loss_total', 'loss_ins_plh', 'loss_ins_tok', 'loss_del']:
    df_full[f'{col}_ema'] = df_full[col].ewm(span=1000, adjust=True).mean()
    df[f'{col}_ema'] = df_full.loc[df.index, f'{col}_ema']

# Smooth grad_norm on full data, then subsample
df_full['grad_norm_smooth'] = df_full['grad_norm'].rolling(201, center=True, min_periods=1).median()
df['grad_norm_smooth'] = df_full.loc[df.index, 'grad_norm_smooth']

# ── Palette (dataviz reference, light mode) ─────────────────────────────
BLUE    = '#2a78d6'
GREEN   = '#008300'
MAGENTA = '#e87ba4'
YELLOW  = '#eda100'
AQUA    = '#1baf7a'
ORANGE  = '#eb6834'
VIOLET  = '#4a3aa7'
RED     = '#e34948'

SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
INK_SEC   = '#52514e'
INK_MUTED = '#898781'
GRIDLINE  = '#e1e0d9'

# ── Layout defaults ─────────────────────────────────────────────────────
PLOT_BG = SURFACE
PAPER_BG = SURFACE
FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'
FONT_COLOR = INK
AXIS_COLOR = INK_MUTED
GRID_COLOR = GRIDLINE

common_layout = dict(
    plot_bgcolor=PLOT_BG,
    paper_bgcolor=PAPER_BG,
    font=dict(family=FONT_FAMILY, color=FONT_COLOR, size=12),
    hovermode='x unified',
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.28,
        xanchor='center',
        x=0.5,
        font=dict(size=10, color=INK_SEC),
        bgcolor='rgba(0,0,0,0)',
        itemwidth=30,
    ),
    margin=dict(l=60, r=20, t=50, b=80),
    xaxis=dict(
        showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
        zeroline=False,
        linecolor=AXIS_COLOR, linewidth=1,
        title=dict(text='Step', font=dict(color=INK_MUTED, size=11)),
    ),
    yaxis=dict(
        showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
        zeroline=False,
        linecolor=AXIS_COLOR, linewidth=1,
    ),
)

# ── Chart 1: Loss curves ────────────────────────────────────────────────
fig1 = go.Figure(layout=common_layout)

# Training losses: raw (thin, muted) + EMA span=1000 (bold, solid)
series_config = [
    ('loss_total',    'Total Loss',        BLUE,    True),
    ('loss_ins_plh',  'Insert Placeholder', GREEN,   False),
    ('loss_ins_tok',  'Insert Token',       MAGENTA, False),
    ('loss_del',      'Delete',             YELLOW,  False),
]

for col, name, color, is_primary in series_config:
    # Raw line — thin, muted, secondary in legend
    fig1.add_trace(go.Scatter(
        x=df['step'], y=df[col],
        mode='lines',
        name=f'{name} (raw)',
        line=dict(color=color, width=0.8),
        opacity=0.35,
        legendgroup=col,
        hovertemplate='%{y:.3f}<extra></extra>',
        showlegend=True,
    ))
    # EMA line — bold, same color
    fig1.add_trace(go.Scatter(
        x=df['step'], y=df[f'{col}_ema'],
        mode='lines',
        name=f'{name} (EMA)',
        line=dict(color=color, width=2),
        legendgroup=col,
        hovertemplate='%{y:.3f}<extra></extra>',
        showlegend=True,
    ))

# Validation loss as markers
fig1.add_trace(go.Scatter(
    x=val_df['step'], y=val_df['val_loss'],
    mode='markers+lines',
    name='Val Loss',
    line=dict(color=ORANGE, width=2, dash='dot'),
    marker=dict(color=ORANGE, size=8, line=dict(color=SURFACE, width=2)),
    hovertemplate='%{y:.3f}<extra></extra>',
))

fig1.update_layout(
    title=dict(
        text='Training & Validation Loss',
        font=dict(size=14, color=INK),
        x=0.02, xanchor='left',
    ),
    yaxis=dict(
        title=dict(text='Loss', font=dict(color=INK_MUTED, size=11)),
        rangemode='tozero',
    ),
    hoverlabel=dict(
        bgcolor=SURFACE,
        font=dict(color=INK, family=FONT_FAMILY, size=12),
        bordercolor=GRIDLINE,
    ),
)

# ── Chart 2: Learning rate schedule (small multiples — one per optimizer) ─
fig2 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=('AdamW LR', 'Muon LR (40× AdamW)'),
)

fig2.add_trace(go.Scatter(
    x=df['step'], y=df['lr_adamw'],
    mode='lines',
    name='AdamW LR',
    line=dict(color=BLUE, width=2),
    hovertemplate='%{y:.2e}<extra></extra>',
    showlegend=False,
), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=df['step'], y=df['lr_muon'],
    mode='lines',
    name='Muon LR',
    line=dict(color=GREEN, width=2),
    hovertemplate='%{y:.2e}<extra></extra>',
    showlegend=False,
), row=2, col=1)

fig2.update_layout(
    **common_layout,
    title=dict(
        text='Learning Rate Schedule',
        font=dict(size=14, color=INK),
        x=0.02, xanchor='left',
    ),
    hoverlabel=dict(
        bgcolor=SURFACE,
        font=dict(color=INK, family=FONT_FAMILY, size=12),
        bordercolor=GRIDLINE,
    ),
    height=500,
)
fig2.update_xaxes(
    showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
    zeroline=False, linecolor=AXIS_COLOR, linewidth=1,
    title=dict(text='Step', font=dict(color=INK_MUTED, size=11)),
    row=2, col=1,
)
fig2.update_xaxes(
    showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
    zeroline=False, linecolor=AXIS_COLOR, linewidth=1,
    row=1, col=1,
)
fig2.update_yaxes(
    showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
    zeroline=False, linecolor=AXIS_COLOR, linewidth=1,
    tickformat='.1e', title='',
    row=1, col=1,
)
fig2.update_yaxes(
    showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
    zeroline=False, linecolor=AXIS_COLOR, linewidth=1,
    tickformat='.1e', title='',
    row=2, col=1,
)
# Style subplot titles
for ann in fig2.layout.annotations:
    ann.font = dict(size=11, color=INK_SEC)

# ── Chart 3: Gradient norm ──────────────────────────────────────────────
fig3 = go.Figure(layout=common_layout)

fig3.add_trace(go.Scatter(
    x=df['step'], y=df['grad_norm'],
    mode='lines',
    name='Grad Norm (raw)',
    line=dict(color=GRIDLINE, width=0.5),
    hovertemplate='%{y:.2f}<extra></extra>',
    showlegend=True,
))
fig3.add_trace(go.Scatter(
    x=df['step'], y=df['grad_norm_smooth'],
    mode='lines',
    name='Grad Norm (median 201)',
    line=dict(color=BLUE, width=2),
    hovertemplate='%{y:.2f}<extra></extra>',
    showlegend=True,
))

fig3.update_layout(
    title=dict(
        text='Gradient Norm',
        font=dict(size=14, color=INK),
        x=0.02, xanchor='left',
    ),
    yaxis=dict(
        title=dict(text='Gradient Norm', font=dict(color=INK_MUTED, size=11)),
    ),
    hoverlabel=dict(
        bgcolor=SURFACE,
        font=dict(color=INK, family=FONT_FAMILY, size=12),
        bordercolor=GRIDLINE,
    ),
)

# ── Assemble HTML ───────────────────────────────────────────────────────
# Convert figures to HTML divs
html_parts = []
html_parts.append(fig1.to_html(full_html=False, include_plotlyjs='cdn', div_id='chart1'))
html_parts.append(fig2.to_html(full_html=False, include_plotlyjs=False, div_id='chart2'))
html_parts.append(fig3.to_html(full_html=False, include_plotlyjs=False, div_id='chart3'))

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training Progress — prog.csv</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: {FONT_FAMILY};
    background: #f9f9f7;
    color: {INK};
    padding: 24px 32px;
    max-width: 1100px;
    margin: 0 auto;
  }}
  .viz-root {{
    color-scheme: light;
    --surface-1: {SURFACE};
    --text-primary: {INK};
    --text-secondary: {INK_SEC};
    --text-muted: {INK_MUTED};
  }}
  h1 {{
    font-size: 20px;
    font-weight: 600;
    color: {INK};
    margin-bottom: 4px;
  }}
  .subtitle {{
    font-size: 13px;
    color: {INK_SEC};
    margin-bottom: 24px;
  }}
  .chart-container {{
    background: {SURFACE};
    border-radius: 8px;
    margin-bottom: 24px;
    padding: 8px 4px;
  }}
  .note {{
    font-size: 12px;
    color: {INK_MUTED};
    margin-top: 16px;
    padding: 12px 0;
    border-top: 1px solid {GRIDLINE};
  }}
  @media (prefers-color-scheme: dark) {{
    body {{
      background: #0d0d0d;
      color: #ffffff;
    }}
    .chart-container {{
      background: #1a1a19;
    }}
    .note {{
      color: #c3c2b7;
      border-top-color: #2c2c2a;
    }}
  }}
</style>
</head>
<body class="viz-root">
<h1>Training Progress</h1>
<p class="subtitle">
  val_loss NaN = no validation at that step
</p>

<div class="chart-container">{html_parts[0]}</div>
<div class="chart-container">{html_parts[1]}</div>
<div class="chart-container">{html_parts[2]}</div>

<p class="note">
  Validation runs every 2,000 steps.
  EMA span=1000 applied to all training loss curves. LR schedule: warmup then cosine decay. Grad norm smoothed with rolling median (window=201).
</p>
</body>
</html>'''

with open('prog_viz.html', 'w') as f:
    f.write(html)

print(f"Written: prog_viz.html ({len(html):,} bytes)")
print(f"Charts: 3 (loss, LR, grad norm)")
