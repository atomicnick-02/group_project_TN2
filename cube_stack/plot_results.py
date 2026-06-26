import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

# Each source CSV is tagged with the method name shown in the legend.
SOURCES = [
    ("behavior_cloning_results.csv", "Behavior Cloning"),
    ("rollout_results.csv",          "Diffusion"),
]
os.makedirs("figures", exist_ok=True)

# Set to "avg" or "median" to choose which aggregate to plot.
STAT = "avg"

# (csv column suffix, axis label, title fragment, filename, format as %)
METRICS = [
    ("success_rate",        "Success Rate (%)",       "Success Rate",      "success",     True),
    (f"{STAT}_jitter",      "Trajectory Jitter",      "Trajectory Jitter", "jitter",      False),
    (f"{STAT}_path_effort", "Path Effort",            "Path Effort",       "effort",      False),
    (f"{STAT}_joint_cost",  "Joint Cost",             "Joint Cost",        "joint_cost",  False),
]

plt.rcParams.update({
    'font.size': 13,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.figsize': (7, 4.5),
})

# --- Load every source CSV and group rows by (dataset, method, goal) series ---
series = defaultdict(list)  # (dataset, method, goal) -> list of row dicts
for csv_path, method in SOURCES:
    if not os.path.exists(csv_path):
        print(f"Source not found, skipping: {csv_path}")
        continue
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            series[(row["dataset"], method, row["goal"])].append(row)

# Sort each series by execution steps so the lines are drawn left-to-right.
for key in series:
    series[key].sort(key=lambda r: int(r["execute_steps"]))

COLORS = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']


def label_for(dataset, method, goal):
    # Only annotate the goal when it is meaningful (diffusion runs); BC uses "-".
    if goal in ("with_goal", "no_goal"):
        goal_str = "with goal" if goal == "with_goal" else "no goal"
        return f"{method} ({goal_str})"
    return method


def is_stack_three(dataset):
    return dataset.startswith("stack_three")


# Split the series into the two-cube "Stack" task and the three-cube "Stack Three"
# task so each gets its own set of figures.
GROUPS = [
    ("Stack",       "stack",       lambda d: not is_stack_three(d)),
    ("Stack Three", "stack_three", lambda d: is_stack_three(d)),
]


def make_plot(group_series, column, ylabel, title_frag, task_title, filename, fmt_pct):
    fig, ax = plt.subplots()
    plotted = False
    for i, (key, rows) in enumerate(sorted(group_series.items())):
        xs = [int(r["execute_steps"]) for r in rows]
        ys = []
        for r in rows:
            val = r.get(column, "")
            ys.append(float(val) if val != "" else float("nan"))
        if not xs:
            continue
        ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], linestyle='-',
                color=COLORS[i % len(COLORS)], linewidth=2, markersize=8,
                label=label_for(*key))
        plotted = True

    if not plotted:
        plt.close(fig)
        print(f"Skipped {filename}: no data for column '{column}'")
        return

    ax.set_xlabel('Execution Steps ($H$)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{task_title} — {title_frag} vs Execution Steps')
    # Use the union of all execution-step values in this group as ticks.
    all_steps = sorted({int(r["execute_steps"]) for rows in group_series.values() for r in rows})
    ax.set_xticks(all_steps)
    ax.legend()
    if fmt_pct:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    fig.tight_layout()
    out = f'figures/{filename}_comparison.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


for task_title, task_slug, belongs in GROUPS:
    group_series = {key: rows for key, rows in series.items() if belongs(key[0])}
    if not group_series:
        print(f"No data for task '{task_title}', skipping.")
        continue
    for column, ylabel, title_frag, filename, fmt_pct in METRICS:
        make_plot(group_series, column, ylabel, title_frag, task_title,
                  f"{task_slug}_{filename}", fmt_pct)

print('All figures generated.')
