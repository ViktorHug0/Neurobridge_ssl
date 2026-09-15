import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
)

# Data - "Neural-MCRL" and "Shallow Alignment" split into two lines
labels = ["UBP", "Neural-\nMCRL", "Neuro-\nBridge", "Shallow\nAlignment", "SAGE"]
base_acc = [12.4, 14.0, 19.0, 21.8, 35.9]
total_acc = 77.0

# Plotting - Preserving the large 22.5 x 12.75 dimensions
fig, ax = plt.subplots(figsize=(22.5, 12.75))

# Ensure grid lines are behind the bars
ax.set_axisbelow(True)
plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# Bar width preserved at 0.66
bar_width = 0.66

# Non-SAGE bars (grey)
plt.bar(labels[:-1], base_acc[:-1], width=bar_width, color='grey',
        edgecolor='black', linewidth=4.8, zorder=3)

# SAGE bar - Subject-disentangled training
plt.bar(labels[-1], base_acc[-1], width=bar_width, color='#38fffd34',
        edgecolor='black', linewidth=4.8, label='Subject-disentangled training', zorder=3)

# SAGE bar - Transductive geometric alignment (stacked on SAGE)
plt.bar("SAGE", total_acc - 35.9, bottom=35.9, width=bar_width, color='#ff620334',
        edgecolor='black', linewidth=4.8, label='Transductive geometric alignment', zorder=3)

# Font sizes increased by another 50%
plt.ylabel("Top-1 Accuracy (%)", fontsize=80, labelpad=40, fontweight="bold")
plt.xticks(fontsize=60)
plt.yticks(fontsize=68)
plt.setp(ax.get_xticklabels(), fontweight="bold")
plt.setp(ax.get_yticklabels(), fontweight="bold")
ax.tick_params(axis="x", pad=plt.rcParams["xtick.major.pad"] + 15)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2.0)
ax.spines['bottom'].set_linewidth(2.0)

# Legend adjustment - increased by another 50%
plt.legend(
    loc='upper left',
    bbox_to_anchor=(0.0, 1.2),
    bbox_transform=ax.transAxes,
    frameon=True,
    framealpha=0.9,
    edgecolor='0.8',
    prop={'size': 75, 'weight': 'bold'},
)

plt.ylim(0, 100)
plt.tight_layout()

# Save
plt.savefig('sage_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
