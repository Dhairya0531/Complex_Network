import re

with open("main.py", "r") as f:
    content = f.read()

# Make fonts bigger globally
content = content.replace(
'''    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )''',
'''    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.titlesize": 26,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
        }
    )''')

# Add legends to plot 1
content = content.replace(
'''    ax1.set_ylim(0, max(clean_vals) * 1.3)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")''',
'''    ax1.set_ylim(0, max(clean_vals) * 1.4)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.legend(bars, labels_short, loc='upper left')''')

# Add legends to plot 2
content = content.replace(
'''    ax2.set_ylim(0, max(clean_vals) * 1.3)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")''',
'''    ax2.set_ylim(0, max(clean_vals) * 1.4)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.legend(bars, labels_short, loc='upper left')''')

# Add legends to plot 3
content = content.replace(
'''    ax3.set_ylim(0, max(clean_vals) * 1.3)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")''',
'''    ax3.set_ylim(0, max(clean_vals) * 1.4)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    ax3.legend(bars, labels_short, loc='upper left')''')

# Add legends to plot 4
content = content.replace(
'''    ax4.set_ylabel("Avg Wait Time (s)")
    ax4.grid(axis="y", alpha=0.3, linestyle="--")''',
'''    ax4.set_ylim(bottom=0)
    ax4.set_ylabel("Avg Wait Time (s)")
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    ax4.legend(bp["boxes"], labels_short, loc='best')''')

# Plot 5 legend size
content = content.replace(
'''    ax5.legend(fontsize=10)''',
'''    ax5.legend(loc='upper left')''')

# Plot 6 legend size
content = content.replace(
'''    ax6.legend(fontsize=10)''',
'''    ax6.legend(loc='upper left')''')

# Plot 7 legend
content = content.replace(
'''    ax7.hist(
        list(betweenness.values()),
        bins=20,
        color="#2c3e50",
        edgecolor="black",
        alpha=0.7,
    )''',
'''    ax7.hist(
        list(betweenness.values()),
        bins=20,
        color="#2c3e50",
        edgecolor="black",
        alpha=0.7,
        label="Centrality"
    )
    ax7.legend(loc='upper right')''')

# Plot 8 legend
content = content.replace(
'''    ax8.set_ylabel("Value")
    ax8.grid(axis="y", alpha=0.3, linestyle="--")''',
'''    ax8.set_ylabel("Value")
    ax8.grid(axis="y", alpha=0.3, linestyle="--")
    ax8.legend(bp8["boxes"], ["Alpha", "Beta", "Gamma"], loc='lower right')''')

# Fix fig sizes globally as well so legends fit
content = content.replace('''figsize=(6, 5)''', '''figsize=(8, 6)''')

with open("main.py", "w") as f:
    f.write(content)
