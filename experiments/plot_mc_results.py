import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIG: Make text HUGE for publication ---
plt.rcParams.update({
    'font.size': 20,          # Base font size
    'axes.titlesize': 26,     # Plot title
    'axes.labelsize': 24,     # Axis labels (x and y)
    'xtick.labelsize': 20,    # X-axis tick numbers
    'ytick.labelsize': 20,    # Y-axis tick numbers
    'legend.fontsize': 20,    # Legend text
    'figure.titlesize': 30    # Figure supertitle
})
# -----------------------------------------------

df = pd.read_csv("results/monte_carlo_results.csv")

# Slightly increased figure size to accommodate larger text
plt.figure(figsize=(12, 8))

sns.boxplot(data=df, x="qubit", y="amp_err", color="lightblue")
sns.stripplot(data=df, x="qubit", y="amp_err", color="black", alpha=0.5, jitter=True)

plt.title("Amplitude Error Distribution per Qubit (50 Seeds)", pad=20)
plt.ylabel("Absolute Amplitude Error", labelpad=15)
plt.xlabel("Qubit Index", labelpad=15)

plt.yscale("log") # Useful if errors are very small
plt.grid(True, which="both", ls="--", alpha=0.3)

# Essential: Adjusts margins so huge labels don't get cut off
plt.tight_layout()

# Save figure
plt.savefig("figures/monte_carlo_amplitude_errors.png", dpi=300, bbox_inches='tight')
print("Figure saved to: figures/monte_carlo_amplitude_errors.png")

plt.show()
