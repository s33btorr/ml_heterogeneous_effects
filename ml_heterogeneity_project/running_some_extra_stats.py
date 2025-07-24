import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt

from data_management.clean_data import clean_dataset
from data_management.normalizing_df import normalize_data
from  analysis.initial_stats.descriptive_statistics import graph_distributions_treated_vs_not, graph_soft_distributions_treated_vs_not, graph_heatmap_corr_cov
from analysis.initial_stats.placebo_test import placebo_testing


SRC = Path(__file__).parent.resolve()
data_path = SRC / "data" / "waves123_augmented_consent.dta"
BLD = SRC / "bld" 
waves_data = pd.read_stata(data_path, convert_categoricals=False)

data_after_cleaning = clean_dataset(waves_data)
data_done = normalize_data(data_after_cleaning)

y_list = ["Q1_1", "Q1_2", "Q2_1", "Q2_2"]
d_list =["Q1_1_treat", "Q1_2_treat", "Q2_1_treat", "Q2_2_treat"]

list_of_cov = ["age", "education", "conscientiousness", "female", "extraversion"]

new_list_1 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
              "rationality_score", "growthmind", "Trust", "Altruism", "Risk_Preferences",
              "marielboatlift", "empathic_concern_score"]

list_for_heatmap = new_list_1

# DESCRIPTIVE STATISTICS

for i in list_of_cov:
    plt.figure()
    graph_distributions_treated_vs_not(data_done, i, "Q1_1_treat")
    plt.tight_layout()
    plt.savefig(f"bld/graph_hist_{i}.png", dpi=300)
    plt.close()

for i in list_of_cov:
    plt.figure()
    graph_soft_distributions_treated_vs_not(data_done, i, "Q1_1_treat")
    plt.tight_layout()
    plt.savefig(f"bld/graph_kde_{i}.png", dpi=300)
    plt.close()

plt.figure()
graph_heatmap_corr_cov(data_done, list_for_heatmap)
plt.tight_layout()
plt.savefig("bld/heatmap_covariates.png", dpi=300)
plt.close()

# PLACEBO TEST
for i,j,k in zip(d_list, y_list, range(4)):
    fig = placebo_testing(data_done, i, j, k+1)
    fig.savefig(f"bld/placebo_test_question_{k+1}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
