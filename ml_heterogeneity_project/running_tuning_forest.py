import pandas as pd 
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor

from data_management.clean_data import clean_dataset
from data_management.normalizing_df import normalize_data
from analysis.forest.causal_forest import generating_causal_forest, graph_distribution_indiv_treatment, graph_importance_variables, graph_representative_tree, printing_some_characteristics, calculate_RScorer
from analysis.forest.tuning_forest import tuning_causal_forest

SRC = Path(__file__).parent.resolve()
data_path = SRC / "data" / "waves123_augmented_consent.dta"
BLD = SRC / "bld" 
waves_data = pd.read_stata(data_path, convert_categoricals=False)

data_after_cleaning = clean_dataset(waves_data)
data_done = normalize_data(data_after_cleaning)

y_1 = data_done["Q1_1"]
z_1 = data_done["Q1_1_treat"].astype(int)
y_2 = data_done["Q1_2"]
z_2 = data_done["Q1_2_treat"].astype(int)
y_3 = data_done["Q2_1"]
z_3 = data_done["Q2_1_treat"].astype(int)
y_4 = data_done["Q2_2"]
z_4 = data_done["Q2_2_treat"].astype(int)

talk_list_1 = ["open_to_experience", "PC1", "empathic_concern_score", "Altruism", 
               "Positive_Reciprocity", "Negative_Reciprocity", "Trust", "Risk_Preferences",
               "rationality_score", "optimism_bias", "three_tax", "three_ban",
               "education", "age", "financialwellbeing", "born_in_lux"]

selected_list = talk_list_1
x_cov = data_done[talk_list_1]

outcome_list = [y_1, y_2, y_3, y_4]
d_var_list = [z_1, z_2, z_3, z_4]

model_regression = RandomForestRegressor(random_state=23)
model_propensity = RandomForestClassifier(random_state=23)

n_trees = [100, 500, 1000, 1500] # Number of trees
min_obs_leaf = [5, 20, 60] # Min observations per leaf
division_sample = [0.2, 0.3, 0.45, 0.5] # Number of samples to use for each subsample that is used to train each tree

for y, d, q in zip(outcome_list, d_var_list, range(1,5)):
    print(f'Question{q}')
    tuning_causal_forest(n_trees, min_obs_leaf, division_sample,
                        d, y, x_cov, 0.4,
                        model_regression, model_propensity, 23)