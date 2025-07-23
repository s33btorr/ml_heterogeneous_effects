# THIS FILE WILL: Call the dta, clean it, normalize it, run Causal Forest and print results.
# I will try to describe step by step in any case something is not very clear.

import pandas as pd 
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor

### 1. We call the functions generated (explained better in each file, look at them for more info). ###
from data_management.clean_data import clean_dataset
from data_management.normalizing_df import normalize_data
from analysis.forest.causal_forest import generating_causal_forest, graph_distribution_indiv_treatment, graph_importance_variables, graph_representative_tree, printing_some_characteristics
from analysis.forest.dr_tester import cf_drtest

#### 2. We define the path of our data and read it. ###
SRC = Path(__file__).parent.resolve()
data_path = SRC / "data" / "waves123_augmented_consent.dta"
BLD = SRC / "bld" 
waves_data = pd.read_stata(data_path, convert_categoricals=False)

#### 3. We clean the data and then we normalize it. ###
data_after_cleaning = clean_dataset(waves_data)
data_done = normalize_data(data_after_cleaning)

#### 4. Now we define important variables in our data to be able to run the Causal Forest. ###

# Models for predicting
model_regression = RandomForestRegressor(random_state=23)
model_propensity = RandomForestClassifier(random_state=23)

# Q1 outcome and treatment columns
y_1 = data_done["Q1_1"]
z_1 = data_done["Q1_1_treat"].astype(int)
# Q2 outcome and treatment columns
y_2 = data_done["Q1_2"]
z_2 = data_done["Q1_2_treat"].astype(int)
# Q3 outcome and treatment columns
y_3 = data_done["Q2_1"]
z_3 = data_done["Q2_1_treat"].astype(int)
# Q4 outcome and treatment columns
y_4 = data_done["Q2_2"]
z_4 = data_done["Q2_2_treat"].astype(int)

# List of names for covariates columns
talk_list_1 = ["open_to_experience", "PC1", "empathic_concern_score", "Altruism", 
               "Positive_Reciprocity", "Negative_Reciprocity", "Trust", "Risk_Preferences",
               "rationality_score", "optimism_bias", "three_tax", "three_ban",
               "education", "age", "financialwellbeing", "born_in_lux"]

# We just define the selected list here
selected_list = talk_list_1

# DataFrame with for covariates columns
x_cov = data_done[talk_list_1]

### 5. Now we want to predict the individual treatment effects on the sample AND graph it in a distribution. ###

# We define the lists of our 4 outcomes so we can do a loop and iterate in each outcome the functions.
outcome_list = [y_1, y_2, y_3, y_4]
d_var_list = [z_1, z_2, z_3, z_4]

# We iterate, this could also just be done separately by just errasing the loop and choosing manually the variables.
for y, d, q in zip(outcome_list, d_var_list, range(1,5)):
    model = generating_causal_forest(model_regression, model_propensity, 100, 10, 0.5, 23, y, d, x_cov)
    graph_distribution_indiv_treatment(model, x_cov, q)

### 6. Given the importance of the test, we divide our sample into test and train and do a DRTest of our model. ###
# Again, looping to get the outcomes for the 4 questions.

for y, d, q in zip(outcome_list, d_var_list, range(1,5)):
    cf_drtest(y, d, x_cov, q, 
              model_regression, model_propensity, 
              2, 0.4,
              100, 10, 0.5, 23)

### 7. We could also do some graphs for better interpretation ###
# Here we do variable importances and a representative tree WITH the 100% of the sample.

for y, d, q in zip(outcome_list, d_var_list, range(1,5)):
    graph_importance_variables(model, x_cov, q)
    graph_representative_tree(model, x_cov, selected_list, q, 3, 10, 23)

# If you want to do with only some part of the sample, use function from train_test_split.py,
# then train a model only with this data and use this model for the graphs.