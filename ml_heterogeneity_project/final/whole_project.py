import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor

from  descriptive_statistics import graph_distributions_treated_vs_not, graph_soft_distributions_treated_vs_not, graph_heatmap_corr_cov
from placebo_test import placebo_testing
from normalizing_df import normalize_data
from causal_forest import generating_causal_forest, graph_distribution_indiv_treatment, graph_importance_variables, graph_representative_tree, printing_some_characteristics
from dr_tester import cf_drtest


# pensar si ya agrego estas cosas a config
SRC = Path(__file__).parent.resolve()
data_ready = pd.read_csv(SRC/"data_cleaned.csv") 

data_done = data_ready.copy()
#data_done = data_done[data_done['gender'].notna()] # MIRAR PORQUE NO QUIERO ELIMINAR A ESTA GENTE...

### ANTES DE TODO DEBERIA IMPORTAR EL DF Y PASARLO POR CLEANING Y POR NORMALIZING LA DATA

y_list = ["Q1_1", "Q1_2", "Q2_1", "Q2_2"]
d_list =["Q1_1_treat", "Q1_2_treat", "Q2_1_treat", "Q2_2_treat"]

list_of_cov = ["age", "education", "conscientiousness", "female", "extraversion"]

list_of_all_cov = [
    "employment", "education", "age", 
    "female", "houseowner", "financialwellbeing", 
    "optimism_bias", "openness", "conscientiousness", 
    "extraversion", "agreeableness", "neuroticism", 
    "rationality_score", "growthmind", "lambda_risky", "Social_Anxiety", 
    "Public_SelfConsciousness", "Private_SelfConsciousness", 
    "ProcrastinationExAnte", "ProcrastinationExPost", 
    "empathic_concern_score", "perspective_taking_score"]

the_chosen_one = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", 
                "world_issues", "trust_in_science", "policy_preferences", "Public_SelfConsciousness",
                "longlistbeh_score", "persconcern", "perspriority_3"]

basic_list = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", 
                "world_issues", "trust_in_science", "policy_preferences", "longlistbeh_score"]

final_list_1 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", 
                "world_issues", "trust_in_science", "policy_preferences", "longlistbeh_score",
                "country_residence", "age", "female_created", "education", "financialwellbeing", "residence"
                ]

final_list_2 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", 
                "world_issues", "trust_in_science", "policy_preferences", "longlistbeh_score",
                "Public_SelfConsciousness", "empathic_concern_score", "Negative_Reciprocity",
                "persconcern", "perspriority_3"
                ]

# esta era la chosen
final_list_3 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", 
                "world_issues", "trust_in_science", "policy_preferences", "Public_SelfConsciousness",
                "longlistbeh_score", "persconcern", "perspriority_3"
                ]

# we leave out big 5
final_list_4 = ["world_issues", "trust_in_science", "policy_preferences", "Public_SelfConsciousness",
                "longlistbeh_score", "persconcern", "perspriority_3",
                "empathic_concern_score", "Negative_Reciprocity"
                ]

final_list_5 = ["world_issues", "trust_in_science", "policy_preferences", "Public_SelfConsciousness",
                "longlistbeh_score", "empathic_concern_score", "Negative_Reciprocity",
                "country_residence", "age", "female_created", "education", "financialwellbeing"
                ]

# we leave out the not big 5
final_list_6 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", 
                "empathic_concern_score", "Negative_Reciprocity",
                "trustclimatechange1", "persconcern", "perspriority_3"
                ]

# we leave everything
final_list_7 = ["empathic_concern_score", "Negative_Reciprocity",
                "country_residence", "age", "female_created", "education", "financialwellbeing"]


test_list_1 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", 
                "world_issues", "trust_in_science", "policy_preferences", "Public_SelfConsciousness",
                "longlistbeh_score", "persconcern", "perspriority_3", "marielboatlift"
                ]

the_chosen_one = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", 
                "world_issues", "trust_in_science", "policy_preferences", "Public_SelfConsciousness",
                "longlistbeh_score", "persconcern", "perspriority_3"]

# probando incluyendo la informacion ex ante treatment
another_try = ["exante_beliefs", "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", 
                "world_issues", "trust_in_science", "policy_preferences", "Public_SelfConsciousness",
                "longlistbeh_score", "persconcern", "perspriority_3"]

another_try_1 = ["marielboatlift_1", "world_issues", "trust_in_science", "policy_preferences", "Public_SelfConsciousness",
                "longlistbeh_score", "persconcern", "perspriority_3",
                "empathic_concern_score", "Negative_Reciprocity"]

another_try_2 = ["marielboatlift_2", "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", 
                "world_issues", "trust_in_science", "policy_preferences", "Public_SelfConsciousness",
                "longlistbeh_score", "persconcern", "perspriority_3"]

felix_list_1 = [ "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
                "trust_in_science", "marielboatlift_1", "female_created",
                "age", "education", "born_in_lux", "income_filled"
                ]

felix_list_2 = [ "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
                "trust_in_science", "marielboatlift_1", "female_created",
                "age", "education", "born_in_lux", "income_filled",
                "Positive_Reciprocity", "Negative_Reciprocity", "Altruism", "Trust", "Time_Preferences"
                ]

david_list_1 = ["resource_food_wi", "pandemic_risk_wi", "rest_wi", "migration_wi", "energy_saving_ll", 
                "sustainable_food_ll", "second_hand_ll", "three_tax", "climatedesobedience_1", "cartax_w1"]

david_list_2 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
                "age", "education", "born_in_lux", "income_filled",
                "trust_in_science", "migration_wi",
                ]

david_theory = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

my_list = ["resource_food_wi", "pandemic_risk_wi", "rest_wi", "migration_wi", "trust_in_science", 
                "three_tax", "climatedesobedience_1", "Public_SelfConsciousness",
                #"longlistbeh_score", 
                "persconcern", "perspriority_3",
                "empathic_concern_score", "Negative_Reciprocity"
                ]

my_list_2 = ["resource_food_wi", "pandemic_risk_wi", "rest_wi", "migration_wi",
             "trust_in_science", "three_tax", "climatedesobedience_1",
             "Public_SelfConsciousness", "empathic_concern_score", "Negative_Reciprocity",
             "born_in_lux", "age", "female_created", "education", "financialwellbeing"
                ]

my_list_3 = ["resource_food_wi", "pandemic_risk_wi", "rest_wi", "migration_wi",
             "trust_in_science", "three_tax", "climatedesobedience_1",
             "energy_saving_ll", "sustainable_food_ll", "second_hand_ll",
             "Public_SelfConsciousness", "empathic_concern_score", "Negative_Reciprocity",
             "born_in_lux", "age", "female_created", "education", "financialwellbeing"
                ]

my_list_4 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
             "world_issues", "trust_in_science", "empathic_concern_score", "three_tax",
             "born_in_lux", "age", "education", "financialwellbeing"
                ]

my_list_5 = ["trust_in_science", "three_tax",
             "Public_SelfConsciousness", "empathic_concern_score", "Negative_Reciprocity",
             "born_in_lux", "age", "female_created", "education", "financialwellbeing"]

# AFTER MEETING
new_list_1 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
              "rationality_score", "growthmind", "Trust", "Altruism", "Risk_Preferences",
              "marielboatlift_1", "empathic_concern_score"]


# NORMALIZING DATASET
data_done = normalize_data(data_done)
"""

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
graph_heatmap_corr_cov(data_done, the_chosen_one)
plt.tight_layout()
plt.savefig("bld/heatmap_covariates.png", dpi=300)
plt.close()

# PLACEBO TEST
for i,j,k in zip(d_list, y_list, range(4)):
    fig = placebo_testing(data_done, i, j, k+1)
    fig.savefig(f"bld/placebo_test_question_{k+1}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
"""

# CAUSAL FOREST RESULTS

y_1 = data_done["Q1_1"]
d_var = data_done["Q1_1_treat"] # quite esto, pendiente por si no funciona por esto: .astype(int)  (no creo)

y_2 = data_done["Q1_2"]

y_3 = data_done["Q2_1"]

y_4 = data_done["Q2_2"]

control_var = data_done[["marielboatlift"]]

x_cov = data_done[new_list_1]


model_regression = RandomForestRegressor(random_state=23)
model_propensity = RandomForestClassifier(random_state=23)

outcome_list = [y_1, y_2, y_3, y_4]

# habria que cambiar estas funciones para que tb agarren la dataset...como en dr test
for y, q in zip(outcome_list, range(1,5)):
    model = generating_causal_forest(model_regression, model_propensity, 100, 10, 0.5, 23, y, d_var, x_cov)
    graph_distribution_indiv_treatment(model, x_cov, q)


    # SOME INTERPRETATION GRAPHS
    graph_importance_variables(model, x_cov, q)
    graph_representative_tree(model, x_cov, new_list_1, q, 3, 10, 23)
    #printing_some_characteristics(data_done, model, x_cov, q)


y_name_list = ["Q1_1", "Q1_2", "Q2_1", "Q2_2"]
for y, q in zip(y_name_list, range(1,5)):
    # DR TEST
    cf_drtest(data_done, "Q1_1_treat", y, new_list_1, q, 
              model_regression, model_propensity, 2, 0.4,
              100, 10, 0.5, 23)
    

