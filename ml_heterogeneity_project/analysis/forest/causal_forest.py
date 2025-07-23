import numpy as np
from sklearn import tree
import pandas as pd
from econml.dml import CausalForestDML as CausalForest
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from econml.cate_interpreter import SingleTreeCateInterpreter
from econml.dml import LinearDML
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from econml.score import RScorer

from analysis.forest.train_test_split import split_data



def generating_causal_forest(model_outcome, model_treatment, n_trees, min_sample_per_leaf, max_samples, random_seed, outcome, treatment, x_cov):
    """ Generates a Causal Forest.

    Args:
        model_outcome (object): Model that predicts the Outcome,
        it can be selected from EconML documentation.

        model_treatment (object): Model that predicts the Treatment,
        it can be selected from EconML documentation.

        n_trees (int): Number of trees wanted in the forest.

        min_sample_per_leaf (int): Minimum number of observations per leaf.

        max_samples (float): In how much the data should be splitted when
        doing the forest. The maximum allowed is 0.5 given Honesty.

        random_seed (int): Seed for reproducibility.

        outcome (column): Column that represents the outcome.

        treatment (column): Column that represents the treatment.

        x_cov (DataFrame): DataFrame of all columns representing the covariates.

    Returns:
        model(object): Trained model that can be used later for graphs or tests.
    """

    model = CausalForest(
        model_y=model_outcome,
        model_t=model_treatment,
        n_estimators=n_trees,
        discrete_treatment=True,
        criterion="het",
        min_samples_leaf=min_sample_per_leaf,
        max_samples=max_samples,
        random_state=random_seed
    ).fit(outcome, treatment, X=x_cov)

    #print(model.summary()) #quitar
    return model


# ESTE SE IRIA A OTRO PY o no?
def graph_distribution_indiv_treatment(model, x_cov, n_question):
    """ Generates a graph with the distribution of predictions made by
    the Causal Forest model.

    Args:
        model (object): Model trained previously.
        
        x_cov (DataFrame): DataFrame of all columns representing the covariates.

        n_question (int): Number of the question that has the outcome.
    """

    ate_cf = model.ate(x_cov)
    cate = model.effect(x_cov)
    sns.histplot(cate, bins=15, color='lightgray', edgecolor='black')
    plt.title(f'Question {n_question}: Histogram of estimated CATE')
    plt.xlabel("Estimated CATE")
    plt.ylabel("Frequency")
    plt.axvline(ate_cf, color="green", linestyle="--", label=f'ATE={ate_cf:.2f}')
    plt.legend()
    plt.savefig(f'bld/predicted_cate_question{n_question}.png', dpi=300)
    plt.close()


# ESTE IRIA CON LS GRFICAS FINALES TB o no?
# no se si estos se deberian hacer con train solo o con el total, ahora se andan haciendo con el total
def graph_importance_variables(model, x_cov, question):
    """ Generates a graph with "importance" of each variable given
    how much it was selected for a split during the Causal Forest.

    Args:
        model (object): Model trained previously.
        
        x_cov (DataFrame): DataFrame of all columns representing the covariates.

        question (int): Number of the question that has the outcome.
    """

    importances = model.feature_importances_
    features = x_cov.columns
    plt.barh(features, importances)
    plt.xlabel("Importance")
    plt.savefig(f'bld/variable_importance_question{question}.png', dpi=300, bbox_inches='tight')
    plt.close()

def graph_representative_tree(model, x_cov, covariates_names, question, tree_depth, min_per_leaf, random_seed):
    """ Generates a graph with a representative tree of Causal Forest.

    Args:
        model (object): Model trained previously.
        
        x_cov (DataFrame): DataFrame of all columns representing the covariates.

        covariates_names (list): List of the names of the covariates columns.

        question (int): Number of the question that has the outcome.

        tree_depth (int): Number indicating how deep the tree will be.

        min_per_leaf (int): Minimum observations in each leave.

        random_seed (int): Random seed for reproducibility.
    """

    intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=tree_depth, min_samples_leaf=min_per_leaf, random_state=random_seed)
    intrp.interpret(model, x_cov)
    intrp.plot(feature_names=covariates_names)
    plt.savefig(f'bld/tree_question{question}.png', dpi=300, bbox_inches='tight')
    plt.close()

# esta funcion esta fatal, la hice rapido para tener resultados, luego si eso corregir
def printing_some_characteristics(df, model, x_cov, question):
    cate = model.effect(x_cov)
    cate_mean = np.mean(cate)
    grupo_bajo = cate <= cate_mean
    grupo_alto = cate > cate_mean
    grupo_bajo_df = df[grupo_bajo]
    grupo_alto_df = df[grupo_alto]

    print(f'Question{question}')
    print("Big 5")
    print("Mean openness for group 1:", round(grupo_bajo_df["openness"].mean(), 2))
    print("Mean openness for group 2:", round(grupo_alto_df["openness"].mean(), 2))
    print("Mean conscientiousness for group 1:", round(grupo_bajo_df["conscientiousness"].mean(), 2))
    print("Mean conscientiousness for group 2:", round(grupo_alto_df["conscientiousness"].mean(), 2))
    print("Mean extraversion for group 1:", round(grupo_bajo_df["extraversion"].mean(), 2))
    print("Mean extraversion for group 2:", round(grupo_alto_df["extraversion"].mean(), 2))
    print("Mean agreeableness for group 1:", round(grupo_bajo_df["agreeableness"].mean(), 2))
    print("Mean agreeableness for group 2:", round(grupo_alto_df["agreeableness"].mean(), 2))
    print("Mean neuroticism for group 1:", round(grupo_bajo_df["neuroticism"].mean(), 2))
    print("Mean neuroticism for group 2:", round(grupo_alto_df["neuroticism"].mean(), 2))
    print("Not Big 5")
    print("Mean trust_in_science for group 1:", round(grupo_bajo_df["trust_in_science"].mean(), 2))
    print("Mean trust_in_science for group 2:", round(grupo_alto_df["trust_in_science"].mean(), 2))
    print("Mean policy_preferences for group 1:", round(grupo_bajo_df["policy_preferences"].mean(), 2))
    print("Mean policy_preferences for group 2:", round(grupo_alto_df["policy_preferences"].mean(), 2))
    print("Mean age for group 1:", round(grupo_bajo_df["age"].mean(), 2))
    print("Mean age for group 2:", round(grupo_alto_df["age"].mean(), 2))
    print("Mean female for group 1:", round(grupo_bajo_df["female_created"].mean(), 2))
    print("Mean female for group 2:", round(grupo_alto_df["female_created"].mean(), 2))
    print("Mean education for group 1:", round(grupo_bajo_df["education"].mean(), 2))
    print("Mean education for group 2:", round(grupo_alto_df["education"].mean(), 2))

def calculate_RScorer(model_y, model_t, treatment, outcome, covariates, percentage_test, model):

    x_train, x_test, d_train, d_test, y_train, y_test = split_data(treatment, outcome, covariates, percentage_test)

    scorer = RScorer(model_y=model_y, model_t=model_t,
                    discrete_treatment=True, cv=3,
                    mc_iters=3, mc_agg='median')
    scorer.fit(y_test, d_test, X=x_test)

    print(f'RScore: {scorer.score(model)}')


if __name__ == "__main__":

    from pathlib import Path
    from clean_data import clean_dataset
    from normalizing_df import normalize_data

    SRC = Path(__file__).parent.parent.resolve()
    data_path = SRC / "data" / "waves123_augmented_consent.dta"
    BLD = SRC / "bld" 
    waves_data = pd.read_stata(data_path, convert_categoricals=False)

    data_after_cleaning = clean_dataset(waves_data)
    data_done = normalize_data(data_after_cleaning)


    y_list = ["Q1_1", "Q1_2", "Q2_1", "Q2_2"]
    d_list =["Q1_1_treat", "Q1_2_treat", "Q2_1_treat", "Q2_2_treat"]
    my_list_4 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
             "world_issues", "trust_in_science", "empathic_concern_score", "three_tax",
             "born_in_lux", "age", "education", "financialwellbeing"
                ]
    
    x_cov = data_done[my_list_4]

    model_regression = RandomForestRegressor(random_state=23)
    model_propensity = RandomForestClassifier(random_state=23)

    y_1 = data_done["Q1_1"]
    d_var = data_done["Q1_1_treat"] # quite esto, pendiente por si no funciona por esto: .astype(int)  (no creo)

    y_2 = data_done["Q1_2"]

    y_3 = data_done["Q2_1"]

    y_4 = data_done["Q2_2"]

    outcome_list = [y_1, y_2, y_3, y_4]

    for y, d, q in zip(outcome_list, d_var_list, range(1,5)):
            model = generating_causal_forest(model_regression, model_propensity, 100, 10, 0.5, 23, y, d, x_cov)
            graph_distribution_indiv_treatment(model, x_cov, q)

