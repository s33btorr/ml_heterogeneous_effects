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
from econml.metalearners import TLearner
from econml.dml import DML
from econml.validate.drtester import DRTester

from train_test_split import split_data
from causal_forest import generating_causal_forest


def cf_drtest(outcome, treatment, covariates, question, 
              model_outcome, model_treatment, 
              division_for_cate, percentage_test,
              n_trees, min_sample_per_leaf, max_samples, random_seed
              ):

    """ Saves graphs and print summary of the DRTest.

    Args:
        outcome (column): Column that represents the outcome.

        treatment (column): Column that represents the treatment.

        covariates (DataFrame): DataFrame of all columns representing the covariates.

        model_outcome (object): Model that predicts the Outcome,
        it can be selected from EconML documentation.

        model_treatment (object): Model that predicts the Treatment,
        it can be selected from EconML documentation.

        division_for_cate (int): Number of divisions made in sample for testing.
        If we choose 4, it will be divided into quartiles (which is the default).

        percentage_test (float): Percentage of sample that we use for testing.
        0.4 means 40% of sample is use for testing and the other 60% is use for training.

        n_trees (int): Number of trees wanted in the forest.

        min_sample_per_leaf (int): Minimum number of observations per leaf.

        max_samples (float): In how much the data should be splitted when
        doing the forest. The maximum allowed is 0.5 given Honesty.

        random_seed (int): Seed for reproducibility.
    """


    x_train, x_test, d_train, d_test, y_train, y_test = split_data(treatment, outcome, covariates, percentage_test)
    model = generating_causal_forest(model_outcome, model_treatment, n_trees, min_sample_per_leaf, max_samples, random_seed, y_train, d_train, x_train)

    x_train_t = x_train.to_numpy()
    x_test_t = x_test.to_numpy()
    d_train_t = d_train.to_numpy()
    d_test_t = d_test.to_numpy()
    y_train_t = y_train.to_numpy()
    y_test_t = y_test.to_numpy()

    # ojo, hago esto porque me da error si tiene decimales diferentes el test....
    x_train_t = np.round(x_train_t, 2)
    x_test_t = np.round(x_test_t, 2)
    d_train_t = np.round(d_train_t, 2)
    d_test_t = np.round(d_test_t, 2)
    y_train_t = np.round(y_train_t, 2)
    y_test_t = np.round(y_test_t, 2)

    cf_tester = DRTester(
        model_regression=model_outcome,
        model_propensity=model_treatment,
        cate=model
    ).fit_nuisance(x_test_t, d_test_t, y_test_t, x_train_t, d_train_t, y_train_t)

    res_cf = cf_tester.evaluate_all(x_test_t, x_train_t, n_groups=division_for_cate)

    print(res_cf.summary())


    res_cf.plot_cal(1)
    plt.savefig(f'bld/cal_plot_question{question}.png', dpi=300, bbox_inches='tight')
    plt.close()
    res_cf.plot_qini(1)
    plt.savefig(f'bld/qini_plot_question{question}.png', dpi=300, bbox_inches='tight')
    plt.close()
    #res_cf.plot_toc(1) este no lo voy a guardar hasta entenderlo bien