import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=10)

# Average CV score on the training set was: 0.5084025622486628
exported_pipeline = make_pipeline(
    PCA(iterated_power=6, svd_solver="randomized"),
    StackingEstimator(estimator=BernoulliNB(alpha=0.001, fit_prior=False)),
    SGDClassifier(alpha=0.0, eta0=1.0, fit_intercept=False, l1_ratio=0.25, learning_rate="constant", loss="hinge", penalty="elasticnet", power_t=1.0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 10)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
