
# coding: utf-8

# # Create a logistic regression model to predict TP53 mutation from gene expression data in TCGA

# In[178]:

import os
import urllib
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from statsmodels.robust.scale import mad


# In[179]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# ## Specify model configuration

# In[180]:

# We're going to be building a 'TP53' classifier 
GENE = '7157' # TP53


# *Here is some [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) regarding the classifier and hyperparameters*
# 
# *Here is some [information](https://ghr.nlm.nih.gov/gene/TP53) about TP53*

# ## Load Data

# In[181]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('download', 'expression-matrix.tsv.bz2')\nexpression = pd.read_table(path, index_col=0)")


# In[182]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('download', 'mutation-matrix.tsv.bz2')\nY = pd.read_table(path, index_col=0)")


# In[183]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('download', 'covariates.tsv')\ncovariates = pd.read_table(path, index_col=0)\n\n# Select acronym_x and n_mutations_log1p covariates only\nselected_cols = [col for col in covariates.columns if 'acronym_' in col]\nselected_cols.append('n_mutations_log1p')\ncovariates = covariates[selected_cols]")


# In[184]:

y = Y[GENE]


# In[185]:

# The Series now holds TP53 Mutation Status for each Sample
y.head(6)


# In[186]:

# Here are the percentage of tumors with NF1
y.value_counts(True)


# ## Pre-process data set
# TODO: currently running PCA on both train and test partitions

# In[187]:

# Pre-process expression data for use later
n_components = 65
scaled_expression = StandardScaler().fit_transform(expression)
pca = PCA(n_components).fit(scaled_expression)
explained_variance = pca.explained_variance_
expression_pca = pca.transform(scaled_expression)
expression_pca = pd.DataFrame(expression_pca)
expression_pca = expression_pca.set_index(expression.index.values)


# In[188]:

print('fraction of variance explained: ' + str(pca.explained_variance_ratio_.sum()))


# In[189]:

# Create full feature matrix (expression + covariates)
X = pd.concat([covariates,expression_pca],axis=1)
print('Gene expression matrix shape: {0[0]}, {0[1]}'.format(expression.shape))
print('Full feature matrix shape: {0[0]}, {0[1]}'.format(X.shape))


# ## Set aside 10% of the data for testing

# In[190]:

# Typically, this can only be done where the number of mutations is large enough
train_index, test_index = next(ShuffleSplit(n_splits=2, test_size=0.1, random_state=0).split(y))

X_partitions = {
    'full': {
        'train': X.ix[train_index], 
        'test': X.ix[test_index]
        },
    'expressions': {
        'train': expression_pca.ix[train_index], 
        'test': expression_pca.ix[test_index]
        },
    'covariates': {
        'train': covariates.ix[train_index], 
        'test': covariates.ix[test_index]
        }    
    } 

y_train = y[train_index]
y_test = y[test_index]

'Size: {:,} features, {:,} training samples, {:,} testing samples'.format(
    len(X_partitions['full']['train'].columns), 
    len(X_partitions['full']['train']), 
    len(X_partitions['full']['test']))


# ## Median absolute deviation feature selection

# In[191]:

def fs_mad(x, y):
    """    
    Get the median absolute deviation (MAD) for each column of x
    """
    scores = mad(x) 
    return scores, np.array([np.NaN]*len(scores))


# ## Define pipeline and Cross validation model fitting

# In[192]:

# Parameter Sweep for Hyperparameters
param_grid = {
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': [10 ** x for x in range(-3, 1)],
    'classify__l1_ratio': [0, 0.2, 0.8, 1],
}

pipeline = Pipeline(steps=[
    ('standardize', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced'))
])

models = ['full', 'expressions', 'covariates']

cv_pipelines = {mod: GridSearchCV(estimator=pipeline, 
                             param_grid=param_grid, 
                             n_jobs=1, 
                             scoring='roc_auc') for mod in models}


# In[193]:

get_ipython().run_cell_magic('time', '', "for model, pipeline in cv_pipelines.items():\n    print('Fitting CV for model: {0}'.format(model))\n    pipeline.fit(X=X_partitions.get(model).get('train'), y=y_train)\n# cv_pipeline_full.fit(X=X_train_full, y=y_train)")


# In[194]:

# Best Params
for model, pipeline in cv_pipelines.items():
    print('{0}: {1:.3%}'.format(model, pipeline.best_score_))

    # Best Params
    print(pipeline.best_params_)


# ## Visualize hyperparameters performance

# In[195]:

cv_results_df_dict = {model: 
    pd.concat([
        pd.DataFrame(pipeline.cv_results_),
        pd.DataFrame.from_records(pipeline.cv_results_['params']),
    ], axis='columns') for model, pipeline in cv_pipelines.items()}

model = 'full'

cv_results_df_dict[model].head(2)


# In[196]:

# Cross-validated performance heatmap
model = 'full'

cv_score_mat = pd.pivot_table(cv_results_df_dict[model],
                              values='mean_test_score', 
                              index='classify__l1_ratio',
                              columns='classify__alpha')
ax = sns.heatmap(cv_score_mat, annot=True, fmt='.1%')
ax.set_xlabel('Regularization strength multiplier (alpha)')
ax.set_ylabel('Elastic net mixing parameter (l1_ratio)');


# ## Use Optimal Hyperparameters to Output ROC Curve

# In[197]:

y_pred_dict = {
    model: {
        'train': pipeline.decision_function(X_partitions[model]['train']),
        'test':  pipeline.decision_function(X_partitions[model]['test'])
    } for model, pipeline in cv_pipelines.items()
}

def get_threshold_metrics(y_true, y_pred):
    roc_columns = ['fpr', 'tpr', 'threshold']
    roc_items = zip(roc_columns, roc_curve(y_true, y_pred))
    roc_df = pd.DataFrame.from_items(roc_items)
    auroc = roc_auc_score(y_true, y_pred)
    return {'auroc': auroc, 'roc_df': roc_df}

metrics_dict = {    
    model: {
        'train': get_threshold_metrics(y_train, y_pred_dict[model]['train']),
        'test':  get_threshold_metrics(y_test, y_pred_dict[model]['test'])
    } for model in y_pred_dict.keys()
}


# In[198]:

# Plot ROC
plt.figure()
for model, metrics_partition in metrics_dict.items():
    for partition, metrics in metrics_partition.items():
        model_name = '{0} {1}'.format(model, partition)
        roc_df = metrics['roc_df']
        plt.plot(roc_df.fpr, roc_df.tpr,
            label='{} (AUROC = {:.1%})'.format(model_name, metrics['auroc']))
        
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Predicting TP53 mutation from gene expression (ROC curves)')
plt.legend(loc='lower right');


# ## What are the classifier coefficients?

# In[199]:

final_pipelines = {
    model: pipeline.best_estimator_
    for model, pipeline in cv_pipelines.items()
}
final_classifiers = {
    model: pipeline.named_steps['classify']
    for model, pipeline in final_pipelines.items()
}


# In[200]:

def get_coefficients(classifier, X_mat):
    coef_df = pd.DataFrame.from_items([
        ('feature', X_mat.columns),
        ('weight', classifier.coef_[0]),
    ])

    coef_df['abs'] = coef_df['weight'].abs()
    coef_df = coef_df.sort_values('abs', ascending=False)
    
    return coef_df

coef_df_dict = {
    model: get_coefficients(classifier, X_partitions[model]['train'])
    for model, classifier in final_classifiers.items()
}


# In[201]:

model = 'full'

print('{:.1%} zero coefficients; {:,} negative and {:,} positive coefficients'.format(
    (coef_df_dict[model].weight == 0).mean(),
    (coef_df_dict[model].weight < 0).sum(),
    (coef_df_dict[model].weight > 0).sum()
))
coef_df.head(10)


# ## Investigate the predictions

# In[202]:

model = 'full'

X_all = X_partitions[model]['train'].append(X_partitions[model]['test'])
X_test_index = X_partitions[model]['test'].index
y_all = y_train.append(y_test)

predict_df = pd.DataFrame.from_items([
    ('sample_id', X_all.index),
    ('testing', X_all.index.isin(X_test_index).astype(int)),
    ('status', y_all),
    ('decision_function', final_pipelines[model].decision_function(X_all)),
    ('probability', final_pipelines[model].predict_proba(X_all)[:, 1])
])

predict_df['probability_str'] = predict_df['probability'].apply('{:.1%}'.format)


# In[203]:

# Top predictions amongst negatives (potential hidden responders)
predict_df.sort_values('decision_function', ascending=False).query("status == 0").head(10)


# In[204]:

# Ignore numpy warning caused by seaborn
warnings.filterwarnings('ignore', 'using a non-integer number instead of an integer')

ax = sns.distplot(predict_df.query("status == 0").decision_function, hist=False, label='Negatives')
ax = sns.distplot(predict_df.query("status == 1").decision_function, hist=False, label='Positives')


# In[205]:

ax = sns.distplot(predict_df.query("status == 0").probability, hist=False, label='Negatives')
ax = sns.distplot(predict_df.query("status == 1").probability, hist=False, label='Positives')

