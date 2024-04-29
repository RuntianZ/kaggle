# %%
# Random forest with catboost

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def prepare_dataset(df):
    # For categorical features, replace NA with "Missing"
    categorical_columns = df.select_dtypes(include=['object']).columns
    for key in categorical_columns:
        df[key] = df[key].astype('category').cat.add_categories("Missing").fillna("Missing")
    categorical_columns = df.select_dtypes(include=['category']).columns
    return df, categorical_columns

# %%
train_file_path = "data/house-prices/train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.shape))
dataset_df.head(3)

# %%
dataset_df, categorical_columns = prepare_dataset(dataset_df)
# dataset_df[categorical_columns].head(10)

# %%
# Prepare dataset
target_column = 'SalePrice'
y = dataset_df.pop(target_column)
X = dataset_df.drop('Id', axis=1)




# %% [markdown]
# # Cross Validation

# %%
# Grid Search
from sklearn.model_selection import GridSearchCV

# Define the model
model = CatBoostRegressor(
    boosting_type='Plain',         # Use random forest mode
    # bootstrap_type='Poisson',    # Random sampling of data points
    task_type='GPU',               # needed for Poisson            
    verbose=200,                    # Output training progress every 100 iterations
    cat_features=categorical_columns.to_list()
)

# Set up the parameter grid
param_grid = {
    'bootstrap_type': ['Poisson'],
    'depth': [4, 6, 8, 10],
    'iterations': [500, 1000, 2000],
    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
}

# Configure GridSearchCV
# When cv=None, default is 5-fold cross validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=None, scoring='neg_mean_squared_error', verbose=2)

# Fit GridSearchCV
grid_search.fit(X, y)

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best RMSE:", (-grid_search.best_score_) ** 0.5)


# %%
# Initialize CatBoostRegressor
model = CatBoostRegressor(
    boosting_type='Plain',         # Use random forest mode
    task_type='GPU',               # GPU is needed for Poisson
    # random_seed=42,                # Seed for reproducibility
    verbose=100,                   # Output training progress every 100 iterations
    cat_features=categorical_columns.to_list(),
    **grid_search.best_params_
)

# Train the model
model.fit(X, y)

# %%
feature_importances = model.get_feature_importance()
print(pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False))

# %%
test_file_path = "data/house-prices/test.csv"
test_data = pd.read_csv(test_file_path)
ids = test_data.pop('Id')

test_data, _ = prepare_dataset(test_data)
# test_data.head(3)

# %%
sample_submission_df = pd.read_csv('data/house-prices/sample_submission.csv')
sample_submission_df['SalePrice'] = model.predict(test_data)
sample_submission_df.to_csv('working/catboost-rf-poisson-auto.csv', index=False)
sample_submission_df.head()
