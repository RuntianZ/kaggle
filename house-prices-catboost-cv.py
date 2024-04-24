# %%
import pandas as pd
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# %%
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
# dataset_df.head(3)

# %%
dataset_df, categorical_columns = prepare_dataset(dataset_df)
# dataset_df[categorical_columns].head(10)

# %%
# Prepare dataset
target_column = 'SalePrice'
y = dataset_df.pop(target_column)
X = dataset_df.drop('Id', axis=1)


# %%
# Grid Search
from sklearn.model_selection import GridSearchCV

# Define the model
model = CatBoostRegressor()

# Set up the parameter grid
param_grid = {
    'depth': [4, 6, 8, 10, 12],
    'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.12, 0.15, 0.18, 0.2],
    'iterations': [500, 1000, 1500, 2000, 2500, 3000, 4000]
}

# Configure GridSearchCV
# When cv=None, default is 5-fold cross validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=None, scoring='neg_mean_squared_error', verbose=2)

# Fit GridSearchCV
grid_search.fit(X, y, cat_features=categorical_columns.to_list(), verbose=200)

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best RMSE:", (-grid_search.best_score_) ** 0.5)

# %%
# Initialize CatBoostRegressor
model = CatBoostRegressor(
    cat_features=categorical_columns.to_list(),
    verbose=200,
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
sample_submission_df.to_csv('working/catboost-cv-auto.csv', index=False)
sample_submission_df.head()


