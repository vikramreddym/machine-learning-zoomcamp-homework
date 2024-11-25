# # Ad click prediction
# Dataset link: https://www.kaggle.com/datasets/arashnic/ctr-in-advertisement/data?select=Ad_click_prediction_train+%281%29.csv




import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier     
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV    

import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mutual_info_score, f1_score



df = pd.read_csv("Ad_click_prediction_train (1).csv")


df.columns = df.columns.str.lower()
df["datetime"] = pd.to_datetime(df["datetime"])

# Handle missing values
df.drop(columns=["product_category_2"], inplace=True)

# Drop rows with missing values in the specified columns
columns_to_check = ['user_group_id', 'gender', 'age_level', 'user_depth']
df = df.dropna(subset=columns_to_check)


null_indices = df[df['city_development_index'].isnull()].index

# Generate random values (1, 2, 3, 4) with equal probabilities
random_values = np.random.choice([1, 2, 3, 4], size=len(null_indices), replace=True)

# Assign random values to the null indices
df.loc[null_indices, 'city_development_index'] = random_values


# Create a train/test split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
y_train = df_train.is_click.values
y_test = df_test.is_click.values

del df_train["is_click"]
del df_test["is_click"]


# ### Prepare data

# Train data
df_train['hour'] = df_train['datetime'].dt.hour
df_train.drop(columns=["datetime"], inplace=True)

dv = DictVectorizer(sparse=False)
train_dicts = df_train.to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)
dv.get_feature_names_out()

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)

# Test data
df_test['hour'] = df_test['datetime'].dt.hour
df_test.drop(columns=["datetime"], inplace=True)

test_dicts = df_test.to_dict(orient="records")
X_test = dv.transform(test_dicts)

X_test = std_scaler.transform(X_test)


# using the model

xgb = XGBClassifier(learning_rate=0.01, max_depth=7, min_child_weight=5, n_estimators=200, scale_pos_weight=scale_pos_weight)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
print("Test F1 Score:", f1_score(y_test, y_pred))



# Save DictVectorizer
with open('dict_vectorizer.pkl', 'wb') as f:
    pickle.dump(dv, f)

# Save StandardScaler
with open('standard_scaler.pkl', 'wb') as f:
    pickle.dump(std_scaler, f)

# Save XGBClassifier
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)





