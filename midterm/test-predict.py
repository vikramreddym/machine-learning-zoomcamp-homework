import pickle 

# Load DictVectorizer
with open('dict_vectorizer.pkl', 'rb') as f:
    dv = pickle.load(f)

# Load StandardScaler
with open('standard_scaler.pkl', 'rb') as f:
    std_scaler = pickle.load(f)

# Load XGBClassifier
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

client = {'session_id': 317526,
 'user_id': 16554,
 'product': 'H',
 'campaign_id': 359520,
 'webpage_id': 13787,
 'product_category_1': 4,
 'user_group_id': 3.0,
 'gender': 'Male',
 'age_level': 3.0,
 'user_depth': 1.0,
 'city_development_index': 4.0,
 'var_1': 0,
 'hour': 10}

X = dv.transform([client])
X = std_scaler.transform(X)

print(model.predict(X)[0])