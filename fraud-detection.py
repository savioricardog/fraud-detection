#%% [markdown]
# # --- IMPORTS ---
#%%
import kagglehub
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from feature_engine import discretisation, encoding
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.append(os.path.abspath(os.path.join('..')))

#%% [markdown]
# ## -- CONFIGURING JUPYTER PAGE --
#%%
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)

%load_ext autoreload
%reload_ext autoreload
%autoreload 2

#%% [markdown]
# ## -- DOWNLOAD DATASET LATEST VERSION --
#%%
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)

#%% [markdown]
# ## -- VARIABLES CONFIG --
#%%
path_dataset = os.path.join(path, 'creditcard.csv')

#%% [markdown]
# # -- READ AND SAMPLE DATASET --
#%%
df = pd.read_csv(path_dataset)
df.head(3)

#%% [markdown]
# # -- UNDERSTANDING DATASET - EDA --
#%% [markdown]
# ## -- DATASET GENERAL INFOS --
#%%
print(df.info())
print(f'\n Shape df: {df.shape}')
#%% [markdown]
# ## -- DATASET TARGET ANALYSE --
#%%
df_class_count = df['Class'].value_counts() # COUNTING HOW MUCH VALUES WE HAVE TO ZERO VALUE AND ONE VALUE
df_class_count

#%% [markdown]
# ## -- BARPLOT CLASS VALUE COUNTS BASED ON TARGET --
#%%
plt.figure(figsize=(5,5))
sns.barplot(data=df_class_count)
plt.show()

#%% [markdown]
# ## -- BOXPLOT COMPARING TARGET VS AMOUNT FEATURE --
#%%

plt.figure(figsize=(5,5))
sns.boxplot(x='Class', y='Amount', data=df, hue = 'Class')
plt.yscale('log')
plt.show()

#%% [markdown]
# # --- X/y : TRAIN/TEST ---
#%%
target = 'Class'
X, y = df.drop(columns=[target], errors='ignore'), df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    stratify=y,
                                                    test_size=0.25)


#%% [markdown]
# # --- TRANSFORMING VALUES NOT STANDARDIZED ---
#%%

# INSTANTING ROBUST SCALER
scaler_amount = RobustScaler()
scaler_time = RobustScaler()

# CREATING FEATURES SCALEDS IN TRAIN X (USING [[]] FOR ENSURE 2D DATAFRAME)
X_train['scaled_amount'] = scaler_amount.fit_transform(X_train[['Amount']])
X_train['scaled_time'] = scaler_time.fit_transform(X_train[['Time']])


# TEST ONLY TRANSFORM (NOT FIT X_TEST) | CANT USE "FIT" IN TRAINING
X_test['scaled_amount'] = scaler_amount.transform(X_test[['Amount']])
X_test['scaled_time'] = scaler_time.transform(X_test[['Time']])

X_train.drop(columns=['Amount','Time'], axis=1, inplace=True)
X_test.drop(columns=['Amount','Time'], axis=1, inplace=True)


print(f"New X_train (with scaled columns): {X_train.shape[1]}")
print(X_train.head(3))

#%% [markdown]
# # --- INSTANTING AND FITTING MODEL ---
#%%

lightgbm = LGBMClassifier(n_estimators=3000,
                        learning_rate= 0.01,
                        num_leaves=95,
                        max_depth=-1,
                        class_weight='balanced',
                        min_child_samples = 3,
                        subsample  = 0.9,
                        colsample_bytree = 0.9,
                        n_jobs = 5,
                        importance_type='gain',
                        objective = 'binary',
                        verbose = -1
)

first_fit = lightgbm.fit(X_train, y_train) # FITTING X_TRAIN AND Y_TRAIN

#%% [markdown]
# # --- MODEL PREDICT ---
#%%

first_predict = first_fit.predict(X_test) # GENERATING PREDICT W/ X_TEST
first_predict_proba = first_fit.predict_proba(X_test)[:,1]


#%% [markdown]
# # --- ANALYSE METRICS ---
#%%

novo_limite = 0.30 
y_pred_ajustado = (first_predict_proba >= novo_limite).astype(int)

# CLASSIFICATION REPORT
class_report = classification_report(y_test, y_pred_ajustado)
print("\n" + "="*40)
print(F'📋 CLASSIFICATION REPORT: MODEL {lightgbm}')
print("="*40)
print(class_report) 

# CONFUSION MATRIX
cm = confusion_matrix(y_test, first_predict)
print("\n" + "="*40)
print(F'📋 CONFUSION MATRIX REPORT: {lightgbm}')
print("="*40)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=['N Fraude','Fraude'])
fig, ax = plt.subplots(figsize=(8,6))
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title("📋 CONFUSION MATRIX")
plt.show()

#%%

y_pred = first_predict
y_proba = first_predict_proba

df_analise = pd.DataFrame({
    'Real': y_test,
    'Previsão': y_pred_ajustado,
    'Probabilidade': y_proba
})


erros_fatais = df_analise[(df_analise['Real'] == 1) & (df_analise['Previsão'] == 0)]

print('As fraude que o modelo perdeu tinham estas probabilidades: ')
erros_fatais.sort_values(by=['Probabilidade'], ascending=False)
#%%

#%% [markdown]
# ## -- ANALYSING BEST FEATURES --
#%%

# FEATURE IMPORTANCE TO UNDERSTAND BEST USED FEATURES
importance = lightgbm.feature_importances_
feature_names = X_train.columns


feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis', legend=False, hue='Feature')
plt.title('Best Features for the Model')
plt.show()

#%% [markdown]
# # --- SAVING MODEL ---
#%%

final_pack = {
    'model': lightgbm,
    'scaler_amount': scaler_amount,
    'scaler_time': scaler_time
}

joblib.dump(final_pack, 'Models/model_fraud_V1.pkl')
print(f'✅ Model saved sucessfully! Archive "model_fraud_V1" was created.')


#%% [markdown]
# # --- TESTING NEW PREDICT ---
#%% [markdown]
# ## -- LOADING MODEL --
#%%

model = joblib.load('models/model_fraud_V1.pkl')

model_prod = model['model']
scaler_amount_prod = model['scaler_amount']
scaler_time_prod = model['scaler_time']

print('='*40)
print('✅ MODEL SUCESSFULLY LOADED')
print('='*40)

#%% [markdown]
# ## -- GENERATING NEW DATA ---
#%%

# GENERATING NEW DATA (SIMULATION)
new_data = X_test.iloc[0:1].copy()
new_data['Amount'] = 1500.00
new_data['Time'] = 50

# STANDATIZATION NEW DATA
new_data['scaled_amount'] = scaler_amount_prod.transform(new_data[['Amount']])
new_data['scaled_time'] = scaler_time_prod.transform(new_data[['Time']])
new_data.drop(columns=['Amount','Time'], axis=1, inplace=True)

#%% [markdown]
# ## -- GENERATING NEW PREDICT ---
#%%

prev = model_prod.predict(new_data)
prob = model_prod.predict_proba(new_data)


print('\n --- ANALYSE RESULT ---')
if prev [0] == 1:
    print('🚨 ALERT: FRAUD DETECTED!')
else:
    print("✅ REGULAR TRANSACTION APPROVED")

print(F"MODEL CONFIDENCE: {prob[0][prev[0]]:.2%} OF ENSURE")
