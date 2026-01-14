
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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
# # -- SEPARATING VARIABLES IN NOT USED - NUMBER
#%% 

blacklist = ['Class'] # TARGET COLUMN

num_vars = [
    col for col in X_train.columns
        if col not in blacklist
]   # COLUMNS USED TO PREDICT TARGET RESULT

#%% [markdown]
# # -- PIPELINE --

#%%
# NUMBER PIPELINE TRANSFORMATION
num_pipe = Pipeline([
    ('scaler', RobustScaler())
])

#%% [markdown]
# ## --- PREPROCESSOR (AUTOMATIZE NUMBER-STR-DATE-CODE TRANSFORMATION) ---
#%%

preprocessor = ColumnTransformer(
    transformers=[
    ('tr_num', num_pipe, num_vars)
    ],
    remainder='drop'
)

#%% [markdown]
# ## --- FINAL PIPELINE (ENCAPSULATING ALL TRANSFORMATION IN PREPROCESSOR AND OTHER PIPELINES) ---
#%%

final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('modelo', KNeighborsClassifier(n_jobs=5, n_neighbors=6))],
    memory=None
)

#%% [markdown]
# ## --- FINAL PIPELINE PARAMS (TESTING DIFFERENT MODELS TO DISCOVER THE BEST) ---
#%%
# ratio = (float((y_train == 0).sum() / (y_train == 1).sum())) * 2

params = [
    # --- CENÁRIO 1: LIGHTGBM (O Veloz - Microsoft) - ATENÇÃO: É muito leve e rápido.  ---
    {
        'modelo': [LGBMClassifier(n_jobs=1, force_col_wise=True)],      # DEFINING NUMBER OS PROCESSOR USED IN PROCESS
        'modelo__n_estimators': [3000],                                 # ESTIMATED NUMBER OF TREES
        'modelo__learning_rate': [0.01],                                # STEP SIZE (ETA) | LOW = MOST ACCURATE, BUT TAKES MORE TIME
        'modelo__num_leaves': [100],                                    # NUMBER OF NODES IN EACH BRANCH (MUCH HIGH VALUES = HIGH OVERFITTING PROBABILITY ) | MAIN PARAM FROM LIGHTGBM
        'modelo__max_depth': [-1],                                      # SIZE OF EACH TREE BRANCH
        'modelo__class_weight' : ['balanced'],                          # DEFINING WEIGHTS FOR CLASS FRAUD
        'modelo__min_child_samples': [3],                               # DEFINING AMOUNT OF VALUES TO CAPTURE THE PATTERN
        'modelo__subsample': [0.9],                                     # % RAMDOM SAMPLES OF ROWS FOR EACH TRAIN IN EACH BRANCH 
        'modelo__colsample_bytree': [0.9],                              # % RAMDOM SAMPLES OF COLS FOR EACH TRAIN IN EACH BRANCH |
        'modelo__importance_type': ['gain'],                            # GAIN DEFINE QUALITY AGAINST AMOUNT
        'modelo__boosting_type': ['gbdt'],                              # CHANGE THE MODEL ENGINE 
        'modelo__objective':['binary']                                  # DEFINE HOW THE VALUE OF COLUMN TARGET IS
    }
]


#%% [markdown]
# ## --- SPLIT TIME SERIES AND CONFIG GRIDSEARCH ---
#%%
# CONFIGURATION GRIDSERACH PARAMS
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# CONFIGURATION GRIDSERACH PARAMS
grid = GridSearchCV(
    final_pipe,
    param_grid = params,
    cv = kfold,
    scoring = 'recall',
    verbose = 2,
    n_jobs = -1
)

#%% [markdown]
# ## --- FITTING MODEL GRIDSEARCH ---
#%%
grid.fit(X_train, y_train)

#%% [markdown]
# # --- MODEL PREDICT ---
#%%

grid_predict = grid.predict(X_test) # GENERATING PREDICT W/ X_TEST
grid_predict_proba = grid.predict_proba(X_test)[:,1] # GENERATING PREDICT PROBABILITY W/ X_TEST

#%% [markdown]
# # --- ANALYSE METRICS ---
#%%

new_limit = 0.10 # DEFINE MANUALLY
y_pred_ajust = (grid_predict_proba >= new_limit).astype(int)  # DEFINING AJUST PROBABILITIES

# FEATURE IMPORTANCE TO UNDERSTAND BEST USED FEATURES
best_model = grid.best_estimator_.named_steps['modelo']
importance = best_model.feature_importances_
feature_names = X_train.columns

# CLASSIFICATION REPORT
class_report = classification_report(y_test, y_pred_ajust)
print("\n" + "="*40)
print(F'📋 CLASSIFICATION REPORT: MODEL {best_model}')
print("="*40)
print(f'Class Report: {class_report}') 

# CONFUSION MATRIX
cm = confusion_matrix(y_test, grid_predict)
print("\n" + "="*40)
print(F'📋 CONFUSION MATRIX REPORT: {best_model}')
print("="*40)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=['Not Fraud','Fraud'])
fig, ax = plt.subplots(figsize=(8,6))
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title("📋 CONFUSION MATRIX")
plt.show()

# DATAFRAME PROBABILITIES OF UNDETECTED FRAUD TRANSACTIONS
y_pred = grid_predict
y_proba = grid_predict_proba
df_analyse = pd.DataFrame({
    'Real': y_test,
    'Prev': y_pred_ajust,
    'Prob': y_proba,
})
fatal_error = df_analyse[(df_analyse['Real'] == 1) & (df_analyse['Prev'] == 0)]

print('Probability of undetected fraud transactions: ')
fatal_error.sort_values(by=['Prob'], ascending=False)

#%% [markdown]
# # --- FINANCIAL RETURN ---
#%%
real_values = df.loc[X_test.index, 'Amount']

df_return = pd.DataFrame({
    'Real': y_test,
    'Prev': y_pred_ajust,
    'Amount': real_values
})

detected_frauds = df_return[(df_return['Real'] == 1) & (df_return['Prev'] == 1)]
money_economy = detected_frauds['Amount'].sum()

undetected_frauds = df_return[(df_return['Real'] == 1) & (df_return['Prev'] == 0)]
money_loss = undetected_frauds['Amount'].sum()

print(f'Money saved by model: {money_economy:.0f}')
print(f'Money lost (ERRORS): {money_loss:.0f}')
print(f'Money saved rate: {money_economy / (money_economy+money_loss):.1%}')

categories = ['Money Saved','Money loss']
values = [money_economy, money_loss]
color = ['#2ecc71', '#e74c3c']

plt.figure(figsize=(10,6))
bars = plt.bar(categories, values, color=color, width=0.6)
plt.title('Financial model impact', fontsize=16, pad=20)
plt.ylabel('Monetary Value', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
sns.despine()

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.,
                 height + (max(values) * 0.01), 
                 f'${height:,.0f}',
                 ha='center', va='bottom', fontsize=14, color="#080808")

add_labels(bars)
plt.tight_layout()
plt.show()

#%% [markdown]
# ## -- ANALYSING BEST FEATURES --
#%%

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
    'model': grid,
}

joblib.dump(final_pack, 'Models/model_fraud_V2.pkl')
print(f'✅ Model saved sucessfully! Archive "model_fraud_V2" was created.')

#%% [markdown]
# # --- TESTING NEW PREDICT ---
#%% [markdown]
# ## -- LOADING MODEL --
#%%

model = joblib.load('models/model_fraud_V2.pkl')

model_prod = model['model']

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
