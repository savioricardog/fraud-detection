#%% [markdown]
# # --- Imports ---
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

# from sklearn.compose import 
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
# # --- UNDERSTANDING DATASET - EDA ---
#%%
#%% [markdown]
# ## -- FIRST IMPRESSIONS --
#%%
# DATASET INFOS
print(df.info())
print(f'\n Shape df: {df.shape}')
#%%
# UNDERSTAND TARGET VARIABLE
df_class_count = df['Class'].value_counts() # COUNTING HOW MUCH VALUES WE HAVE TO ZERO VALUE AND ONE VALUE
df_class_count
#%%
# PLOTING CLASS VALUE COUNTS

# DISTRIBUTION ZEROS VS ONES
plt.figure(figsize=(5,5))
sns.barplot(data=df_class_count)
plt.show()

#%%
# ANALYSING FINANCE BEHAVIOR
# df_behavior = df.groupby('Class')['Amount'].describe().reset_index()
#%%

plt.figure(figsize=(5,5))
sns.boxplot(x='Class', y='Amount', data=df, hue = 'Class')
# plt.yscale('log')
plt.show()

#%% [markdown]
# ## -- X/y : TRAIN/TEST --
#%%
target = 'Class'
X, y = df.drop(columns=[target], errors='ignore'), df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    stratify=y,
                                                    test_size=0.25)


#%%

# INSTANTING ROBUSTSCALER OBJECTS
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

#%%
# PRINTING FOR ENSURE TRANSFORMATION
print(f"New X_train (with scaled columns): {X_train.shape[1]}")
print(X_train.head(3))

#%%
# FIRST FIT

log_reg = LogisticRegression(max_iter=500, penalty='l2', solver='saga') # INSTANTING MODEL
first_fit = log_reg.fit(X_train, y_train) # FITTING X_TRAIN AND Y_TRAIN

#%%
# FIRST PREDICT
first_predict = first_fit.predict(X_test) # GENERATING PREDICT W/ X_TEST

#%%
# PREDICT METRICS. PREDICT VS Y_TEST
class_report = classification_report(first_predict, y_test)
print(f'Classification report:\n{class_report}')

conf_matrix = confusion_matrix(first_predict, y_test)
print(f'Confusion Matrix:\n{conf_matrix}')




# #%% [markdown]
# # # -- SEPARATING VARIABLES IN NOT USED - NUMBER
# #%% 

# blacklist = ['Class'] # TARGET COLUMN

# num_vars = [
#     col for col in X_train.columns
#         if col not in blacklist
# ]   # COLUMNS USED TO PREDICT TARGET RESULT

# #%% [markdown]
# # # -- PIPELINE --

# #%%
# # NUMBER PIPELINE TRANSFORMATION
# num_pipe = Pipeline([
#     ('scaler', RobustScaler())
# ])

# #%% [markdown]
# # ## --- PREPROCESSOR (AUTOMATIZE NUMBER-STR-DATE-CODE TRANSFORMATION) ---
# #%%

# preprocessor = ColumnTransformer(
#     transformers=[
#     ('tr_num', num_pipe, num_vars)
#     ],
#     remainder='drop'
# )

# #%%

# preprocessor

# #%% [markdown]
# # ## --- FINAL PIPELINE (ENCAPSULATING ALL TRANSFORMATION IN PREPROCESSOR AND OTHER PIPELINES) ---
# #%%

# final_pipe = Pipeline([
#     ('preprocessor',preprocessor),
#     ('modelo', KNeighborsClassifier(n_jobs=5, n_neighbors=6))],
#     memory=None
# )