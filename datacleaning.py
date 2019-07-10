import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

diabetic_df = pd.read_csv('dataset_diabetes/diabetic_data.csv')
diabetic_df = diabetic_df.drop(['weight', 'payer_code', 'medical_specialty'], axis=1)

diabetic_df.loc[diabetic_df.readmitted != '<30', 'readmitted'] = 0
diabetic_df.loc[diabetic_df.readmitted == '<30', 'readmitted'] = 1

y_value = diabetic_df.loc[:, ['readmitted']]

uniques = diabetic_df.apply(lambda x: x.nunique())
diabetic_df = diabetic_df.drop(uniques[uniques==1].index, axis=1)

diabetic_df = diabetic_df.replace('?', np.nan)
diabetic_df = diabetic_df.dropna(axis=0, how='any')

features = diabetic_df.iloc[:, 21:42]
diabetic_df = diabetic_df.drop(features, axis = 1)
features = pd.get_dummies(features)
# print(features.head())
# print(list(features))

dummy_variables = diabetic_df.loc[:, ['race', 'gender', 'age', 'max_glu_serum', 'A1Cresult']]
dummy_variables = pd.get_dummies(dummy_variables)

for elem in diabetic_df['admission_type_id'].unique():
    diabetic_df['admission_type_id_'+str(elem)] = diabetic_df['admission_type_id'] == elem

for elem in diabetic_df['discharge_disposition_id'].unique():
    diabetic_df['discharge_disposition_id_'+str(elem)] =  diabetic_df['discharge_disposition_id'] == elem

for elem in diabetic_df['admission_source_id'].unique():
    diabetic_df['admission_source_id_'+str(elem)] = diabetic_df['admission_source_id'] == elem


diag = ['diag_1', 'diag_2', 'diag_3']
diag_data = diabetic_df.loc[:, diag]

boolean_columns = ['change', 'diabetesMed']
boolean_data = diabetic_df.loc[:, boolean_columns]

diabetic_df = diabetic_df.drop(['admission_type_id', 'discharge_disposition_id', 'admission_source_id', \
	'race', 'gender', 'max_glu_serum', 'A1Cresult', 'readmitted', 'diag_1', 'diag_2', 'diag_3', 'change', \
	'diabetesMed', 'age'], axis = 1)

diabetic_df = pd.concat([diabetic_df, dummy_variables], axis=1)

diabetic_df = pd.concat([diabetic_df, features], axis=1)

lb_style = LabelEncoder()
relabel_diag_data = diag_data.apply(lb_style.fit_transform)
diabetic_df = pd.concat([diabetic_df, relabel_diag_data], axis=1)

boolean_data = boolean_data.replace('No', 0)
boolean_data = boolean_data.replace('Ch', 1)
boolean_data = boolean_data.replace('Yes', 1)
diabetic_df = pd.concat([diabetic_df, boolean_data], axis=1)

diabetic_df = pd.concat([diabetic_df, y_value], axis=1)

diabetic_df = diabetic_df.dropna(axis=0, how='any')

print(diabetic_df.head(10))
print(diabetic_df.shape)

diabetic_df.to_csv('dataset_diabetes/diabetic_data_cleaned.csv')
