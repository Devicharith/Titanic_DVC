import os
import pandas as pd
import yaml
import sys
from sklearn.model_selection import train_test_split


# parameters

split=yaml.safe_load(open('params.yaml'))['prepare']['split']

# import data

train_file_name=sys.argv[1]
test_file_name=sys.argv[2]

train_data=pd.read_csv(train_file_name)
test_data=pd.read_csv(test_file_name)
# pre-processing 

train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)

# filling missing values

train_data.fillna(0,inplace=True)
test_data.fillna(0,inplace=True)


# create a combine group 

combine=[train_data,test_data]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

#drop the name feature since it contains no more useful information.
train_data = train_data.drop(['Name'], axis = 1)
test_data = test_data.drop(['Name'], axis = 1)

#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train_data['Sex'] = train_data['Sex'].map(sex_mapping)
test_data['Sex'] = test_data['Sex'].map(sex_mapping)


#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3,0 : 0}
train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)

# dropping fare values
train_data = train_data.drop(['Fare'], axis = 1)
test_data = test_data.drop(['Fare'], axis = 1)

# train validation split 
train,valid=train_test_split(train_data,test_size=split)

# create folder to save file
data_path = 'prepared'
os.makedirs(data_path, exist_ok=True)

# saving prepared data
train.to_csv(os.path.join(data_path, "out_train.csv"))
valid.to_csv(os.path.join(data_path, "out_validation.csv"))
test_data.to_csv(os.path.join(data_path,"out_test.csv"))