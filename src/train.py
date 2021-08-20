from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import yaml
import pickle
import sys

#parameters
depth = yaml.safe_load(open('params.yaml'))['train']['depth']
trees = yaml.safe_load(open('params.yaml'))['train']['trees']
split = yaml.safe_load(open('params.yaml'))['train']['split']


#load train data
train_file_name=sys.argv[1]
train_data=pd.read_csv(train_file_name)

# model
model_name=sys.argv[2]

x_train=train_data.drop(['PassengerId','Survived'],axis=1)
y_train=train_data['Survived']

# model training 
model = RandomForestClassifier(n_estimators=trees ,max_depth=depth ,min_samples_split = split ,random_state=0)
model.fit(x_train, y_train)

# save model
with open(model_name, 'wb') as fd:
    pickle.dump(model, fd)