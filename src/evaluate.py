import sys
import os
import pickle
import json
import pandas as pd

from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics

model_name=sys.argv[1]
with open(model_name, 'rb') as fd:
    model = pickle.load(fd)

valid_file_name=sys.argv[2]
valid_data=pd.read_csv(valid_file_name)
x=valid_data.drop(['PassengerId','Survived'],axis=1)
labels=valid_data['Survived']


test_file_name=sys.argv[3]
valid_data=pd.read_csv(valid_file_name)
test_data=pd.read_csv(test_file_name)
test=test_data.drop('PassengerId',axis=1)
id=test_data['PassengerId']


predictions_by_class = model.predict_proba(x)
predictions = predictions_by_class[:, 1]
test_predictions=model.predict(test)

precision, recall, _ = precision_recall_curve(labels, predictions)

auc = metrics.auc(recall, precision)

with open('scores.json', 'w') as fd:
    json.dump({'auc': auc}, fd)

with open('prc.json', 'w') as fd:
    json.dump({'prc': [{
            'precision': p,
            'recall': r
        } for p, r in zip(precision, recall)
    ]}, fd)

output = pd.DataFrame({ 'PassengerId' : id, 'Survived': test_predictions })
output.to_csv(os.path.join('results', "submission.csv"))