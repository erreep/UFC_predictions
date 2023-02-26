import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV


df3=pd.read_csv('df3_101.csv')
wins_dict = defaultdict(lambda: defaultdict(int))
losses_dict = defaultdict(lambda: defaultdict(int))
draws_dict = defaultdict(lambda: defaultdict(int))

def create_dicts(df):
    for fighter, division, result in zip(df['fighter'], df['division'], df['result']):
        wins_dict[fighter][division] += result == 'W'


    for fighter, division, result in zip(df3['fighter'], df['division'], df['result']):
        losses_dict[fighter][division] += result == 'L'

    for fighter, division, result in zip(df3['fighter'], df['division'], df['result']):
        draws_dict[fighter][division] += result == 'D'
    
    return wins_dict, losses_dict, draws_dict


wins_dict, losses_dict, draws_dict = create_dicts(df3)

def result_division(df):

    df['division_wins'] = df.apply(lambda row: wins_dict[row['fighter']][row['division']], axis=1)
    df['division_losses'] = df.apply(lambda row: losses_dict[row['fighter']][row['division']], axis=1)
    df['division_draws'] = df.apply(lambda row: draws_dict[row['fighter']][row['division']], axis=1)
    df['opponent_division_wins'] = df.apply(lambda row: wins_dict[row['opponent']][row['division']], axis=1)
    df['opponent_division_losses'] = df.apply(lambda row: losses_dict[row['opponent']][row['division']], axis=1)
    df['opponent_division_draws'] = df.apply(lambda row: draws_dict[row['opponent']][row['division']], axis=1)
    return df

df3 = pd.get_dummies(df3, columns=['result'])

df3['fighter'] = df3['fighter'].astype('category')
df3['opponent'] = df3['opponent'].astype('category')
df3['division'] = df3['division'].astype('category')
df3['stance'] = df3['stance'].astype('category')
df3['opponent_stance'] = df3['opponent_stance'].astype('category')


fw = np.ones(shape=(len(df3.columns),))


#hyperparameter tuning
# Define the parameter grid for tuning
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 1, 2, 3, 4]
}

# Define the XGBoost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc')



# Define the RandomizedSearchCV object
xgb_random = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, n_iter=100,
                                cv=5, verbose=3, random_state=42, n_jobs=-1)

# Fit the model to the training data
xgb_random.fit(X_train, y_train)

# Make predictions for the test data using the best model
y_pred = xgb_random.best_estimator_.predict(X_test)
predictions = np.round(y_pred).astype(int)

# Evaluate the performance of the best model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Print the confusion matrix for the best model
confusion = multilabel_confusion_matrix(y_test, predictions)
print(confusion)

# Print the best hyperparameters
print("Best hyperparameters: ", xgb_random.best_params_)

#split the data into train and test
X=df3.drop(['result_D', 'result_L', 'result_W'], axis=1)
y=df3[['result_D', 'result_L', 'result_W']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#convert the data into xgboost format
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

#set the parameters
param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.05,  # the training step for each iteration
    'gamma': 2,
    'subsample': 1,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',  # error evaluation for multiclass training
    'eval_metric': 'auc'
}

#set the number of iterations
num_round = 100

#set feature weights
dtrain.set_info(feature_weights=fw)

#train the model
bst = xgb.train(param, dtrain, num_round)

#make predictions for test data
y_pred = bst.predict(dtest)
predictions = np.round(y_pred).astype(int)

#evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#confusion matrix
confusion = multilabel_confusion_matrix(y_test, predictions)
print(confusion)


#pickle the model

filename = 'classifier_ufc.pkl'
pickle.dump(bst, open(filename, 'wb'))
