import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
from pathlib import Path

my_file = Path("/path/to/file")
# save the iris classification model as a pickle file
model_pkl_file = "accelerometer_MODEL.pkl"  



# Example new data
new_record = {'timeframe': [0, 1, 2, 3], 'acc_x': [0.2, 0.3, 0.4, 0.5], 'acc_y': [0.2, 0.3, 0.4, 0.5], 'acc_z': [0.2, 0.3, 0.4, 0.5]}

record_prediction = None

def extract_features(record):
    features = []
    # Example feature: mean of x, y, z
    features.append(np.mean(record['acc_x']))
    features.append(np.mean(record['acc_y']))
    features.append(np.mean(record['acc_z']))
    # Add more features as needed
    return features

def prediction(): 
    


    clf_previous = RandomForestClassifier(n_estimators=100, random_state=42)
    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(clf_previous, file)


    with open(model_pkl_file, 'rb') as file:  
        clf_previous = pickle.load(file)

    with open(model_pkl_file, 'rb') as file:  
        model = pickle.load(file)

    # evaluate model 
    # new_features_y = extract_features(new_record)
    y_predict = model.predict(new_record)
    print("Predicted Category(ypredict):", y_predict)
    new_features = extract_features(new_record)
    record_prediction = clf_previous.predict([new_features])
    print("Predicted Category:", record_prediction[0])



def model_training(): 
    # Example data format: list of records, each record is a dictionary with timeframe, acceleration (x, y, z), and category
    data = [
        {'timeframe': [0, 1, 2, 3], 'acc_x': [0.1, 0.2, 0.3, 0.4], 'acc_y': [0.1, 0.2, 0.3, 0.4], 'acc_z': [0.1, 0.2, 0.3, 0.4], 'category': 'A'},
        {'timeframe': [0, 1, 2, 3], 'acc_x': [0.5, 0.6, 0.7, 0.8], 'acc_y': [0.5, 0.6, 0.7, 0.8], 'acc_z': [0.5, 0.6, 0.7, 0.8], 'category': 'B'},
        # More records...
    ]

    if record_prediction is not None:
        new_data = new_record
        new_data['category'] = record_prediction[0]
        data.append(new_data)

    # Prepare the dataset
    X = []
    y = []

    for record in data:
        X.append(extract_features(record))
        y.append(record['category'])

    X = np.array(X)
    y = np.array(y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)


    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    
    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(clf, file)


if my_file.is_file():
    prediction()
    model_training()
else: 
    model_training()
    prediction()