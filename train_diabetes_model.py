import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load the dataset
diabetes = pd.read_csv('diabetes.csv')

# Split the dataset into features (X) and target (y)
X = diabetes.loc[:, diabetes.columns != 'Outcome']
y = diabetes['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)

# Initialize and train the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(knn, file)

print("Model training completed and saved as 'diabetes_model.pkl'")
