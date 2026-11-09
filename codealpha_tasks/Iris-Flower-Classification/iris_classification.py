import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

# Convert data into a table
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add flower species
df["species"] = iris.target

# Display first 5 records
print(df.head())

# Display dataset size
print("\nDataset Shape:", df.shape)

# Display flower names
print("\nFlower Names:", iris.target_names)
# Separate input and output

X = df.drop("species", axis=1)

y = df["species"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())
# Split data into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
# Create KNN model

model = KNeighborsClassifier(n_neighbors=5)
# Train the model

model.fit(X_train, y_train)

print("Model training completed!")
# Predict flower species

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)
# Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")
# Convert predictions into flower names

predicted_names = iris.target_names[y_pred]

print("\nPredicted Flower Names:")

print(predicted_names)
# Predict a new flower using manual input

print("\nEnter flower measurements:")

while True:
    try:
        sepal_length = float(input("Sepal length (cm): "))
        sepal_width = float(input("Sepal width (cm): "))
        petal_length = float(input("Petal length (cm): "))
        petal_width = float(input("Petal width (cm): "))

        if min(sepal_length, sepal_width, petal_length, petal_width) <= 0:
            print("Measurements must be positive!")
            continue

        break

    except ValueError:
        print("Please enter valid numbers!")

# Make prediction

new_flower = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=iris.feature_names
)

prediction = model.predict(new_flower)

flower_name = iris.target_names[prediction[0]]

print("\nPredicted Flower Species:", flower_name)