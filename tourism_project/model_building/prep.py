import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("tourism_project/data/tourism.csv")
df.drop(columns=['CustomerID'], inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Split the data into training and testing sets
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed: train/test splits written.")
print("TypeofContact values kept as:", sorted(X["TypeofContact"].unique()))
