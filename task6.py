import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
BASE_PATH = '/Users/palaksharma/Downloads/EXP/'
X_train = np.load(BASE_PATH + 'X_train.npy')
X_test = np.load(BASE_PATH + 'X_test.npy')
y_train = np.load(BASE_PATH + 'y_train.npy')
y_test = np.load(BASE_PATH + 'y_test.npy')

print("Original shape:", X_train.shape)
pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Reduced shape:", X_train_pca.shape)
np.save(BASE_PATH + 'X_train_pca.npy', X_train_pca)
np.save(BASE_PATH + 'X_test_pca.npy', X_test_pca)
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by PCA")
plt.grid()

plt.savefig(BASE_PATH + 'explained_variance.png')
plt.show()
model = LogisticRegression(max_iter=1000)

model.fit(X_train_pca, y_train)

y_pred = model.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy after PCA:", accuracy)
joblib.dump(pca, BASE_PATH + 'pca_model.pkl')

with open(BASE_PATH + 'pca_model_accuracy.txt', 'w') as f:
    f.write(f"Accuracy after PCA: {accuracy}")