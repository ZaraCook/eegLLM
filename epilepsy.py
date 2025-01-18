import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('Epilepsy.h5')

# Load your data
df = pd.read_csv('data.csv')

# Extract features (modify column indices as per your dataset structure)
X = df.iloc[:, 1:].values  # Assuming the first column is an index, and the rest are features

# Reshape the data for compatibility with the model
X_test = X.reshape(-1, 178, 1)

# Normalize the data
X_test = (X_test - X_test.mean()) / X_test.std()

# Predict using the model
y_pred = model.predict(X_test)

# Convert predictions to binary class labels
# Threshold is 0.5 because the output is probabilistic for binary classification
y_pred_binary = (y_pred > 0.5).astype(int).flatten()

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.plot(y_pred_binary, label="Predicted Binary Labels", alpha=0.7, color="red")
plt.legend()
plt.title("Predicted Binary Labels (Epileptic: 1, Non-Epileptic: 0)")
plt.xlabel("Sample Index")
plt.ylabel("Predicted Binary Class")
plt.show()

# Save the predictions to a CSV file
output_df = pd.DataFrame({
    "Sample Index": range(len(y_pred_binary)),
    "Predicted Label": y_pred_binary
})
output_df.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")
