import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

# Load dataset
df = pd.read_csv('./sign_language_landmarks2.csv')

# Check for empty dataset
if df.empty:
    print("CSV is empty. Please run the data extraction script again.")
    exit()

# Extract features and labels
X = df.iloc[:, 1:].values  # Landmark positions
y = df['Label'].values  # Gesture labels

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize landmark values row-wise
X = X / np.max(X, axis=1, keepdims=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build improved model
model = Sequential([
    Flatten(input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_train)), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
print(f'\nCNN Model Accuracy: {test_acc * 100:.2f}%')

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Gesture Confusion Matrix")
plt.show()

# Save model and LabelEncoder
model.save('sign_language_model2.h5')
print("Model saved successfully!")

with open('label_encoder2.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("LabelEncoder saved successfully!")
