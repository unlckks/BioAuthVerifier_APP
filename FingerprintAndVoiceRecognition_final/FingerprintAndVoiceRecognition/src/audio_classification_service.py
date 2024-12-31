import numpy as np
import librosa
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, regularizers
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize FastAPI
app = FastAPI()

# Path to the trained model
model_path = "/Users/mingyunzhang/Desktop/FingerprintAndVoiceRecognition/src/dataset/updated_model.h5"
# Note: The structure of this model is based on the open source model and fine-tuned. The following code is fine-tuned using chatgpt.
# Function to create the model
def create_model(input_shape=(40, 500, 1)):
    model = Sequential([
        layers.InputLayer(input_shape=input_shape),
        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Third convolutional block
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Reshape for recurrent layers
        layers.Reshape((-1, 128)),

        # Bidirectional LSTM layers
        layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
        layers.BatchNormalization(),
        layers.Bidirectional(layers.LSTM(128, dropout=0.3, recurrent_dropout=0.2)),
        layers.BatchNormalization(),

        # Dense layer
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),

        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the trained model
model = create_model()
model.load_weights(model_path)

# Function to extract audio features
def extract_features(audio_path, max_length=500):
    try:
        # Load audio file
        audio, _ = librosa.load(audio_path, sr=16000)
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
        # Pad or truncate the feature array
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]
        return mfccs
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

@app.get("/")
def root():
    return {"message": "Audio Classification API is running"}

@app.post("/upload_and_predict/")
async def upload_and_predict(file: UploadFile = File(...)):
    try:
        # Validate file extension
        if not file.filename.endswith((".wav", ".mp3")):
            raise HTTPException(status_code=400, detail="Unsupported file format. Only .wav or .mp3 are allowed.")

        # Save the uploaded file to a temporary location
        file_location = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)  # Create temp directory if not exists
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Extract features from the saved file
        features = extract_features(file_location)
        # Reshape features to fit the model's input
        features = features.reshape(1, 40, 500, 1)
        # Make prediction
        prediction = model.predict(features)
        label = "real" if prediction[0][0] >= 0.2 else "fake"

        # Remove the temporary file after processing
        os.remove(file_location)

        return {"prediction": label, "probability": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# API endpoint for evaluation
@app.post("/evaluate/")
def evaluate(files: list[str], labels: list[int]):
    try:
        # Ensure that the number of files matches the number of labels
        if len(files) != len(labels):
            raise HTTPException(status_code=400, detail="Files and labels must have the same length.")

        predictions = []
        true_labels = []

        # Iterate through files and labels
        for file, label in zip(files, labels):
            features = extract_features(file)
            features = features.reshape(1, 40, 500, 1)
            pred = model.predict(features)
            predictions.append(1 if pred[0][0] > 0.5 else 0)
            true_labels.append(label)

        # Calculate evaluation metrics
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        accuracy = accuracy_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)

        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["real", "fake"], yticklabels=["real", "fake"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")  # Save confusion matrix as an image

        return {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_image": "confusion_matrix.png"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Evaluation error: {str(e)}")