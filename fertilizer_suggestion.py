import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import joblib

# Check if the model file exists
if os.path.exists('fertilizer_model.h5'):
    print("Model already trained")
else:
    # Load the dataset
    df = pd.read_csv('FertPredictDataset.csv')
    x = df.drop('class', axis=1)
    y = df['class']

    # Integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)

    # Binary encode
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y = onehot_encoder.fit_transform(integer_encoded)

    # Normalize the input data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

    # Neural network parameters
    n_nodes_hl1 = 500
    n_nodes_hl2 = 500
    n_nodes_hl3 = 500
    n_classes = 4
    batch_size = 100

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(9,)),
        tf.keras.layers.Dense(n_nodes_hl1, activation='relu'),
        tf.keras.layers.Dense(n_nodes_hl2, activation='relu'),
        tf.keras.layers.Dense(n_nodes_hl3, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Train the model and save it
    history = model.fit(x_train, y_train, epochs=20, batch_size=batch_size, validation_data=(x_test, y_test))
    model.save('fertilizer_model.h5')

    # Save the scaler and label encoder
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    print("Model trained and saved")
