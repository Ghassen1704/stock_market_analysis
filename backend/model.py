import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Step 1: Get stock data
def get_stock_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2024-01-01'
data = get_stock_data(symbol, start_date, end_date)

# Step 2: Preprocess data
def preprocess_data(data, time_step=60):
    close_prices = data['Close'].values.reshape(-1, 1)
    print("Close prices : ",close_prices)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    def create_sequences(dataset):
        X, y = [], []
        for i in range(time_step, len(dataset)):
            X.append(dataset[i - time_step:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

X, y, scaler = preprocess_data(data)

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Build the model (added more regularization)
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Added Dropout for regularization
    model.add(LSTM(30, return_sequences=False))
    model.add(Dropout(0.2))  # Added Dropout for regularization
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_model((X_train.shape[1], 1))

# Step 5: Train the model with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,  # Increased the number of epochs
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Step 6: Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# Step 7: Save model and scaler
model.save('stock_predict_model.h5')
joblib.dump(scaler, 'scaler.save')

print("Model and scaler saved successfully.")

# Step 8: Predict next day price
def predict_next_day(model, data, scaler, time_step=60):
    last_sequence = data['Close'].values[-time_step:].reshape(-1, 1)
    last_scaled = scaler.transform(last_sequence)
    X_input = last_scaled.reshape(1, time_step, 1)
    prediction_scaled = model.predict(X_input)
    return scaler.inverse_transform(prediction_scaled)[0][0]

next_day_price = predict_next_day(model, data, scaler)
print(f"Predicted next day's price: ${next_day_price:.2f}")

# Optional: Reload model + scaler later like this
def load_model_and_scaler(model_path='stock_predict_model.h5', scaler_path='scaler.save'):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Example usage:
# model, scaler = load_model_and_scaler()
# prediction = predict_next_day(model, data, scaler)
