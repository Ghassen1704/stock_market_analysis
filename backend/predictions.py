import tensorflow as tf
import numpy as np
import joblib  # To load the scaler

# Load the scaler
scaler = joblib.load('scaler.save')

def predict_stock_price(data, previous_prices=None):
    """
    Predict stock price based on a single price or sequence of past prices.
    
    :param data: The current stock price (can be a single float).
    :param previous_prices: A list or array of previous stock prices (if available).
    :return: Predicted stock price.
    """

    # If no previous prices are provided, we will simulate a sequence
    if previous_prices is None:
        previous_prices = [data] * 10  # Use the current price for a dummy sequence

    # Ensure data is a NumPy array and reshape it for scaling
    data = np.array(previous_prices).reshape(-1, 1)  # Make it 2D for scaling
    print(f"Sequence for prediction: {data.flatten()}")

    # Scale the sequence using the same scaler that was used during training
    print("Scaling is gonna start")
    try:
        scaled_data = scaler.transform(data)
        print(f"Scaled data shape: {scaled_data.shape}")
        print(f"Scaled data: {scaled_data}")
    except Exception as e:
        print(f"Error during scaling: {e}")
        return None

    print("Scaling ended")

    # Load the trained model
    model = tf.keras.models.load_model('stock_predict_model.h5')

    # Ensure the input shape is correct (1 sample, sequence_length, 1 feature)
    sequence_length = len(scaled_data)  # This should match the sequence length used during training
    scaled_data = scaled_data.reshape(1, sequence_length, 1)  # Reshape to (1, sequence_length, 1)
    print(f"Reshaped scaled data for prediction: {scaled_data.shape}")

    # Make prediction using the model
    try:
        prediction = model.predict(scaled_data)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction: {prediction}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

    # Inverse transform the prediction to get the real predicted value
    try:
        real_predicted_price = scaler.inverse_transform(prediction.reshape(-1, 1))
        print(f"Real predicted price shape after inverse transform: {real_predicted_price.shape}")
        print(f"Real predicted price: {real_predicted_price}")
    except Exception as e:
        print(f"Error during inverse transformation: {e}")
        return None

    # Return the predicted price in the original scale
    if real_predicted_price.size > 0:
        return real_predicted_price[0][0]
    else:
        print("Error: Invalid prediction shape or values.")
        return None
