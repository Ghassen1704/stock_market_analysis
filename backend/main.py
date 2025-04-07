from fastapi import FastAPI, WebSocket
import redis.asyncio as redis  # Use async Redis client
import json
import asyncio
from collections import deque  # To maintain a fixed-length list of previous prices
from predictions import predict_stock_price  # Import the prediction function
from sentiment import analyze_sentiment  # Import the sentiment analysis function

app = FastAPI()

# Connect to Redis (async version)
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Consumer function for Redis Streams (Async)
async def consume_redis_stream():
    last_id = '0'  # Keep track of the last processed ID

    while True:
        try:
            # Use the last processed ID to read only new messages
            message = await redis_client.xread({'stock_updates': last_id}, block=0, count=1)

            if not message:
                continue  # If no new message, continue waiting

            # Extract the actual data from the message
            stock_data_json = message[0][1][0][1].get('data')  # Extract the 'data' field correctly
            print(f"Raw data extracted: {stock_data_json}")

            # Parse the JSON string inside 'data'
            stock_data = json.loads(stock_data_json)  # Deserialize the JSON string into a Python dict

            # Log the consumed stock data
            print(f"Consumed stock data from Redis: {stock_data}")

            new_id = message[0][1][0][0]  # Get the ID of the current message

            # Update last_id to the ID of the current message
            last_id = new_id

            yield stock_data  # Yield stock data to WebSocket

        except Exception as e:
            print(f"Error consuming Redis stream: {e}")
            await asyncio.sleep(1)  # Sleep to avoid spinning too fast
@app.websocket("/ws/stock/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    await websocket.accept()

    # Log that the WebSocket connection was established
    print(f"WebSocket connection established for symbol: {symbol}")

    # Maintain a queue of previous stock prices (e.g., last 5 prices)
    previous_prices = deque(maxlen=10)  # You can change the maxlen to any length you need

    # Consuming Redis Stream and sending to WebSocket
    try:
        async for stock_data in consume_redis_stream():
            if stock_data['symbol'] == symbol:
                # Log the stock data before prediction
                print(f"Received stock data for {symbol}: {stock_data}")

                # Add the current stock price to the previous prices queue
                previous_prices.append(stock_data['price'])

                # Predict stock price using LSTM model
                predicted_price = None
                if len(previous_prices) == 10:  # Only predict once we have a full sequence
                    predicted_price = predict_stock_price(stock_data['price'], previous_prices)

                # Analyze sentiment of the news (if provided in stock_data)
                sentiment_score = 0
                if 'news' in stock_data:
                    sentiment_score = analyze_sentiment(stock_data['news'])

                # Add predicted price and sentiment to stock data
                stock_data['predicted_price'] = float(predicted_price) if predicted_price is not None else None
                stock_data['sentiment_score'] = float(sentiment_score)

                # Log the updated stock data with prediction and sentiment score
                print(f"Sending updated stock data to WebSocket: {stock_data}")

                # Send updated stock data to WebSocket client
                await websocket.send_text(json.dumps(stock_data))

    except Exception as e:
        print(f"Error sending WebSocket data: {e}")
        await websocket.close()
