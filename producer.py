import redis
import random
import time
import json

# Connect to Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

symbols = ['AAPL', 'GOOG', 'AMZN', 'TSLA']

# Example list of stock-related news for sentiment analysis
news_list = [
    "Apple's new product launch is a huge success!",
    "Tesla stock soars after new Model S announcement.",
    "Amazon sees a big increase in Prime Day sales.",
    "Google faces legal challenges over search engine dominance.",
    "TSLA hits record high after strong quarterly earnings."
]

while True:
    symbol = random.choice(symbols)
    price = round(random.uniform(100, 200), 2)
    news = random.choice(news_list)  # Select random news for sentiment analysis
    
    stock_data = {
        "symbol": symbol,
        "price": price,
        "news": news  
    }

    # Push to Redis Stream
    redis_client.xadd('stock_updates', {'data': json.dumps(stock_data)})
    print(f"Sent: {stock_data}")
    time.sleep(2)
