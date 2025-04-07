from rest_framework import serializers
from .models import Portfolio, Stock

class PortfolioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Portfolio
        fields = ['user', 'symbol', 'quantity', 'purchase_price']

class StockSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stock
        fields = ['symbol', 'date', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']
