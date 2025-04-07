from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Portfolio, Stock
from .serializers import PortfolioSerializer, StockSerializer

class PortfolioView(APIView):
    def get(self, request):
        # Fetch the user’s portfolio
        portfolio = Portfolio.objects.all()  # You should filter by user
        serializer = PortfolioSerializer(portfolio, many=True)
        return Response(serializer.data)

class StockDataView(APIView):
    def get(self, request, symbol):
        # Fetch stock data by symbol
        stock_data = Stock.objects.filter(symbol=symbol)
        serializer = StockSerializer(stock_data, many=True)
        return Response(serializer.data)
