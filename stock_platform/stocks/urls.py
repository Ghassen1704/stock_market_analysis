from django.urls import path
from stock_data import views  # Import your views here

urlpatterns = [
    path('api/portfolio/', views.PortfolioView.as_view(), name='portfolio'),
    path('api/stock/<symbol>/', views.StockDataView.as_view(), name='stock_data'),
    path('api/predict/', views.PredictionView.as_view(), name='predict_stock'),
]
