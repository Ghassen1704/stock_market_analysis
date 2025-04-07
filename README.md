# Real-time Stock Market Analytics Platform

A real-time, AI-powered stock market analytics platform showcasing modern backend architecture, data streaming, and predictive modeling.

---

## 🔍 Why This Project?

This project serves as a hands-on showcase of:

- Real-time data processing
- WebSocket communication
- Time Series Forecasting (LSTMs)
- News Sentiment Analysis
- Portfolio Simulation & Risk Analysis

---

## 💡 Tech Stack

| Layer         | Technology Used        |
|---------------|------------------------|
| **Backend**   | FastAPI + Django (Admin) |
| **Streaming** | Redis Streams + WebSockets |
| **AI Models** | LSTMs (via Darts) for stock forecasting |
| **Database**  | PostgreSQL (core data), Redis (stream buffer/cache) |

---

## ⚙️ Features

- ✅ Real-time Stock Price Generation + Broadcasting
- ✅ Dummy News Headlines + Sentiment Score
- ✅ Redis Streaming for Live Updates
- ✅ Predictive AI (coming soon using LSTM via `darts`)
- ✅ FastAPI API endpoints for client apps
- ✅ Django admin for managing stocks/news/users
- 🔄 Portfolio Management (simulate assets, risk metrics)

---

## 📂 Project Structure

```bash
stock_market_analytics/
├── producer.py              # Simulates stock updates with sentiment
├── consumer.py              # (Planned) WebSocket-based client
├── fastapi_app/             # FastAPI backend (routes, services)
├── django_admin/            # Django backend for admin panel
├── models/                  # LSTM, forecasting logic (planned)
├── redis/                   # Redis configs or helpers
├── requirements.txt
├── README.md
└── .gitignore