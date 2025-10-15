# Simple container for Streamlit + FastAPI
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501 8000

# Default: run Streamlit app; override with --entrypoint for FastAPI
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
