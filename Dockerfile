FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 81

CMD ["streamlit", "run", "app.py", "--server.port=81"]
