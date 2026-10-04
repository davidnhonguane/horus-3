FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HORUS_QUIET=1
EXPOSE 8000
CMD ["python", "-m", "horus", "serve", "--host", "0.0.0.0", "--port", "8000"]
