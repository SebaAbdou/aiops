FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY locustfile.py .

EXPOSE 8089

# No -t (run forever), no -u/-r (MasterLoadShape controls user count)
CMD ["locust", "-f", "locustfile.py", "--headless", "--host=http://echo-server:5000"]
