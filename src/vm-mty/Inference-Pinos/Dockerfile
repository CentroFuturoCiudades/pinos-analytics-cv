# service2/Dockerfile
FROM ultralytics/yolov5:latest

# Set the working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install requirements
RUN pip install --upgrade pip && \
pip install -r requirements.txt

# Copy the rest of the application files to /app
COPY . .

CMD ["python", "-u", "main.py"]