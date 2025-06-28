# service2/Dockerfile
FROM ultralytics/yolov5:latest

# Install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt

# Set the working directory
WORKDIR /app

# Copy the rest of the application files to /app
COPY . /app

CMD ["python", "main.py"]