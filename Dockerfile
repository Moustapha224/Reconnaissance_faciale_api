FROM python:3.9-slim
RUN apt-get update && apt-get install -y
build-essential
cmake
libgl1-mesa-glx
libglib2.0-0
libsm6
libxrender1
libxext6
&& rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
