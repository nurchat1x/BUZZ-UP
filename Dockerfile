FROM python:3.11-slim

WORKDIR /app

# OpenCV + MediaPipe native libs (libGLESv2.so.2, libGL.so.1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p models && curl -L \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
    -o models/face_landmarker.task

COPY app.py drowsiness_detector.py bus_stops.json ./

ENV PORT=8501
EXPOSE 8501

HEALTHCHECK CMD curl --fail "http://localhost:${PORT}/_stcore/health" || exit 1

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
