---
title: BUZZ-UP
emoji: 😴
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# BUZZ-UP — детекция сонливости для водителей

Система мониторинга сонливости в реальном времени: **MediaPipe Face Landmarker + EAR**, WebRTC-камера и карта точек отдыха на маршруте.

## Возможности

- Детекция сонливости по глазам через веб-камеру
- Метрика **EAR** (Eye Aspect Ratio) в реальном времени
- Карта маршрута и ближайшие точки отдыха (АЗС, кафе, отели)
- GPS и ручной ввод координат

## Как пользоваться

1. Нажми **START** у видеоплеера и разреши доступ к камере
2. Смотри в камеру — следи за **EAR** и статусом
3. Выбери маршрут → **Найти ближайшую остановку**

## Локальный запуск

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Стек

Python · Streamlit · MediaPipe · OpenCV · streamlit-webrtc · pydeck
