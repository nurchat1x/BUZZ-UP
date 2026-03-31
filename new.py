"""
Streamlit приложение для детекции сонливости в реальном времени.
Использует веб-камеру для отслеживания глаз и показывает статус: Спит или Не Спит.
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import os
import json
import math
from PIL import Image, ImageDraw, ImageFont
import time
from collections import deque
import statistics
import streamlit.components.v1 as components
import importlib
import importlib.util

# Динамический импорт folium: если пакет не установлен, не будет падения и иногда
# это помогает подавить предупреждения статического анализатора ("could not be resolved").
# Установите folium в окружение, чтобы убрать предупреждение и включить карту:
#   pip install folium
if importlib.util.find_spec("folium") is not None:
    folium = importlib.import_module("folium")
    FOLIUM_AVAILABLE = True
else:
    folium = None
    FOLIUM_AVAILABLE = False

# Настройка страницы
st.set_page_config(
    page_title="Детекция Сонливости",
    page_icon="😴",
    layout="wide"
)

@st.cache_resource
def load_model():
    """
    Загружает обученную модель и связанные файлы с кэшированием
    """
    try:
        classifier = joblib.load('eye_classifier.pkl')
        scaler = joblib.load('eye_scaler.pkl')
        class_mapping = joblib.load('class_mapping.pkl')
        return classifier, scaler, class_mapping
    except FileNotFoundError as e:
        st.error(f"❌ Файл модели не найден: {e}")
        st.error("Сначала запустите обучение: python train.py")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке модели: {e}")
        return None, None, None

@st.cache_resource
def load_bus_stops():
    """
    Загружает данные остановок из JSON файла и дополняет их новыми координатами от пользователя
    """
    try:
        with open('bus_stops.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error("❌ Файл bus_stops.json не найден")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке остановок: {e}")
        return None

    # Новые остановки (только name, lat, lng и оценка удобства -> convenience)
    additions = {
        "almaty-astana": [
            {"name": "fluricaffe", "lat": 43.2235534, "lng": 76.8291227, "type": "fluricaffe", "safety": 4, "convenience": 4},
            {"name": "Shafiraytest utanto", "lat": 43.2256206, "lng": 76.8278883, "type": "Shafiraytest utanto", "safety": 4, "convenience": 4},
            {"name": "Mosque", "lat": 43.2293332, "lng": 76.8290196, "type": "Mosque", "safety": 5, "convenience": 5},
            {"name": "Alastoretsur anti", "lat": 43.2191337, "lng": 76.8923882, "type": "Alastoretsur anti", "safety": 4, "convenience": 5},
            {"name": "Barissa Caffe", "lat": 43.2139535, "lng": 76.8921331, "type": "Barissa Caffe", "safety": 3, "convenience": 3},
        ],
        "almaty-aktau": [
            {"name": "parmacy", "lat": 43.2154487, "lng": 76.9631482, "type": "parmacy", "safety": 3, "convenience": 4},
            {"name": "Ramicaffei", "lat": 43.2122072, "lng": 76.9664393, "type": "Ramicaffei", "safety": 4, "convenience": 4},
            # исправлена опечатка: убран лишний ', 21' в исходных данных
            {"name": "Ouen Coffe", "lat": 43.2097270, "lng": 76.9545837, "type": "Ouen Coffe", "safety": 3, "convenience": 4},
        ],
        # Если нужно добавить другие маршруты — можно расширить additions
    }

    # Убедимся, что структура имеет ключ 'routes'
    if not isinstance(data, dict):
        data = {}
    if 'routes' not in data or not isinstance(data['routes'], dict):
        data.setdefault('routes', {})

    # Вставляем/дополняем остановки — добавляем только в уже существующие маршруты
    for route_id, stops in additions.items():
        if route_id in data['routes']:
            route_entry = data['routes'][route_id]
            route_entry.setdefault('stops', [])
            # добавляем только поля name/lat/lng/convenience
            for s in stops:
                # защитный фильтр: не дублировать абсолютно одинаковые записи
                exists = any(
                    (abs(existing.get('lat', 0) - s['lat']) < 1e-6 and
                     abs(existing.get('lng', 0) - s['lng']) < 1e-6)
                    for existing in route_entry['stops']
                )
                if not exists:
                    route_entry['stops'].append(s)
        else:
            # если маршрута нет — пропускаем (не создаём новый)
            # можно логировать или собирать пропущенные записи при необходимости
            pass

    return data

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Вычисляет расстояние между двумя точками по формуле Haversine
    Возвращает расстояние в километрах
    """
    # Радиус Земли в километрах
    R = 6371.0
    
    # Конвертируем градусы в радианы
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Разности координат
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Формула Haversine
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def find_nearest_stop(user_lat, user_lon, route_name, bus_stops_data):
    """
    Находит ближайшую остановку на указанном маршруте
    """
    if not bus_stops_data or route_name not in bus_stops_data['routes']:
        return None
    
    route_stops = bus_stops_data['routes'][route_name]['stops']
    nearest_stop = None
    min_distance = float('inf')
    
    for stop in route_stops:
        distance = calculate_distance(user_lat, user_lon, stop['lat'], stop['lng'])
        if distance < min_distance:
            min_distance = distance
            nearest_stop = stop.copy()
            nearest_stop['distance_km'] = round(distance, 2)
    
    return nearest_stop

def get_user_location():
    """
    Получает геолокацию пользователя через JavaScript.
    При успешном получении редиректит страницу, добавляя ?lat=...&lon=...
    Это позволяет Streamlit прочитать параметры через st.experimental_get_query_params()
    """
    location_html = """
    <script>
    (function(){
        function setQuery(lat, lon) {
            try {
                const url = new URL(window.location.href);
                url.searchParams.set('lat', lat);
                url.searchParams.set('lon', lon);
                // avoid infinite loop: add a marker
                url.searchParams.set('coords_set', '1');
                window.location.href = url.toString();
            } catch (e) {
                // fallback: simple href replace
                window.location.href = window.location.pathname + '?lat=' + lat + '&lon=' + lon + '&coords_set=1';
            }
        }
        if (!window.location.search.includes('coords_set')) {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(position) {
                    setQuery(position.coords.latitude, position.coords.longitude);
                }, function(error) {
                    alert('Ошибка геолокации: ' + error.message);
                }, {enableHighAccuracy: true, timeout: 10000});
            } else {
                alert('Геолокация не поддерживается этим браузером');
            }
        } else {
            // already have coords in URL
            console.log('coords_set present');
        }
    })();
    </script>
    """
    return location_html

def detect_eyes(gray_img, eye_cascade):
    """
    Детекция глаз с оптимизированными параметрами
    """
    height, width = gray_img.shape
    
    if width > 1920:
        scale_factor = 1.05
        min_neighbors = 3
        min_size = (40, 40)
    elif width > 1280:
        scale_factor = 1.1
        min_neighbors = 4
        min_size = (35, 35)
    else:
        scale_factor = 1.1
        min_neighbors = 5
        min_size = (30, 30)
    
    eyes = eye_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=scale_factor, 
        minNeighbors=min_neighbors, 
        minSize=min_size,
        maxSize=(width//4, height//4)
    )
    
    return eyes

def classify_eye_state_fast(eye_img, classifier, scaler):
    """
    Быстрая классификация состояния глаза
    """
    if classifier is None or scaler is None:
        return None
    
    # Изменяем размер до 32x32 (как при обучении)
    eye_resized = cv2.resize(eye_img, (32, 32))
    eye_flattened = eye_resized.flatten().reshape(1, -1)
    
    # Нормализация и предсказание
    eye_scaled = scaler.transform(eye_flattened)
    prediction = classifier.predict(eye_scaled)[0]
    probability = classifier.predict_proba(eye_scaled)[0]
    
    return prediction, probability

def get_drowsiness_status(eyes, classifier, scaler, class_mapping):
    """
    Определяет общий статус сонливости на основе обнаруженных глаз
    """
    if not eyes or classifier is None:
        return "Неизвестно", 0.0
    
    drowsy_count = 0
    total_eyes = len(eyes)
    total_confidence = 0
    
    for (x, y, w, h) in eyes:
        # Здесь нужно будет получить ROI глаза для классификации
        # Пока используем упрощенную логику
        pass
    
    # Упрощенная логика для демонстрации
    # В реальности здесь должна быть классификация каждого глаза
    if total_eyes > 0:
        # Если найдены глаза, считаем что человек не спит
        return "Не Спит", 0.8
    else:
        return "Спит", 0.6

def main():
    """
    Основная функция Streamlit приложения
    """
    st.title("😴 Детекция Сонливости в Реальном Времени")
    st.markdown("---")
    
    # Загружаем модель и данные остановок
    classifier, scaler, class_mapping = load_model()
    bus_stops_data = load_bus_stops()
    
    # Инициализация флагов в session_state (если ещё нет)
    if 'alert_active' not in st.session_state:
        st.session_state['alert_active'] = False
    # добавляем таймер и флаг локального оповещения
    if 'drowsy_start_time' not in st.session_state:
        st.session_state['drowsy_start_time'] = None
    if 'show_local_alert' not in st.session_state:
        st.session_state['show_local_alert'] = False
    # таймаут (unix time) до которого локальный баннер/звук не будут вновь включены
    if 'local_silence_until' not in st.session_state:
        st.session_state['local_silence_until'] = 0.0

    # Callback для кнопки "Остановить локальный сигнал (камерное)"
    def stop_local_callback():
        # Скрываем только баннер и временно блокируем повторную автоматическую активацию.
        st.session_state['show_local_alert'] = False
        st.session_state['drowsy_start_time'] = time.time()
        st.session_state['local_silence_until'] = time.time() + 10.0

    # Попытка прочитать координаты из query params (если JS редирект сделал перезагрузку)
    params = st.experimental_get_query_params()
    if 'lat' in params and 'lon' in params and 'coords_set' in params:
        try:
            lat_val = float(params.get('lat')[0])
            lon_val = float(params.get('lon')[0])
            # Сохраняем в session_state чтобы использовать автоматически
            st.session_state['auto_lat'] = lat_val
            st.session_state['auto_lon'] = lon_val
            # помечаем чтобы не делать повторно
            st.experimental_set_query_params()  # очищаем params чтобы не повторять
            st.experimental_rerun()
        except Exception:
            pass

    if classifier is None:
        st.stop()
    
    # Информация о модели
    with st.expander("ℹ️ Информация о модели"):
        st.write(f"**Загружена модель для {len(class_mapping)} классов:**")
        for name, class_id in class_mapping.items():
            st.write(f"- {name}: класс {class_id}")
    
    # Основные колонки
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("📹 Видео с веб-камеры")
        
        # Добавлена опция открытия отдельного большого окна камеры
        if 'open_window' not in st.session_state:
            st.session_state.open_window = False
        open_window = st.checkbox("Открыть отдельное большое окно камеры (локально)", value=st.session_state.open_window)
        st.session_state.open_window = open_window
        
        # Кнопка для запуска/остановки камеры
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
        
        if st.button("🎥 Запустить камеру", disabled=st.session_state.camera_running):
            st.session_state.camera_running = True
        
        if st.button("⏹️ Остановить камеру", disabled=not st.session_state.camera_running):
            st.session_state.camera_running = False
        
        # Кнопка для остановки локального сигнала (баннера/звука в камере)
        # используем on_click чтобы обработка была в callback и минимально влияла на основной цикл камеры
        st.button("🛑 Остановить локальный сигнал (камерное)", on_click=stop_local_callback)

        # Показываем видео с камеры
        if st.session_state.camera_running:
            # Создаем placeholder для видео
            video_placeholder = st.empty()
            
            # Инициализируем камеру
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Не удалось открыть веб-камеру")
                st.session_state.camera_running = False
                cap.release()
                return
            
            # Если нужно отдельное окно — создаём его
            cv_window_created = False
            if st.session_state.open_window:
                try:
                    # Переименовано окно на LIVE
                    cv2.namedWindow("LIVE", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("LIVE", 1280, 720)
                    cv_window_created = True
                except Exception:
                    cv_window_created = False
            
            # Загружаем каскадный классификатор
            eye_cascade = cv2.CascadeClassifier('lol.xml')
            if eye_cascade.empty():
                st.error("❌ Не удалось загрузить каскадный классификатор")
                st.session_state.camera_running = False
                cap.release()
                if cv_window_created:
                    try:
                        cv2.destroyWindow("LIVE")
                    except Exception:
                        pass
                return
            
            st.info("🎥 Камера запущена! Смотрите в камеру для детекции глаз.")
            
            # Основной цикл обработки видео
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Зеркально отображаем изображение
                frame = cv2.flip(frame, 1)
                
                # Конвертируем в серый
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Детектируем глаза
                eyes = detect_eyes(gray, eye_cascade)
                
                # Обрабатываем каждый найденный глаз
                drowsiness_detected = False
                total_confidence = 0
                
                for (x, y, w, h) in eyes:
                    eye_roi = gray[y:y+h, x:x+w]
                    
                    result = classify_eye_state_fast(eye_roi, classifier, scaler)
                    if result is not None:
                        eye_state, probability = result
                        # защитимся от индексации
                        try:
                            confidence = float(max(probability))
                        except Exception:
                            confidence = 0.0
                        total_confidence += confidence
                        
                        # Определяем цвет рамки на основе состояния глаза
                        if eye_state == 0:  # Закрытые глаза
                            color = (0, 0, 255)  # Красный
                            drowsiness_detected = True
                        elif eye_state == 1:  # Сонные глаза
                            color = (0, 165, 255)  # Оранжевый
                            drowsiness_detected = True
                        elif eye_state == 2:  # Открытые глаза
                            color = (0, 255, 0)  # Зеленый
                        else:
                            color = (128, 128, 128)  # Серый
                        
                        # Рисуем рамку вокруг глаза
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Определяем общий статус
                if drowsiness_detected or len(eyes) == 0:
                    status = "Спит"
                    status_color = (0, 0, 255)  # Красный
                else:
                    status = "Не Спит"
                    status_color = (0, 255, 0)  # Зеленый
                
                # Обновляем статус в session_state для отображения в боковой панели
                st.session_state.current_status = status
                avg_confidence = total_confidence / max(len(eyes), 1)
                st.session_state.current_confidence = avg_confidence
                st.session_state.eyes_detected = len(eyes)
                
                # Если обнаружена сонливость — включаем оповещение (однократно)
                if status == "Спит":
                    # браузерное оповещение (существует отдельно)
                    if not st.session_state.get('alert_active', False):
                        st.session_state['alert_active'] = True
                    # запускаем/поддерживаем таймер локальной сонливости
                    if st.session_state['drowsy_start_time'] is None:
                        st.session_state['drowsy_start_time'] = time.time()
                    else:
                        elapsed = time.time() - st.session_state['drowsy_start_time']
                        # порог в секундах
                        if elapsed >= 7:
                            # не включаем, если недавно нажали "стоп" (прерывание пользователя)
                            if time.time() < st.session_state.get('local_silence_until', 0):
                                # пропускаем автоматический баннер/звук
                                pass
                            else:
                                # включаем визуальный баннер и локальный звук (если ещё не включены)
                                if not st.session_state.get('show_local_alert', False):
                                    st.session_state['show_local_alert'] = True
                else:
                    # Сброс таймера и локального оповещения когда проснулся
                    if st.session_state.get('alert_active', False):
                        st.session_state['alert_active'] = False
                    st.session_state['drowsy_start_time'] = None
                    # выключаем визуальный баннер и локальный звук
                    if st.session_state.get('show_local_alert', False):
                        st.session_state['show_local_alert'] = False
                
                # Добавляем информацию на кадр (с поддержкой кириллицы через PIL)
                def draw_text_pil_bgr(bgr_image: np.ndarray, text: str, position: tuple, 
                                      font_size: int = 24, text_color=(255, 255, 255)) -> np.ndarray:
                    # Конвертируем BGR -> RGB для PIL
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_image)
                    draw = ImageDraw.Draw(pil_img)
                    # Пытаемся найти шрифт с поддержкой кириллицы (Windows)
                    font_paths = [
                        "C:/Windows/Fonts/arial.ttf",
                        "C:/Windows/Fonts/segoeui.ttf",
                        "C:/Windows/Fonts/tahoma.ttf",
                    ]
                    font = None
                    for p in font_paths:
                        try:
                            font = ImageFont.truetype(p, font_size)
                            break
                        except Exception:
                            continue
                    if font is None:
                        # Фолбэк на стандартный шрифт (может не отрисовать кириллицу, но не упадём)
                        font = ImageFont.load_default()
                    draw.text(position, text, font=font, fill=tuple(int(c) for c in text_color))
                    # Обратно RGB -> BGR
                    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    return result

                # Текст слева сверху
                frame = draw_text_pil_bgr(frame, f"Глаза: {len(eyes)}", (10, 10), 24, (255, 255, 255))
                frame = draw_text_pil_bgr(frame, f"Статус: {status}", (10, 40), 24, status_color)

                # Если включено локальное оповещение — рисуем крупный заметный баннер по центру
                if st.session_state.get('show_local_alert', False):
                    # мигание: каждые 0.5s переключаем видимость
                    visible = int(time.time() * 2) % 2 == 0
                    if visible:
                        h, w = frame.shape[:2]
                        # полупрозрачный прямоугольник в центре
                        bx_w, bx_h = int(w * 0.7), int(h * 0.25)
                        bx_x = (w - bx_w) // 2
                        bx_y = (h - bx_h) // 2
                        overlay_local = frame.copy()
                        cv2.rectangle(overlay_local, (bx_x, bx_y), (bx_x + bx_w, bx_y + bx_h), (0, 0, 255), -1)
                        frame = cv2.addWeighted(overlay_local, 0.45, frame, 0.55, 0)
                        # большой текст 'Встань!' по центру
                        text = "ВСТАНЬ!"
                        # подбираем размер шрифта и позицию
                        frame = draw_text_pil_bgr(frame, text, (bx_x + 30, bx_y + int(bx_h/4)), font_size=72, text_color=(255,255,255))
                
                # Бейдж статуса в правом верхнем углу
                h, w = frame.shape[:2]
                badge_text = f"{status}"
                # Рисуем полупрозрачный фон под бейджем
                overlay = frame.copy()
                badge_w, badge_h = 180, 40
                x1, y1 = w - badge_w - 10, 10
                x2, y2 = w - 10, 10 + badge_h
                bg_color = (0, 0, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
                alpha = 0.35
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                # Текст поверх бейджа (через PIL, чтобы не было "????")
                frame = draw_text_pil_bgr(frame, badge_text, (x1 + 10, y1 + 8), 24, status_color)
                
                # Конвертируем BGR в RGB для Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Отображаем кадр
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Если пользователь выбрал отдельное окно — показываем его локально
                if cv_window_created:
                    try:
                        # В локальном окне LIVE показываем только видео (без звуков/оверлеев)
                        # теперь кадр содержит локальное визуальное оповещение, если оно активировано
                        cv2.imshow("LIVE", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            # остановка через клавишу q
                            st.session_state.camera_running = False
                            break
                    except Exception:
                        # игнорируем ошибки отображения окна
                        pass
                
                # Небольшая задержка для плавности
                time.sleep(0.05)
            
            # Освобождаем камеру и окно
            cap.release()
            if cv_window_created:
                try:
                    cv2.destroyWindow("LIVE")
                except Exception:
                    pass
        else:
            st.info("👆 Нажмите 'Запустить камеру' для начала детекции")
    
    with col2:
        st.subheader("📊 Статус детекции")
        
        # Отображаем текущий статус
        if 'current_status' in st.session_state:
            status = st.session_state.current_status
            
            if status == "Спит":
                st.error(f"😴 **{status}**")
                st.markdown("⚠️ **Внимание! Обнаружена сонливость!**")
            else:
                st.success(f"👁️ **{status}**")
                st.markdown("✅ **Человек бодрствует**")
            
            # Дополнительная информация
            if 'current_confidence' in st.session_state:
                confidence = st.session_state.current_confidence
                st.metric("Уверенность", f"{confidence:.2f}")
            
            if 'eyes_detected' in st.session_state:
                eyes_count = st.session_state.eyes_detected
                st.metric("Глаза обнаружены", eyes_count)
        else:
            st.info("🎥 Запустите камеру для получения статуса")
        
        # Информация о ближайшей остановке
        st.markdown("---")
        st.markdown("### 🚌 Ближайшая остановка для отдыха")
        
        if 'nearest_stop' in st.session_state:
            stop = st.session_state.nearest_stop
            st.success(f"📍 **{stop.get('name', 'Неизвестно')}**")
            st.metric("Расстояние", f"{stop.get('distance_km', '—')} км")
            services = stop.get('services') or []
            st.markdown(f"**Услуги:** {', '.join(services) if services else 'Не указано'}")
            
            # Рекомендация по отдыху
            if 'current_status' in st.session_state and st.session_state.current_status == "Спит":
                if stop['distance_km'] <= 20:
                    st.warning("⚠️ **Рекомендуется немедленный отдых!** Близкая остановка найдена.")
                else:
                    st.error("🚨 **Критично!** Нужен отдых, но ближайшая остановка далеко.")
            else:
                if stop['distance_km'] <= 30:
                    st.info("💡 **Близкая остановка** - можно планировать отдых")
                else:
                    st.info("ℹ️ **Дальняя остановка** - продолжайте движение")
        else:
            st.info("🔍 Найдите ближайшую остановку в панели справа")
        
        # Инструкции
        st.markdown("---")
        st.markdown("### 📋 Инструкции:")
        st.markdown("""
        1. Нажмите **"Запустить камеру"**
        2. Смотрите прямо в камеру
        3. Следите за статусом справа
        4. **"Спит"** - глаза закрыты или сонные
        5. **"Не Спит"** - глаза открыты и активны
        """)
        
        # Статистика
        st.markdown("---")
        st.markdown("### 📈 Статистика модели:")
        st.write(f"**Классы:** {len(class_mapping)}")
        for name, class_id in class_mapping.items():
            st.write(f"- {name} (ID: {class_id})")
    
    with col3:
        st.subheader("🚌 Ближайшая остановка")
        
        if bus_stops_data:
            # Выбор маршрута
            route_options = {}
            for route_id, route_data in bus_stops_data['routes'].items():
                route_options[route_data['name']] = route_id
            
            selected_route_name = st.selectbox(
                "Выберите маршрут:",
                options=list(route_options.keys()),
                index=0
            )
            
            selected_route_id = route_options[selected_route_name]
            
            # Сначала: кнопка получения геолокации и ручной ввод координат
            if st.button("📍 Получить мое местоположение"):
                st.markdown(get_user_location(), unsafe_allow_html=True)
                st.info("Разрешите доступ к геолокации в браузере — после разрешения страница обновится")
            
            st.markdown("**Или введите координаты вручную:**")
            col_lat, col_lon = st.columns(2)
            with col_lat:
                manual_lat = st.number_input(
                    "Широта (lat):",
                    value=st.session_state.get('auto_lat', 43.2220),
                    min_value=-90.0,
                    max_value=90.0,
                    step=0.001,
                    format="%.6f",
                    key="manual_lat_input"
                )
            with col_lon:
                manual_lon = st.number_input(
                    "Долгота (lon):",
                    value=st.session_state.get('auto_lon', 76.8512),
                    min_value=-180.0,
                    max_value=180.0,
                    step=0.001,
                    format="%.6f",
                    key="manual_lon_input"
                )
            # Выбираем координаты для карты: приоритет — авто (если есть), иначе ручные
            map_user_lat = st.session_state.get('auto_lat', None) or manual_lat
            map_user_lon = st.session_state.get('auto_lon', None) or manual_lon

            # Обязательно задаём user_lat/user_lon для поиска ближайшей остановки
            user_lat = map_user_lat
            user_lon = map_user_lon

            # Построение карты для выбранного маршрута
            def build_route_map(route_id, user_lat=None, user_lon=None):
                # центрируем карту на пользователе или на первой остановке маршрута
                stops = bus_stops_data['routes'].get(route_id, {}).get('stops', [])
                if user_lat is not None and user_lon is not None:
                    center = (user_lat, user_lon)
                elif stops:
                    center = (stops[0].get('lat', 43.2220), stops[0].get('lng', 76.8512))
                else:
                    center = (43.2220, 76.8512)
                m = folium.Map(location=center, zoom_start=13)
                # маркеры остановок
                for s in stops:
                    lat = s.get('lat')
                    lng = s.get('lng')
                    if lat is None or lng is None:
                        continue
                    popup_items = []
                    if s.get('name'):
                        popup_items.append(f"<b>{s['name']}</b>")
                    if s.get('convenience') is not None:
                        popup_items.append(f"Удобство: {s['convenience']}")
                    if s.get('type'):
                        popup_items.append(f"Тип: {s['type']}")
                    if s.get('safety') is not None:
                        popup_items.append(f"Безопасность: {s['safety']}")
                    popup = "<br>".join(popup_items) if popup_items else None
                    folium.Marker(location=(lat, lng), popup=popup, icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
                # маркер пользователя
                if user_lat is not None and user_lon is not None:
                    folium.Marker(location=(user_lat, user_lon), popup="Вы здесь",
                                  icon=folium.Icon(color='red', icon='user')).add_to(m)
                return m

            # Отобразим карту (используем map_user_lat/map_user_lon)
            if FOLIUM_AVAILABLE:
                route_map = build_route_map(selected_route_id, user_lat=map_user_lat, user_lon=map_user_lon)
                # Увеличенный размер карты
                map_html = route_map._repr_html_()
                components.html(map_html, height=600, scrolling=False)
            else:
                st.warning("Для отображения карты установите пакет folium: pip install folium")
                stops = bus_stops_data['routes'].get(selected_route_id, {}).get('stops', [])
                if stops:
                    rows = []
                    for s in stops:
                        rows.append({
                            "name": s.get('name', '—'),
                            "lat": s.get('lat', '—'),
                            "lng": s.get('lng', '—'),
                            "type": s.get('type', 'Не указано'),
                            "safety": s.get('safety', 'Не указано'),
                            "convenience": s.get('convenience', 'Не указано')
                        })
                    st.table(rows)
                else:
                    st.info("Остановки для этого маршрута отсутствуют.")
            
            # Если есть автоматически полученные координаты из query — используем их и обновляем user_lat/user_lon
            auto_lat = st.session_state.get('auto_lat', None)
            auto_lon = st.session_state.get('auto_lon', None)
            if auto_lat is not None and auto_lon is not None:
                st.success(f"Авто-координаты получены: {auto_lat:.5f}, {auto_lon:.5f}")
                user_lat = auto_lat
                user_lon = auto_lon
                # удаляем чтобы не повторять
                st.session_state.pop('auto_lat', None)
                st.session_state.pop('auto_lon', None)
                # автоматически ищем остановку
                nearest_stop = find_nearest_stop(user_lat, user_lon, selected_route_id, bus_stops_data)
                if nearest_stop:
                    st.session_state.nearest_stop = nearest_stop

            # Поиск ближайшей остановки по кнопке
            if st.button("🔍 Найти ближайшую остановку"):
                nearest_stop = find_nearest_stop(user_lat, user_lon, selected_route_id, bus_stops_data)
                
                if nearest_stop:
                    st.success("✅ Найдена остановка!")
                    
                    st.markdown(f"**📍 {nearest_stop.get('name', 'Неизвестно')}**")
                    st.markdown(f"**Расстояние:** {nearest_stop.get('distance_km', '—')} км")
                    # Описание/Услуги — безопасно
                    st.markdown(f"**Описание:** {nearest_stop.get('description', 'Не указано')}")
                    services = nearest_stop.get('services') or []
                    st.markdown(f"**Услуги:** {', '.join(services) if services else 'Не указано'}")
                    # Тип / Безопасность / Удобство
                    stop_type = nearest_stop.get('type', nearest_stop.get('stop_type', 'Не указано'))
                    stop_safety = nearest_stop.get('safety', 'Не указано')
                    stop_convenience = nearest_stop.get('convenience', 'Не указано')
                    st.markdown(f"**Тип остановки:** {stop_type}")
                    st.markdown(f"**Безопасность (1-5):** {stop_safety}")
                    st.markdown(f"**Удобство (1-5):** {stop_convenience}")
                    
                    # Цветовая индикация расстояния
                    if nearest_stop.get('distance_km', 9999) <= 10:
                        st.success("🟢 Близко! Идеально для отдыха")
                    elif nearest_stop.get('distance_km', 9999) <= 50:
                        st.warning("🟡 Умеренное расстояние")
                    else:
                        st.info("🔵 Дальняя остановка")
                    
                    # Сохраняем в session_state для отображения в основной панели
                    st.session_state.nearest_stop = nearest_stop
                else:
                    st.error("❌ Не удалось найти остановку")
        else:
            st.error("❌ Данные остановок не загружены")

        # Оповещение в браузере и кнопки управления звуком/оверлеем
        if st.session_state.get('alert_active', False):
            st.error("⚠️ Активировано оповещение: обнаружена сонливость!")
            # Встраиваем JS: произнести предупреждение и показать мигающий оверлей
            # только визуальный оверлей (без звука)
            alert_html = """
            <script>
            (function(){
                try {
                    if (!document.getElementById('drowsy_overlay')) {
                        var overlay = document.createElement('div');
                        overlay.id = 'drowsy_overlay';
                        overlay.style.position='fixed';
                        overlay.style.top='0';
                        overlay.style.left='0';
                        overlay.style.width='100%';
                        overlay.style.height='100%';
                        overlay.style.background='rgba(255,0,0,0.25)';
                        overlay.style.zIndex='2147483647';
                        overlay.style.pointerEvents='none';
                        document.body.appendChild(overlay);
                    }
                } catch(e) { console.log(e); }
            })();
            </script>
            """
             # передаём ненулевой height чтобы скрипт гарантированно исполнился в iframe
            components.html(alert_html, height=150)
            if st.button("🛑 Остановить сигнал"):
                # отключаем оповещение и просим JS убрать оверлей/остановить речь
                st.session_state['alert_active'] = False
                stop_html = """
                <script>
                (function(){
                    try {
                        var el = document.getElementById('drowsy_overlay');
                        if (el) el.parentNode.removeChild(el);
                    } catch(e) { console.log(e); }
                })();
                </script>
                """
                components.html(stop_html, height=150)
        else:
            st.info("🔕 Оповещение выключено")

if __name__ == "__main__":
    main()
