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

import pydeck as pdk
from streamlit_js_eval import get_geolocation

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
    Загружает данные остановок из JSON файла
    """
    try:
        with open('bus_stops.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ Файл bus_stops.json не найден")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке остановок: {e}")
        return None

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


def is_rest_point_stop(stop: dict) -> bool:
    """
    «Точка для отдыха» на карте (явные правила):
    - основной критерий: в services есть «отель» (ночлёг);
    - дополнительно: одновременно «кафе» и «туалет» — короткая остановка для отдыха/санитарии.
    Остановки только с convenience без services не попадают в этот слой.
    """
    services = [str(s).lower() for s in stop.get("services", [])]
    if "отель" in services:
        return True
    if "кафе" in services and "туалет" in services:
        return True
    return False


def _unwrap_streamlit_js_eval_payload(raw):
    """Компонент streamlit-js-eval передаёт в Python обёртку {value, dataType: 'json'}."""
    if raw is None:
        return None
    if isinstance(raw, dict) and raw.get("dataType") == "json" and "value" in raw:
        inner = raw["value"]
        if isinstance(inner, str):
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                return inner
        return inner
    return raw


def _view_state_for_points(points: list[tuple[float, float]]) -> pdk.ViewState:
    if not points:
        return pdk.ViewState(latitude=48.0, longitude=67.0, zoom=4)
    lats = [p[0] for p in points]
    lngs = [p[1] for p in points]
    c_lat = sum(lats) / len(lats)
    c_lon = sum(lngs) / len(lngs)
    lat_spread = max(max(lats) - min(lats), 0.02)
    zoom = 7.0 - math.log(lat_spread * 100) / math.log(2)
    zoom = float(max(4.0, min(12.0, zoom)))
    return pdk.ViewState(latitude=c_lat, longitude=c_lon, zoom=zoom, pitch=0)


def build_route_map_deck(
    bus_stops_data: dict,
    route_id: str,
    user_lat: float | None,
    user_lon: float | None,
) -> pdk.Deck | None:
    """Карта маршрута: слой «отдых» + опционально маркер пользователя (GPS или ручная точка)."""
    if not bus_stops_data or route_id not in bus_stops_data.get("routes", {}):
        return None
    stops = bus_stops_data["routes"][route_id]["stops"]
    framing = [(float(s["lat"]), float(s["lng"])) for s in stops if "lat" in s and "lng" in s]
    rest_stops = [s for s in stops if is_rest_point_stop(s)]
    rest_data = [
        {"name": s.get("name", "—"), "lat": float(s["lat"]), "lng": float(s["lng"])}
        for s in rest_stops
    ]

    layers: list[pdk.Layer] = []
    if rest_data:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=rest_data,
                id="rest_points",
                get_position="[lng, lat]",
                get_fill_color=[255, 140, 0, 220],
                get_radius=6000,
                pickable=True,
            )
        )

    user_data = []
    if user_lat is not None and user_lon is not None:
        user_data.append({"name": "Вы (GPS или ручной ввод)", "lat": float(user_lat), "lng": float(user_lon)})
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=user_data,
                id="user_position",
                get_position="[lng, lat]",
                get_fill_color=[30, 144, 255, 255],
                get_radius=9000,
                pickable=True,
            )
        )

    frame_pts = list(framing)
    if user_lat is not None and user_lon is not None:
        frame_pts.append((float(user_lat), float(user_lon)))
    view = _view_state_for_points(frame_pts)

    tooltip = {
        "html": "<b>{name}</b><br/>lat {lat}, lng {lng}",
        "style": {"backgroundColor": "#1e1e1e", "color": "white"},
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )


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

        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
        import av

        class DrowsinessProcessor(VideoProcessorBase):
            def __init__(self):
                self.eye_cascade = cv2.CascadeClassifier('lol.xml')

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                eyes = detect_eyes(gray, self.eye_cascade)

                drowsiness_detected = False
                for (x, y, w, h) in eyes:
                    eye_roi = gray[y:y+h, x:x+w]
                    result = classify_eye_state_fast(eye_roi, classifier, scaler)
                    if result is not None:
                        eye_state, probability = result
                        if eye_state in [0, 1]:
                            color = (0, 0, 255)
                            drowsiness_detected = True
                        else:
                            color = (0, 255, 0)
                        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

                status = "Спит" if (drowsiness_detected or len(eyes) == 0) else "Не Спит"
                color = (0, 0, 255) if status == "Спит" else (0, 255, 0)
                cv2.putText(img, f"Статус: {status}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(img, f"Глаза: {len(eyes)}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="drowsiness",
            video_processor_factory=DrowsinessProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
    
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
            st.success(f"📍 **{stop['name']}**")
            st.metric("Расстояние", f"{stop['distance_km']} км")
            _svc = stop.get("services") or []
            st.markdown(f"**Услуги:** {', '.join(_svc) if _svc else '—'}")
            
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

            if "user_gps_lat" not in st.session_state:
                st.session_state.user_gps_lat = None
                st.session_state.user_gps_lon = None
            if "pending_geolocation" not in st.session_state:
                st.session_state.pending_geolocation = False
            if "geo_nonce" not in st.session_state:
                st.session_state.geo_nonce = 0
            if "map_manual_lat" not in st.session_state:
                st.session_state.map_manual_lat = None
                st.session_state.map_manual_lon = None

            # Геолокация: в браузере нажмите «Разрешить» в системном запросе. Нужен безопасный контекст
            # (https:// или http://localhost при streamlit run); обычный http по IP/домену без TLS может быть заблокирован.

            if st.button("📍 Получить мое местоположение"):
                st.session_state.pending_geolocation = True
                st.session_state.geo_nonce += 1
                st.info("Разрешите доступ к геолокации в браузере (значок замка / запрос у адресной строки).")

            if st.session_state.pending_geolocation:
                st.caption("Запрос координат у браузера…")
                raw_geo = get_geolocation(component_key=f"geo_{st.session_state.geo_nonce}")
                loc = _unwrap_streamlit_js_eval_payload(raw_geo)
                if loc is not None:
                    st.session_state.pending_geolocation = False
                    if isinstance(loc, dict) and loc.get("error"):
                        err = loc["error"]
                        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                        st.warning(f"Геолокация недоступна: {msg}")
                    else:
                        coords = loc.get("coords") or {}
                        lat, lon = coords.get("latitude"), coords.get("longitude")
                        if lat is not None and lon is not None:
                            st.session_state.user_gps_lat = float(lat)
                            st.session_state.user_gps_lon = float(lon)
                            st.success("Координаты получены с GPS.")
                        else:
                            st.warning("Браузер не вернул координаты.")
            
            # Ручной ввод координат (для тестирования)
            st.markdown("**Или введите координаты вручную:**")
            col_lat, col_lon = st.columns(2)
            
            with col_lat:
                user_lat = st.number_input(
                    "Широта (lat):",
                    value=43.2220,  # Алматы
                    min_value=-90.0,
                    max_value=90.0,
                    step=0.001,
                    format="%.3f"
                )
            
            with col_lon:
                user_lon = st.number_input(
                    "Долгота (lon):",
                    value=76.8512,  # Алматы
                    min_value=-180.0,
                    max_value=180.0,
                    step=0.001,
                    format="%.3f"
                )
            
            # Поиск ближайшей остановки
            if st.button("🔍 Найти ближайшую остановку"):
                st.session_state.map_manual_lat = float(user_lat)
                st.session_state.map_manual_lon = float(user_lon)
                nearest_stop = find_nearest_stop(user_lat, user_lon, selected_route_id, bus_stops_data)
                
                if nearest_stop:
                    st.success(f"✅ Найдена остановка!")
                    
                    st.markdown(f"**📍 {nearest_stop['name']}**")
                    st.markdown(f"**Расстояние:** {nearest_stop['distance_km']} км")
                    st.markdown(f"**Описание:** {nearest_stop.get('description', '—')}")
                    
                    # Услуги
                    _ns = nearest_stop.get("services") or []
                    services_text = ", ".join(_ns)
                    st.markdown(f"**Услуги:** {services_text if services_text else '—'}")
                    
                    # Цветовая индикация расстояния
                    if nearest_stop['distance_km'] <= 10:
                        st.success("🟢 Близко! Идеально для отдыха")
                    elif nearest_stop['distance_km'] <= 50:
                        st.warning("🟡 Умеренное расстояние")
                    else:
                        st.info("🔵 Дальняя остановка")
                    
                    # Сохраняем в session_state для отображения в основной панели
                    st.session_state.nearest_stop = nearest_stop
                else:
                    st.error("❌ Не удалось найти остановку")

            st.markdown("---")
            st.markdown("### 🗺️ Карта маршрута")
            st.caption(
                "Легенда: синий — вы (приоритет GPS после успешного запроса; иначе последняя ручная точка "
                "по кнопке «Найти ближайшую остановку»); оранжевый — точки для отдыха на выбранном маршруте."
            )
            u_lat, u_lon = None, None
            if st.session_state.user_gps_lat is not None and st.session_state.user_gps_lon is not None:
                u_lat = st.session_state.user_gps_lat
                u_lon = st.session_state.user_gps_lon
            elif st.session_state.map_manual_lat is not None:
                u_lat = st.session_state.map_manual_lat
                u_lon = st.session_state.map_manual_lon

            deck = build_route_map_deck(bus_stops_data, selected_route_id, u_lat, u_lon)
            if deck is not None:
                st.pydeck_chart(deck, height=520, use_container_width=True)
            else:
                st.info("Нет данных для карты.")
        else:
            st.error("❌ Данные остановок не загружены")

if __name__ == "__main__":
    main()
