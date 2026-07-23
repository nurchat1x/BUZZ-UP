"""
BUZZ-UP — детекция сонливости в реальном времени.
MediaPipe Face Mesh + EAR, WebRTC-камера, карта точек отдыха.
"""

import json
import math
import os

import numpy as np
import pydeck as pdk
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_js_eval import get_geolocation

st.set_page_config(
    page_title="BUZZ-UP — Детекция Сонливости",
    page_icon="😴",
    layout="wide",
)

CV2_IMPORT_ERROR: str | None = None
try:
    import cv2
except ImportError as exc:
    cv2 = None
    CV2_IMPORT_ERROR = str(exc)

if cv2 is not None:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

    from drowsiness_detector import DrowsinessDetector, DrowsinessResult, probe_detector
else:
    av = None
    RTCConfiguration = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None
    DrowsinessDetector = None
    DrowsinessResult = None
    probe_detector = None


@st.cache_resource
def mediapipe_runtime_ok() -> tuple[bool, str | None]:
    if cv2 is None:
        return False, CV2_IMPORT_ERROR or "OpenCV (cv2) недоступен на сервере"
    return probe_detector()


def get_rtc_configuration() -> dict:
    """STUN/TURN для WebRTC на облаке (Render, Streamlit Cloud).

    Локально хватает STUN. На облаке часто нужен TURN — иначе долгое
    «Connection is taking longer than expected».
    """
    ice_servers: list[dict] = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]

    sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth = os.getenv("TWILIO_AUTH_TOKEN")
    if sid and auth:
        try:
            from twilio.rest import Client

            token = Client(sid, auth).tokens.create()
            ice_servers = token.ice_servers
        except Exception:
            pass

    use_public_turn = os.getenv("BUZZUP_USE_PUBLIC_TURN", "true").lower() in ("1", "true", "yes")
    has_turn = any(
        url.startswith("turn:") or url.startswith("turns:")
        for server in ice_servers
        for url in (server.get("urls") if isinstance(server.get("urls"), list) else [server.get("urls")])
        if url
    )
    if use_public_turn and not has_turn:
        ice_servers.append(
            {
                "urls": [
                    "turn:openrelay.metered.ca:80",
                    "turn:openrelay.metered.ca:443",
                    "turn:openrelay.metered.ca:443?transport=tcp",
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            }
        )
        has_turn = True

    config: dict = {"iceServers": ice_servers}
    if has_turn:
        config["iceTransportPolicy"] = "relay"
    return config


@st.cache_resource
def load_bus_stops():
    try:
        with open("bus_stops.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ Файл bus_stops.json не найден")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке остановок: {e}")
        return None


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def find_nearest_stop(user_lat, user_lon, route_name, bus_stops_data):
    if not bus_stops_data or route_name not in bus_stops_data["routes"]:
        return None
    route_stops = bus_stops_data["routes"][route_name]["stops"]
    nearest_stop = None
    min_distance = float("inf")
    for stop in route_stops:
        distance = calculate_distance(user_lat, user_lon, stop["lat"], stop["lng"])
        if distance < min_distance:
            min_distance = distance
            nearest_stop = stop.copy()
            nearest_stop["distance_km"] = round(distance, 2)
    return nearest_stop


def is_rest_point_stop(stop: dict) -> bool:
    services = [str(s).lower() for s in stop.get("services", [])]
    if "отель" in services:
        return True
    if "кафе" in services and "туалет" in services:
        return True
    if stop.get("rating") and ("кафе" in services or "заправка" in services or "отель" in services):
        return True
    return False


def format_rating(stop: dict) -> str:
    rating = stop.get("rating")
    if rating is None:
        return "—"
    reviews = stop.get("reviews_count")
    source = stop.get("rating_source", "")
    base = f"⭐ {rating}/5"
    if reviews:
        base += f" ({reviews} отзывов"
        if source:
            base += f", {source}"
        base += ")"
    elif source:
        base += f" ({source})"
    return base


def render_stop_details(stop: dict) -> None:
    st.markdown(f"**📍 {stop['name']}**")
    st.metric("Расстояние", f"{stop['distance_km']} км")
    if stop.get("address"):
        st.markdown(f"**Адрес:** {stop['address']}")
    if stop.get("description"):
        st.markdown(f"**Описание:** {stop['description']}")
    st.markdown(f"**Рейтинг:** {format_rating(stop)}")
    services = stop.get("services") or []
    amenities = stop.get("amenities") or []
    st.markdown(f"**Услуги:** {', '.join(services) if services else '—'}")
    if amenities:
        st.markdown(f"**Удобства:** {', '.join(amenities)}")


def _unwrap_streamlit_js_eval_payload(raw):
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
    if not bus_stops_data or route_id not in bus_stops_data.get("routes", {}):
        return None
    stops = bus_stops_data["routes"][route_id]["stops"]
    rest_stops = [s for s in stops if is_rest_point_stop(s)]

    if user_lat is not None and user_lon is not None:
        nearby_rest = []
        for stop in rest_stops:
            dist = calculate_distance(user_lat, user_lon, stop["lat"], stop["lng"])
            if dist <= 30:
                nearby_rest.append({**stop, "distance_km": round(dist, 2)})
        frame_stops = nearby_rest if nearby_rest else rest_stops[:8]
        frame_pts = [(float(user_lat), float(user_lon))]
        frame_pts += [(float(s["lat"]), float(s["lng"])) for s in frame_stops]
        zoom = 13.0 if nearby_rest else 11.0
        view = pdk.ViewState(
            latitude=float(user_lat),
            longitude=float(user_lon),
            zoom=zoom,
            pitch=0,
        )
    else:
        frame_pts = [(float(s["lat"]), float(s["lng"])) for s in rest_stops if "lat" in s and "lng" in s]
        view = _view_state_for_points(frame_pts)

    rest_data = [
        {
            "name": s.get("name", "—"),
            "lat": float(s["lat"]),
            "lng": float(s["lng"]),
            "rating": s.get("rating", "—"),
        }
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
                get_radius=80,
                radius_min_pixels=8,
                radius_max_pixels=16,
                pickable=True,
            )
        )

    if user_lat is not None and user_lon is not None:
        user_data = [{"name": "Вы (GPS)", "lat": float(user_lat), "lng": float(user_lon)}]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=user_data,
                id="user_position",
                get_position="[lng, lat]",
                get_fill_color=[30, 144, 255, 255],
                get_radius=50,
                radius_min_pixels=10,
                radius_max_pixels=14,
                pickable=True,
            )
        )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip={
            "html": "<b>{name}</b><br/>⭐ {rating}<br/>lat {lat}, lng {lng}",
            "style": {"backgroundColor": "#1e1e1e", "color": "white"},
        },
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )


def draw_text_pil_bgr(
    bgr_image: np.ndarray,
    text: str,
    position: tuple,
    font_size: int = 24,
    text_color=(255, 255, 255),
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV недоступен")
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_img)
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=tuple(int(c) for c in text_color))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def get_webrtc_processor(webrtc_ctx):
    """Доступ к процессору кадров (API streamlit-webrtc >= 0.47)."""
    return getattr(webrtc_ctx, "video_processor", None) or getattr(webrtc_ctx, "video_transformer", None)


def get_live_result(webrtc_ctx):
    processor = get_webrtc_processor(webrtc_ctx)
    if processor is not None and getattr(processor, "last_result", None) is not None:
        return processor.last_result
    return None


def make_video_processor(ear_threshold: float, consecutive_frames: int):
    """Фабрика процессора для streamlit-webrtc."""

    class DrowsinessVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.ear_threshold = ear_threshold
            self.consecutive_frames = consecutive_frames
            self.detector = None
            self.last_result = None

        def _get_detector(self) -> DrowsinessDetector:
            if self.detector is None:
                self.detector = DrowsinessDetector(
                    ear_threshold=self.ear_threshold,
                    consecutive_frames=self.consecutive_frames,
                    draw_landmarks=False,
                )
            return self.detector

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            if img is None:
                return frame

            img = cv2.flip(img, 1)
            annotated, result = self._get_detector().process_frame(img)
            self.last_result = result

            if result.status == "Спит":
                status_color = (0, 0, 255)
                badge = "DROWSY"
            elif result.status == "Не Спит":
                status_color = (0, 255, 0)
                badge = "AWAKE"
            else:
                status_color = (128, 128, 128)
                badge = "NO FACE"

            cv2.putText(
                annotated,
                f"EAR: {result.ear:.3f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                badge,
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )

            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (w - 130, 10), (w - 10, 50), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                badge,
                (w - 120, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        def __del__(self):
            if hasattr(self, "detector"):
                self.detector.close()

    return DrowsinessVideoProcessor


def render_photo_demo(ear_threshold: float, consecutive_frames: int) -> None:
    """Демо через загрузку фото, если WebRTC/MediaPipe на сервере недоступны."""
    if cv2 is None or DrowsinessDetector is None:
        st.error("OpenCV недоступен на этом сервере — детекция по фото временно отключена.")
        st.info("Карта и поиск остановок работают. Полная версия: `python -m streamlit run app.py` локально.")
        return

    st.warning(
        "На облачном сервере камера может быть недоступна. "
        "Загрузите фото лица или запустите локально: `streamlit run app.py`"
    )
    uploaded = st.file_uploader("Загрузить фото для анализа", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Не удалось прочитать изображение")
        return

    try:
        detector = DrowsinessDetector(
            ear_threshold=ear_threshold,
            consecutive_frames=consecutive_frames,
            draw_landmarks=False,
        )
        annotated, result = detector.process_frame(frame)
        detector.close()
    except OSError as exc:
        st.error(f"MediaPipe недоступен на сервере: {exc}")
        st.info("Полная версия с камерой работает локально на твоём ПК.")
        return

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Статус: {result.status}, EAR: {result.ear:.3f}")
    st.session_state.demo_result = result


def render_live_status(webrtc_ctx) -> None:
    """Панель статуса — читает результат из video_processor."""
    result = get_live_result(webrtc_ctx)

    if result is None:
        st.info("🎥 Запустите камеру для получения статуса")
        return

    status = result.status
    if status == "Спит":
        st.error(f"😴 **{status}**")
        st.markdown("⚠️ **Внимание! Обнаружена сонливость!**")
    elif status == "Не Спит":
        st.success(f"👁️ **{status}**")
        st.markdown("✅ **Человек бодрствует**")
    else:
        st.warning(f"❓ **{status}**")
        st.markdown("Повернитесь лицом к камере")

    st.metric("Уверенность", f"{result.confidence:.2f}")
    st.metric("EAR (средний)", f"{result.ear:.3f}")
    c1, c2 = st.columns(2)
    c1.metric("EAR левый", f"{result.left_ear:.3f}")
    c2.metric("EAR правый", f"{result.right_ear:.3f}")
    st.metric("Кадров ниже порога", result.closed_frames)


def main():
    st.title("😴 BUZZ-UP — Детекция Сонливости")
    st.caption("MediaPipe Face Mesh + EAR · WebRTC · карта точек отдыха")
    st.markdown("---")

    bus_stops_data = load_bus_stops()

    with st.sidebar:
        st.subheader("⚙️ Настройки детекции")
        ear_threshold = st.slider("Порог EAR (ниже = глаза закрыты)", 0.15, 0.30, 0.21, 0.01)
        consecutive_frames = st.slider("Кадров подряд для тревоги", 10, 60, 30, 5)
        st.caption("~30 кадров ≈ 1 сек при 30 FPS")
        with st.expander("ℹ️ Как это работает"):
            st.markdown(
                """
                **EAR** (Eye Aspect Ratio) — отношение высоты глаза к ширине.
                MediaPipe находит 468 точек лица; когда EAR падает и держится
                несколько кадров подряд — система фиксирует сонливость.
                """
            )

    col1, col2, col3 = st.columns([2, 1, 1])

    detector_ok, detector_error = mediapipe_runtime_ok()
    webrtc_ctx = None

    with col1:
        st.subheader("📹 Видео с веб-камеры")
        if cv2 is None:
            st.error("OpenCV не загрузился на сервере Streamlit Cloud.")
            st.info(
                "Карта и поиск точек отдыха справа работают. "
                "Для камеры и EAR запустите приложение локально."
            )
            if CV2_IMPORT_ERROR:
                with st.expander("Техническая ошибка OpenCV"):
                    st.code(CV2_IMPORT_ERROR)
        elif detector_ok:
            st.caption("Нажмите START — браузер запросит доступ к камере.")
            webrtc_ctx = webrtc_streamer(
                key="buzz-up-webrtc",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(get_rtc_configuration()),
                video_processor_factory=lambda: make_video_processor(ear_threshold, consecutive_frames)(),
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                async_processing=True,
            )
            if webrtc_ctx.state.playing:
                st.info("🎥 Камера активна. Смотрите в камеру — следите за EAR и статусом.")
            else:
                st.info("👆 Нажмите **START** выше, чтобы включить камеру.")
        else:
            st.caption("Демо-режим: загрузка фото (MediaPipe на сервере недоступен).")
            if detector_error:
                with st.expander("Техническая ошибка сервера"):
                    st.code(detector_error)
            render_photo_demo(ear_threshold, consecutive_frames)

    with col2:
        st.subheader("📊 Статус детекции")

        if webrtc_ctx is not None and webrtc_ctx.state.playing:

            @st.fragment(run_every=0.3)
            def live_status_panel():
                render_live_status(webrtc_ctx)

            live_status_panel()
        elif "demo_result" in st.session_state:
            result = st.session_state.demo_result
            if result.status == "Спит":
                st.error(f"😴 **{result.status}**")
            elif result.status == "Не Спит":
                st.success(f"👁️ **{result.status}**")
            else:
                st.warning(f"❓ **{result.status}**")
            st.metric("EAR (средний)", f"{result.ear:.3f}")
        elif webrtc_ctx is not None:
            render_live_status(webrtc_ctx)
        else:
            st.info("Загрузите фото слева или запустите app локально для камеры.")

        st.markdown("---")
        st.markdown("### 🚌 Ближайшая остановка для отдыха")

        if "nearest_stop" in st.session_state:
            stop = st.session_state.nearest_stop
            render_stop_details(stop)

            _live = get_live_result(webrtc_ctx) if webrtc_ctx is not None else st.session_state.get("demo_result")

            if _live is not None and _live.status == "Спит":
                if stop["distance_km"] <= 20:
                    st.warning("⚠️ **Рекомендуется немедленный отдых!** Близкая остановка найдена.")
                else:
                    st.error("🚨 **Критично!** Нужен отдых, но ближайшая остановка далеко.")
            elif stop["distance_km"] <= 30:
                st.info("💡 **Близкая остановка** — можно планировать отдых")
            else:
                st.info("ℹ️ **Дальняя остановка** — продолжайте движение")
        else:
            st.info("🔍 Найдите ближайшую остановку в панели справа")

        st.markdown("---")
        st.markdown("### 📋 Инструкции")
        st.markdown(
            """
            1. Нажмите **START** у видеоплеера
            2. Разрешите доступ к камере в браузере
            3. Смотрите прямо в камеру
            4. **Спит** — EAR ниже порога ~1 сек
            5. **Не Спит** — глаза открыты
            """
        )

    with col3:
        st.subheader("🚌 Ближайшая остановка")

        if bus_stops_data:
            route_options = {route_data["name"]: route_id for route_id, route_data in bus_stops_data["routes"].items()}
            selected_route_name = st.selectbox("Выберите маршрут:", options=list(route_options.keys()), index=0)
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

            if st.button("📍 Получить мое местоположение"):
                st.session_state.pending_geolocation = True
                st.session_state.geo_nonce += 1
                st.info("Разрешите доступ к геолокации в браузере.")

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

            st.markdown("**Или введите координаты вручную:**")
            col_lat, col_lon = st.columns(2)
            with col_lat:
                user_lat = st.number_input("Широта (lat):", value=43.2220, min_value=-90.0, max_value=90.0, step=0.0001, format="%.6f")
            with col_lon:
                user_lon = st.number_input("Долгота (lon):", value=76.8512, min_value=-180.0, max_value=180.0, step=0.0001, format="%.6f")

            if st.button("🔍 Найти ближайшую остановку"):
                st.session_state.map_manual_lat = float(user_lat)
                st.session_state.map_manual_lon = float(user_lon)
                nearest_stop = find_nearest_stop(user_lat, user_lon, selected_route_id, bus_stops_data)
                if nearest_stop:
                    st.success("✅ Найдена остановка!")
                    render_stop_details(nearest_stop)
                    if nearest_stop["distance_km"] <= 10:
                        st.success("🟢 Близко! Идеально для отдыха")
                    elif nearest_stop["distance_km"] <= 50:
                        st.warning("🟡 Умеренное расстояние")
                    else:
                        st.info("🔵 Дальняя остановка")
                    st.session_state.nearest_stop = nearest_stop
                else:
                    st.error("❌ Не удалось найти остановку")

            st.markdown("---")
            st.markdown("### 🗺️ Карта маршрута")
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
