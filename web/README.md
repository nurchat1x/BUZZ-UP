# BUZZ-UP — Live Camera (MediaPipe JS)

Браузерная детекция сонливости в реальном времени.  
Камера и MediaPipe Face Landmarker работают **на устройстве пользователя** — видео на сервер не уходит. Поэтому live открывается по обычной публичной ссылке (Vercel / GitHub Pages), без WebRTC и TURN.

Алгоритм EAR совпадает с `drowsiness_detector.py` (те же индексы глаз, порог 0.21, 30 кадров).

## Локальный запуск

```bash
cd web
npm install
npm run dev
```

Открой URL Vite (обычно `http://localhost:5173`).  
Разреши камеру → ▶ Включить камеру.

Сборка:

```bash
npm run build
npm run preview
```

## Деплой на Vercel (публичная ссылка для команды)

1. Запушь репозиторий на GitHub (включая папку `web/`):
   ```bash
   git add web
   git commit -m "Add BUZZ-UP web live 2.0"
   git push origin main
   ```
2. Зайди на [vercel.com](https://vercel.com) → **Sign in with GitHub**.
3. **Add New… → Project** → выбери репозиторий `BUZZ-UP` (`nurchat1x/BUZZ-UP`).
4. Перед Deploy открой настройки:
   - **Root Directory:** нажми Edit → выбери `web` → Continue
   - Framework: Vite (подставится сам)
   - Build Command: `npm run build`
   - Output Directory: `dist`
5. **Deploy**. Через 1–2 минуты получишь ссылку вида `https://buzz-up-….vercel.app`.
6. Открой её на телефоне/ноутбуке команды → **Мониторинг** → ▶ Включить → разреши камеру.

Альтернатива без сайта Vercel (CLI):

```bash
cd web
npm i -g vercel
vercel login
vercel
```

При вопросе Root — папка `web`. После первого деплоя: `vercel --prod`.

Нужен HTTPS — на Vercel он есть. Без HTTPS камера в браузере не откроется (кроме localhost).

## GitHub Pages (опционально)

```bash
cd web
npm run build
```

Опубликуй содержимое `web/dist/` как GitHub Pages (Settings → Pages → Deploy from branch / folder).  
В `vite.config.ts` уже стоит `base: "./"` для относительных путей.

## Что внутри (2.0)

| Файл | Назначение |
|------|------------|
| `src/ear.ts` | EAR + индексы глаз + счётчик «Спит» |
| `src/detector.ts` | MediaPipe Face Landmarker |
| `src/alert.ts` | Web Audio тревога |
| `src/demo.ts` | Demo mode (😴 / 👁️) |
| `src/logger.ts` | Логи в localStorage + CSV |
| `src/route.ts` | Маршруты / ближайшая остановка |
| `src/fleet.ts` | Fleet dashboard (демо-водители) |
| `src/main.ts` | Камера, UI, вкладки |
| `public/bus_stops.json` | Точки отдыха (копия из корня) |

Внизу страницы: `(2.0)`. Локальный Streamlit (`app.py`) не трогаем.
