# AGENTS.md

## Cursor Cloud specific instructions

### What this is
Single-product repo (not a monorepo): a Streamlit web app (Russian UI) for driver
drowsiness detection + nearest rest-stop finder. The real entrypoint is `app.py`
(see `README_STREAMLIT.md`). It is a single, self-contained process — no backend
API, database, queue, or auth. All data is local (`.pkl` models, `lol.xml` Haar
cascade, `bus_stops.json`).

### Running
The dependency refresh (see below) is handled by the startup update script. To run
the app in dev mode:

- `streamlit run app.py --server.port 8501 --server.headless true` (or `python run_app.py`, which also runs preflight checks). Serves on port 8501; health at `/_stcore/health`.

### Non-obvious caveats
- OpenCV version: `requirements.txt` declares `opencv-python-headless>=4.10` with no
  upper bound. Plain `pip install -r requirements.txt` pulls OpenCV **5.x**, which
  removes the top-level `cv2.CascadeClassifier` and breaks the drowsiness feature at
  runtime. The app requires OpenCV 4.x — keep `opencv-python-headless<5` installed
  (the update script re-pins it after installing requirements).
- Webcam: not available in the headless cloud VM, so the "Запустить камеру"
  (start camera) button won't produce video. The **rest-stop finder** (right-hand
  column) works fully headless and is the recommended smoke test: keep default
  Almaty coords (43.222, 76.851), click "🔍 Найти ближайшую остановку" → a stop with
  a distance and an interactive `pydeck` map render.
- Model `.pkl` files show as `modified` in `git status` due to Git LFS clean/smudge
  filters (`*.pkl` is LFS-tracked in `.gitattributes` but real binaries are committed
  directly). This is pre-existing noise — do not commit these files.
- Loading the pickled scikit-learn model prints `InconsistentVersionWarning`; this is
  harmless (model still works).
- The `Dockerfile` entrypoint points at the leftover template `src/streamlit_app.py`,
  NOT the real `app.py`.

### Tests / lint
No automated test suite or lint config exists in the repo. A quick syntax smoke check
is `python3 -m py_compile app.py run_app.py`.
