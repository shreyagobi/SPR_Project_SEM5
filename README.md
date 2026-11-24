# Voice-PIN Security Demo (SPR Project)

Real-time spoken-digit (`0`–`9`) PIN unlock system built with **DTW**, **GMM**, **HMM**, and **GMM-UBM** speaker verification.

## Features
* **Web UI (`server.py` + `static/`)**: Browser-based mic capture (`AudioWorklet`) with adaptive noise-floor calibration, VAD endpointing, and OOV/margin rejection.
* **Three Classical SPR Models**:
  * **DTW** (`dtw.py`) — Numba-accelerated multi-template Dynamic Time Warping
  * **GMM** (`gmm.py`, `gmm_from_sklearn.py`) — Diagonal-covariance Gaussian Mixture Models
  * **HMM** (`hmm.py`, `hmm_from_hmmlearn.py`) — Hidden Markov Models
* **Two-Factor Speaker Verification (`speaker_id.py`)**: GMM-UBM with MAP adaptation to verify *who* is speaking.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy scipy librosa scikit-learn hmmlearn sounddevice soundfile joblib numba fastapi uvicorn python-multipart

.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

## Recording, Training & Team Workflow
See **[VOICE_ENROLLMENT.md](VOICE_ENROLLMENT.md)** for full instructions on recording your voice, retraining models, and syncing recordings across teammates.
