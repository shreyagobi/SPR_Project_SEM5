# Voice Enrollment & Retraining Guide

This guide explains how anyone who clones this repository can run the **Web UI**, record their own spoken digits (`0`–`9`), and retrain the **DTW**, **GMM**, **HMM**, and **GMM-UBM Speaker Verification** models.

---

## 1. Environment Setup

Create and activate a Python virtual environment (Python 3.10+ recommended), then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy scipy librosa scikit-learn hmmlearn sounddevice soundfile joblib numba fastapi uvicorn python-multipart
```

---

## 2. Launch the Web UI

Start the local FastAPI server:

```powershell
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser (use `localhost` / `127.0.0.1` so the browser allows microphone access).

---

## 3. How to Record Your Voice (Tab 2: **Record Voice**)

Yes — **you should record multiple clips for every digit (`0`–`9`)** (recommended: **10–15 clips per digit** if only one person is using it, or **5–10 clips per digit per teammate** if multiple people are enrolling).

1. Enter **Your Name** (e.g., `vanshika`, `alex`) so your clips are tagged under your name.
2. Choose how you want to record multiple clips per digit:
   * **Option A — Round-robin (`Auto-advance 0 → 9` checked, default):** Click **Record Digit (Auto-Save)** once, then speak `0`, pause, `1`, pause, `2`... through `9`. After each clip saves, it automatically advances to the next digit and wraps `9 → 0` so you can do multiple rounds without touching the mouse.
   * **Option B — One digit at a time (uncheck `Auto-advance 0 → 9`):** Click a digit tile (e.g., `0`), click **Record Digit (Auto-Save)**, and say `"zero"` 10 times in a row with a brief pause between each. The tile counter updates live (`0 clips` → `10 clips`). Click **Stop & Save**, then click `1` and repeat.
3. After clicking **Record Digit (Auto-Save)**, wait `0.5s` for room-noise calibration, then speak clearly:
   * When you pause after a digit, the VAD automatically trims silence and **saves the clip immediately** (or click **Stop & Save** at any time to save the current clip and stop).
   * Raw clips are saved to `my_records/<your_name>/digit_<d>/`, and trimmed clips are automatically split **80% train / 20% test** into `my_train_records/` and `my_test_records/`.
   * **Multiple teammates & adding more later:** Existing recordings are never overwritten—new clips are appended with the next index, so you or your teammates can add more clips for any digit at any time.

---

## 4. How to Retrain (Tab 3: **Train Models**)

1. Open the **3. Train Models** tab.
2. *(Optional)* Select an **Owner for 2FA (GMM-UBM)** if you want the app to verify *who* is speaking in addition to the PIN digits.
3. *(Optional)* Check **Use CMVN** if teammates recorded on different microphones or at very different volumes.
4. Click **Train All Models Now**.
   * All 5 digit models (`HMM (hmmlearn)`, `GMM (custom)`, `GMM (sklearn)`, `HMM (custom)`, `DTW (multi-template)`), per-digit rejection thresholds, and the optional GMM-UBM speaker verifier are saved to `./models/`, and test accuracy is displayed in the table.

*(CLI alternative: `.venv\Scripts\python.exe train_all.py --owner <your_name> [--cmvn]`)*

**Which dataset is used for training?** Training reads from `my_train_records/` and `my_test_records/` if those folders exist locally. If they do not exist yet, it falls back to the shared dataset in `processed_train_records/` and `processed_test_records/` that is already in the repository. Once you have recorded your own clips, the team folders take over, so the shared dataset is no longer used. The GMM-UBM speaker verifier also takes its background (impostor) data from `processed_train_records/`, and its enrolled speaker from `my_records/<owner>/`.

---

## 5. Run the Demo (Tab 1: **Unlock Demo**)

1. Choose your model (default: **HMM (hmmlearn)**) and set your **Target PIN** (default: `1234`).
2. Keep **Reject unclear / non-digits** checked so background noise, desk taps, or non-digit words are ignored.
3. *(Optional)* Check **Require owner voice (GMM-UBM 2FA)** to enforce speaker verification.
4. Click **Start Listening**, wait `0.5s` for noise calibration, and speak each digit of your PIN with a brief pause between digits.

---

## 6. Team Git Workflow (Syncing Voices Across Teammates)

### Where Files Are Saved
* **Raw recordings:** `my_records/<your_name>/digit_<d>/<idx>_<d>.wav`
* **Trimmed 80/20 split:** `my_train_records/digit_<d>/<your_name>_<idx>_<d>.wav` and `my_test_records/digit_<d>/<your_name>_<idx>_<d>.wav`
* **Trained models:** `models/*.joblib` *(ignored by `.gitignore` so teammates don't hit binary merge conflicts — everyone retrains locally after pulling).*
* **Original shared dataset (do not edit as part of enrollment):** `records/`, `processed_train_records/`, `processed_test_records/`. These are tracked in the repository and are only used as a fallback for training (see Section 4).

Because every recording is stored under your own `<your_name>` folder/prefix, multiple teammates can record on separate laptops without filename collisions or merge conflicts.

### How to Sync Recordings to `main`
1. Clone the repository, set up `.venv` (Section 1), and launch the Web UI (Section 2).
2. In **Tab 2 (Record Voice)**, enter your name (e.g., `vanshika`, `shreya`, or `kusum`) and record **5–10 clips per digit**.
3. Commit and push **only your audio recordings**. Commit first, then rebase onto `main`, so the rebase is not blocked by uncommitted work:
   ```powershell
   git add my_records my_train_records my_test_records
   git commit -m "Add voice recordings for <your_name>"
   git pull --rebase origin main
   git push origin main
   ```
   *(Or push to your own branch and open a Pull Request into `main`.)*
4. Once all teammates' recordings are merged into `main`:
   ```powershell
   git pull origin main
   ```
   Then open **Tab 3 (Train Models)** and click **Train All Models Now** (check **Use CMVN** if teammates recorded on different microphones). Your local `models/` folder will now be trained on everyone's combined voices!

### Basic Git Commands Quick Reference (For Teammates & AI Agents)
* **First-time clone:**
  ```powershell
  git clone https://github.com/svanshika-afk/SPR_Project_SEM5.git
  cd SPR_Project_SEM5
  ```
* **Check what changed (verify `.venv/` and `models/*.joblib` are ignored):**
  ```powershell
  git status
  ```
* **Get latest recordings/code from `main` before starting work:**
  ```powershell
  git pull --rebase origin main
  ```
* **Stage, commit, and push only your voice recordings:**
  ```powershell
  git add my_records my_train_records my_test_records
  git commit -m "Add voice recordings for <your_name>"
  git pull --rebase origin main
  git push origin main
  ```
* **Undo uncommitted changes to a file if needed:**
  ```powershell
  git checkout -- <file_path>
  ```
* **Rule for AI agents:** Never force-push (`git push --force`) to `main`, never commit `.venv/` or `models/*.joblib`, and always `git pull --rebase origin main` before pushing new recordings.

---

## 7. Quick Troubleshooting
* **Mic permission blocked:** Make sure you open `http://127.0.0.1:8000` or `http://localhost:8000` (not `0.0.0.0` or a LAN IP over plain HTTP).
* **"Too long (X s) — please pause between digits." (Tab 1, unlock):** Speak one digit at a time and leave ~0.5s of silence between digits.
* **"Clip too long (X s) — speak only one digit." (Tab 2, recording):** Each clip should contain a single digit. Stop, then record again.
* **"Too quiet or too short — please speak clearly and try again." (Tab 2, recording):** Move closer to the microphone, speak louder, and wait for the 0.5s noise calibration to finish before speaking.
* **Live demo with a new guest:** Anyone can enter their name in **Tab 2**, record a few clips, click **Train All Models Now** in **Tab 3**, and test immediately in **Tab 1**.
