# RHYTHM — TRACE

**Trust-Routed Artefact-Classified ECG**

TRACE is a software-based ECG analysis prototype designed to identify motion artefacts, assess heartbeat reliability, and support evidence-based cardiac event review.

### Key Features
- **Artefact Classification:** Identifies ECG signal disturbances such as baseline wander, motion artefacts, and lead dropout.
- **Adaptive Signal Processing:** Applies artefact-specific filtering and marks unreliable segments rather than blindly reconstructing missing signals.
- **Beat Trust Scoring:** Evaluates heartbeat reliability using signal quality and rhythm characteristics.
- **Evidence-Based Alerts:** Distinguishes trusted detections from reconstruction-dependent events requiring further review.
- **Interactive Dashboard:** Visualizes ECG signals, artefacts, trust scores, and event evidence.
- **Audit Trail:** Records event decisions and transitions for traceability.

### Technology Stack
- **Frontend:** React, TypeScript, Vite, uPlot
- **Backend:** Python, FastAPI
- **Signal Analysis:** ECG signal processing, R-peak detection, rhythm analysis, and motion-aware filtering
- **Data:** PhysioNet ECG datasets and simulated motion-sensor inputs

### How It Works
1. Ingests ECG data and motion-sensor inputs.
2. Detects and classifies potential signal artefacts.
3. Applies appropriate signal-quality and recovery methods.
4. Scores heartbeat reliability and evaluates possible cardiac events.
5. Presents event evidence and confidence-related status in the dashboard.

### Project Status
Research and prototype development in progress. Motion-sensor inputs may be simulated, and performance requires further validation.

**Disclaimer:** TRACE is a research prototype, not a clinically validated medical device. Its outputs must not be used as a substitute for professional medical assessment.

### Team
Developed for **MEDAITHON'26**.
