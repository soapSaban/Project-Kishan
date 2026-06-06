⚠️ Repository Note: This is an architectural mirror of the [Calcutta_Hacks] project. The original deployment repository is hosted [https://github.com/Jontybr18211/Project-Kishan-Kolkata-Hacks].

I served as the core ML/Backend Engineer for this project, building the Google Earth Engine data pipelines, the FAISS vector database integration, and the local RAG architecture. I am hosting this mirror to showcase the backend systems I developed locally during the event.
---

# 🌱 Project Kishan: AI Agronomist for India 🇮🇳

> **Bridging the gap between Satellite Intelligence and Rural Farmers using Generative AI.**

## Overview

**Project Kishan** is a multimodal, multilingual AI platform designed to help Indian farmers make data-driven decisions. Unlike traditional dashboards that require literacy and technical knowledge, Kishan uses a **Voice-First** and **WhatsApp-First** approach.

It integrates real-time satellite telemetry (Sentinel-2, NASA SMAP) with a Hybrid Machine Learning engine to predict crop yields and provide personalized agronomic advice in **Hindi, Bengali, and English**.

##  Key Features

### 1. Space-to-Farm Intelligence

* **Real-time Satellite Analysis:** Fetches live NDVI (Vegetation Health), NDWI (Water Stress), and Soil Moisture data using the **Google Earth Engine (GEE) API**.
* **Precision Weather:** Hyper-local 7-day forecasts using OpenWeatherMap.

### 2. Hybrid AI Engine

* **Yield Prediction:** A Random Forest Regressor trained on historical agricultural data.
* **Biological Logic Layer:** A custom expert system (`agronomy_config.py`) that cross-references ML predictions with FAO biological constraints to prevent unrealistic "hallucinations."
* **RAG Chatbot:** Context-aware chat powered by **Google Gemini**, grounded in live farm data.

### 3. Radical Accessibility

* **Trilingual Support:** Full UI and AI responses in **English, Hindi, and Bengali**.
* **Voice Assistant:** Farmers can speak their query and hear the response (STT/TTS integration).
* **WhatsApp Bot:** A Flask-based backend allows farmers to send their Live Location and Voice Notes via WhatsApp to get instant analysis.

---

## Architecture

1. **Frontend:** Streamlit (Web App) / WhatsApp (Mobile Interface).
2. **Orchestrator:** Python (Logic Layer).
3. **Data Ingestion:**
* *Satellite:* Google Earth Engine (PyEarthengine).
* *Weather:* OpenWeatherMap API.
* *Knowledge:* RAG Context injection.


4. **AI Models:**
* *Reasoning:* Google Gemini Pro.
* *Prediction:* Scikit-Learn (Random Forest).


5. **Deployment:** Localhost / Ngrok Tunnelling.

---

## Installation & Setup

### Prerequisites

* Python 3.8+
* A Google Cloud Project with **Earth Engine API** enabled.
* API Keys for **Google Gemini** and **OpenWeatherMap**.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/project-kishan.git
cd project-kishan

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

*(Note: You also need `ffmpeg` installed on your system for voice processing)*.

### 3. Configure Secrets

Create a file named `.env` (or `.streamlit/secrets.toml` for Streamlit) and add your keys:

```toml
GEMINI_API_KEY = "your_google_ai_key"
OPENWEATHER_API_KEY = "your_weather_key"
# Google Earth Engine credentials usually require authentication via terminal command:
# earthengine authenticate

```

### 4. Run the Web App

```bash
streamlit run app.py

```

### 5. Run the WhatsApp Bot (Optional)

To enable the WhatsApp interface:

```bash
# Terminal 1: Start the Flask Server
python backend_whatsapp.py

# Terminal 2: Expose to internet
ngrok http 5000

```

*Copy the Ngrok URL to your Twilio Sandbox settings.*

---

## Project Structure

```text
project-kishan/
├── app.py                  # Main Streamlit Dashboard (UI & Logic)
├── backend_whatsapp.py     # Flask Server for WhatsApp Bot
├── agronomy_config.py      # Expert Rules Layer (Biological Constraints)
├── languages.py            # Dictionary for Multilingual Text
├── models/
│   └── best_model.pkl      # Pre-trained ML Yield Predictor
├── assets/                 # Images & Icons
├── requirements.txt        # Python Dependencies
└── README.md               # Documentation

```

##Screenshots

| Interactive Map | Multilingual Chat | Yield Analysis |
| --- | --- | --- |
| *Map Selection Interface* | *Bengali Voice Chat Demo* | *Real-time NDVI & Yield* |
| *(Add Screenshot)* | *(Add Screenshot)* | *(Add Screenshot)* |

---

## Tech Stack

* **Core:** Python, Pandas, NumPy.
* **ML & AI:** Scikit-Learn, Google Gemini (GenAI), SpeechRecognition, gTTS.
* **Geospatial:** Folium, Streamlit-Folium, Google Earth Engine API.
* **Backend:** Flask, Twilio (WhatsApp), Ngrok.
* **Frontend:** Streamlit.

---

## Future Roadmap

* [ ] **Offline Mode:** SMS-based alerts for farmers without data.
* [ ] **Marketplace:** Connect farmers directly to buyers based on predicted harvest dates.

---

## Contributing

Contributions are welcome! Please fork the repo and submit a Pull Request.



---

**Made with ❤️ for Indian Agriculture.**
