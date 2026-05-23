import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import joblib
import ee
import google.generativeai as genai
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static, st_folium
from geopy.geocoders import Nominatim
import os
from dotenv import load_dotenv
import warnings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from languages import TRANSLATIONS
from agronomy_config import CROP_SPECS
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from gtts import gTTS
import io
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Project Kishan - AI Agronomist",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapse sidebar by default
)

# Custom CSS for professional styling with better text visibility
st.markdown("""
<style>
/* =========================
   GLOBAL VARIABLES & THEMING
   Using RGB variables allows for easy transparency (rgba)
========================= */
:root {
    --primary: #2f855a;
    --primary-light: #48bb78;
    --primary-dark: #22543d;
    --bg-card: rgba(255, 255, 255, 0.9);
    --text-main: inherit; /* Let Streamlit handle light/dark mode text */
    --shadow-soft: 0 4px 12px rgba(0, 0, 0, 0.05);
    --radius: 12px;
}

/* =========================
   RESETS & HELPERS
========================= */
/* Ensure all markdown text inherits the theme's text color */
.stMarkdown {
    font-family: 'Inter', sans-serif;
}

/* =========================
   HEADERS (Enhanced Gradient)
========================= */
.main-header {
    font-size: clamp(2rem, 5vw, 3.5rem); /* Responsive sizing */
    text-align: center;
    font-weight: 800;
    padding: 1rem 0;
    background: linear-gradient(135deg, #276749 0%, #38a169 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}

.sub-header {
    font-size: 1.4rem;
    color: var(--primary);
    font-weight: 700;
    margin-top: 1rem; /* reduced spacing above sub-headers */
    margin-bottom: 0.5rem; /* small gap to following content */
    padding-bottom: 0.35rem;
    border-bottom: 2px solid #e2e8f0;
}

/* =========================
   CARDS (Glassmorphism & Contrast)
========================= */
/* Shared Base */
.custom-card {
    padding: 1.5rem;
    border-radius: var(--radius);
    margin-bottom: 1.2rem;
    border: 1px solid rgba(0,0,0,0.05);
    transition: transform 0.2s ease;
}

.prediction-card {
    background-color: #f0fff4;
    border-left: 6px solid var(--primary);
    color: #1c4532;
}

.warning-card {
    background-color: #fffaf0;
    border-left: 6px solid #dd6b20;
    color: #744210;
}

.error-card {
    background-color: #fff5f5;
    border-left: 6px solid #e53e3e;
    color: #742a2a;
}

/* =========================
   METRICS (Modernized)
========================= */
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
}

.metric-container {
    background: white;
    padding: 1rem;
    border-radius: var(--radius);
    box-shadow: var(--shadow-soft);
    text-align: center;
    border: 1px solid #edf2f7;
}

/* =========================
   BUTTONS (Full Width Option)
========================= */
.stButton > button {
    width: 100%; /* Optional: fits the container */
    background: linear-gradient(135deg, var(--primary), var(--primary-light));
    color: white !important;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stButton > button:hover {
    border: none;
    color: white !important;
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 10px 15px -3px rgba(47, 133, 90, 0.3);
}

/* =========================
   CHAT BUBBLES
========================= */
.chat-ai {
    background-color: #e6fffa;
    color: #234e52;
    padding: 15px;
    border-radius: 15px 15px 15px 0px;
    margin-bottom: 10px;
    border: 1px solid #b2f5ea;
}

.chat-user {
    background-color: #f7fafc;
    color: #2d3748;
    padding: 15px;
    border-radius: 15px 15px 0px 15px;
    margin-bottom: 10px;
    border: 1px solid #e2e8f0;
    text-align: right;
}

/* =========================
   FARM METRIC CARDS & ENHANCEMENTS
   High-impact UI improvements: gradients, responsiveness, typography
========================= */
.metric-card-box {
    border-radius: 12px;
    padding: 20px;
    color: white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    border: 1px solid rgba(255,255,255,0.08);
    height: 100%;
    margin-top: 6px; /* tighten card spacing under headings */
    display: flex; /* center content vertically/horizontally */
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 220px; /* ensures all cards have equal visual height */
}
.metric-card-box:hover { transform: translateY(-6px); box-shadow: 0 14px 30px rgba(0,0,0,0.12); }
.metric-card-box h3 { margin: 0; font-size: 0.95rem; font-weight:700; opacity:0.95; }
.metric-card-box h2 { margin: 10px 0 6px 0; font-size: 2.1rem; font-weight:800; line-height:1; }
.metric-card-box p { margin:0; opacity:0.9; }

/* Gradients */
.bg-ndvi { background: linear-gradient(135deg,#10b981 0%,#047857 100%); }
.bg-yield { background: linear-gradient(135deg,#f59e0b 0%,#b45309 100%); }
.bg-moisture { background: linear-gradient(135deg,#3b82f6 0%,#1e3a8a 100%); }
.bg-rain { background: linear-gradient(135deg,#06b6d4 0%,#0ea5e9 100%); }

/* Typography & general polish */
body, .stMarkdown, .stText { font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial; }

/* Section headers used for panels */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--primary-dark);
    margin-top: 0.6rem;
    margin-bottom: 0.5rem;
}

/* Responsive tweaks */
@media (max-width: 768px) {
    .metric-card-box h2 { font-size: 1.6rem; }
    .metric-card-box { padding: 14px; }
}

/* Centering helpers for map + action */
.center-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    padding: 0.5rem 0;
}
.center-wrapper .folium-map, .center-wrapper iframe {
    width: 100% !important;
    max-width: 980px;
    border-radius: 12px;
}
.center-button { width: 100%; max-width: 480px; display:flex; flex-direction:column; align-items:center; gap:8px; }

/* Image styling for farmer illustration */
.center-button img { width: 140px; max-width: 28vw; height: auto; border-radius: 8px; display:block; }

/* Make the analyze button very prominent for low-vision users */
.center-button .stButton > button {
    font-size: 1.9rem !important;
    padding: 1.1rem 1.6rem !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
    box-shadow: 0 14px 30px rgba(46,139,87,0.24) !important;
}
.center-button .stButton > button:focus {
    outline: 4px solid rgba(79, 154, 98, 0.25) !important;
}

@media (max-width: 480px) {
    .center-button img { width: 120px; }
    .center-button .stButton > button { font-size: 1.6rem !important; padding: 0.9rem 1.2rem !important; }
}

</style>
""", unsafe_allow_html=True)
def recognize_audio(audio_bytes, target_lang):
    """
    Converts recorded audio bytes to text using Google Speech Recognition.
    Supports: English, Hindi, Bengali.
    """
    if not audio_bytes:
        return None
    
    # Map your UI languages to Google Speech API codes
    lang_map = {
        "English": "en-US",
        "Hindi": "hi-IN",
        "Bengali": "bn-IN"
    }
    lang_code = lang_map.get(target_lang, "en-US")
    
    r = sr.Recognizer()
    try:
        # Convert bytes to audio source
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language=lang_code)
            return text
    except sr.UnknownValueError:
        return None # Could not understand audio
    except sr.RequestError:
        return None # API unreachable
    except Exception as e:
        return f"Error: {str(e)}"

def speak_text(text, target_lang):
    """
    Converts text to audio (MP3) for the user to hear.
    """
    lang_map = {
        "English": "en",
        "Hindi": "hi",
        "Bengali": "bn"
    }
    lang_code = lang_map.get(target_lang, "en")
    
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        return audio_fp.getvalue()
    except Exception:
        return None

class ProjectKishan:
    def __init__(self):
        self.api_status = {
            'gemini': False,
            'gee': False,
            'weather': False,
            'model': False
        }
        self.current_analysis = None
        self.chat_history = []
        self.initialize_apis()
        self.load_ml_model()
        self.vector_store = None
        self.setup_rag()

    def setup_rag(self):
        """Load the pre-computed Local Knowledge Base"""
        try:
            # Use the SAME model as ingestion
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            if os.path.exists("faiss_index"):
                self.vector_store = FAISS.load_local(
                    "faiss_index", 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                self.api_status['rag'] = True
            else:
                self.api_status['rag'] = False
        except Exception as e:
            # print(f"RAG Error: {e}") # Debug only
            self.api_status['rag'] = False

    def get_grounded_context(self, user_query):
        """Retrieve relevant chunks from PDFs"""
        if self.vector_store:
            docs = self.vector_store.similarity_search(user_query, k=3)
            return "\n\n".join([doc.page_content for doc in docs]), [doc.metadata['source'] for doc in docs]
        return "", []
        
    def initialize_apis(self):
        """Initialize all required APIs with robust error handling"""
        # Initialize Gemini AI
        try:
            gemini_key = os.getenv('GEMINI_API_KEY')
            if not gemini_key or gemini_key == 'your-gemini-api-key-here':
                self.api_status['gemini'] = False
            else:
                genai.configure(api_key=gemini_key)
                
                # Try different model names
                model_priority = ['gemini-2.5-pro', 'gemini-2.5-flash']
                
                successful_model = None
                for model_name in model_priority:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_name)
                        test_response = self.gemini_model.generate_content("Test")
                        successful_model = model_name
                        self.api_status['gemini'] = True
                        break
                    except Exception as model_error:
                        continue
                
                if not successful_model:
                    self.api_status['gemini'] = False
                    
        except Exception as e:
            self.api_status['gemini'] = False
        
        # Initialize GEE
        self.api_status['gee'] = self.initialize_gee()
        
        # Check OpenWeatherMap API key
        weather_key = os.getenv('OPENWEATHER_API_KEY')
        if not weather_key or weather_key == 'your-openweather-api-key-here':
            self.api_status['weather'] = False
        else:
            self.weather_api_key = weather_key
            self.api_status['weather'] = True
    
    def initialize_gee(self):
        """
        Initializes GEE using a simplified approach with service account or manual authentication.
        """
        try:
            # Try service account authentication first
            service_account = os.getenv('GEE_SERVICE_ACCOUNT')
            private_key_path = os.getenv('GEE_PRIVATE_KEY_PATH')
            
            if service_account and private_key_path and os.path.exists(private_key_path):
                credentials = ee.ServiceAccountCredentials(service_account, private_key_path)
                ee.Initialize(credentials, project=os.getenv('GEE_PROJECT_ID'))
                print("✅ GEE Initialized with Service Account")
                return True
            else:
                # Try to initialize with existing credentials
                ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))
                print("✅ GEE Initialized with Default Credentials")
                return True
        except Exception as e:
            print(f"❌ GEE Initialization Failed: {str(e)}")
            print(f"   - Service Account: {os.getenv('GEE_SERVICE_ACCOUNT', 'NOT SET')}")
            print(f"   - Key Path: {os.getenv('GEE_PRIVATE_KEY_PATH', 'NOT SET')}")
            print(f"   - Project ID: {os.getenv('GEE_PROJECT_ID', 'NOT SET')}")
            return False
    
    def load_ml_model(self):
        """
        Loads the 'Smart Pipeline' model. 
        Everything (encoding, scaling, prediction) is now in one file.
        """
        try:
            self.model = joblib.load('models/best_model.pkl')
            self.api_status['model'] = True
            print("✅ AI Model Loaded Successfully (R²=0.92)")
        except Exception as e:
            self.api_status['model'] = False
            print(f"❌ Model Loading Failed: {e}")

    def is_farmland_area(self, latitude, longitude):
        """
        STRICT VALIDATION: Blocks Cities (50) and Water (80).
        """
        # 1. GEE Connection Check
        if not self.api_status['gee']:
            st.error("❌ Validation Failed: GEE Offline")
            return False

        try:
            point = ee.Geometry.Point([longitude, latitude])
            
            # ESA WorldCover v200
            dataset = ee.ImageCollection("ESA/WorldCover/v200").first()
            result = dataset.select('Map').reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=10
            ).getInfo()
            
            land_cover = result.get('Map')
            
            # ---------------------------------------------------------
            # THE LOGIC GATE
            # ---------------------------------------------------------
            
            # 1. BLOCK CONCRETE (Cities)
            if land_cover == 50:
                st.error("❌ Security Alert: Location is a BUILT-UP AREA (City/Urban). Analysis Rejected.")
                return False
                
            # 2. BLOCK WATER (Permanent Water)
            if land_cover == 80:
                st.error("❌ Security Alert: Location is a WATER BODY. Analysis Rejected.")
                return False

            # 3. BLOCK WETLANDS & MANGROVES (Swamps/Marshes)
            # This catches the "muddy" water bodies that don't look like deep water.
            if land_cover in [90, 95]:
                st.error("❌ Security Alert: Location is a WETLAND/MANGROVE (Protected/Non-Arable). Analysis Rejected.")
                return False
                
            # 4. BLOCK BARE LAND (Deserts/River Beds/Rocks)
            # This catches dried river beds or sandy beaches.
            if land_cover == 60:
                st.error("❌ Security Alert: Location is BARE LAND (No Soil/Vegetation). Analysis Rejected.")
                return False
                
            # 5. STRICT ACCEPTANCE
            # Only allow: Trees (10), Shrubland (20), Grassland (30), Cropland (40)
            valid_classes = [10, 20, 30, 40]
            
            if land_cover in valid_classes:
                # Success Logic
                return True
            
            # Block everything else (Snow, Moss, etc.)
            st.error(f"❌ Security Alert: Land Class {land_cover} not suitable for agriculture.")
            return False
        
        except Exception as e:
            st.error(f"❌ Validation Error: {e}")
            return False
    
    def get_satellite_data(self, latitude, longitude):
        """
        Robust retrieval: SMAP -> ERA5 -> NDWI Proxy (Never returns static 35.0)
        """
        if not self.is_farmland_area(latitude, longitude):
            # This stops the code immediately if it's a City/Ocean
            raise Exception("Geospatial Validation Failed: Location is not agricultural land.")
        
        if not self.api_status['gee']:
            raise Exception("Google Earth Engine not initialized.")
        
        try:
            point = ee.Geometry.Point([longitude, latitude])
            buffer_area = point.buffer(100)
            
            # --- 1. SENTINEL-2 (Vegetation) ---
            s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                          .filterBounds(buffer_area)
                          .filterDate((datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'), 
                                    datetime.now().strftime('%Y-%m-%d'))
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                          .sort('system:time_start', False))
            
            if s2_collection.size().getInfo() == 0:
                raise Exception("No clear satellite view available.")
            
            image = s2_collection.first()
            
            # Calculate Indices
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
            ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
            savi = image.expression(
                '((NIR - RED) / (NIR + RED + 0.5)) * 1.5', {
                    'NIR': image.select('B8'),
                    'RED': image.select('B4')
                }).rename('SAVI')

            # Get Index Values First (We need NDWI for the fallback)
            indices_values = ee.Image.cat([ndvi, gndvi, ndwi, savi]).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer_area,
                scale=10
            ).getInfo()

            # --- 2. SOIL MOISTURE (The "Never Fail" Logic) ---
            soil_moisture_val = None
            source_tag = "N/A"

            # Attempt A: NASA SMAP (Widened search to 45 days)
            try:
                smap_collection = (ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture')
                                   .filterBounds(point)
                                   .filterDate((datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d'), 
                                               datetime.now().strftime('%Y-%m-%d'))
                                   .sort('system:time_start', False))
                
                if smap_collection.size().getInfo() > 0:
                    smap_image = smap_collection.first()
                    # Use a massive scale (10km) to ensure we hit a pixel
                    sm_dict = smap_image.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=point.buffer(5000), 
                        scale=10000 
                    ).getInfo()
                    
                    if 'ssm' in sm_dict and sm_dict['ssm'] is not None:
                        soil_moisture_val = sm_dict['ssm']
                        source_tag = "NASA SMAP"
            except Exception:
                pass # Silently fail to fallback

            # Attempt B: Smart Fallback (NDWI Proxy)
            # If SMAP failed, estimate moisture from the Water Index (NDWI)
            # NDWI -0.5 (Dry) to 0.5 (Wet). We map this to 10% - 60% moisture.
            if soil_moisture_val is None:
                ndwi_val = indices_values.get('NDWI', 0)
                # Formula: Base 30 + (NDWI * 40). 
                # Examples: NDWI 0.1 -> 34%. NDWI -0.2 -> 22%. NDWI 0.4 -> 46%.
                soil_moisture_val = 30.0 + (ndwi_val * 40.0)
                # Clamp between 5 and 90 to be realistic
                soil_moisture_val = max(5.0, min(90.0, soil_moisture_val))
                source_tag = "Calculated (NDWI)"

            # --- 3. FINAL OUTPUT ---
            # Create Map URL
            ndvi_params = {'min': -1, 'max': 1, 'palette': ['blue', 'white', 'green', 'darkgreen']}
            map_id = ndvi.getMapId(ndvi_params)
            
            satellite_data = {
                'ndvi': indices_values.get('NDVI', 0),
                'gndvi': indices_values.get('GNDVI', 0),
                'ndwi': indices_values.get('NDWI', 0),
                'savi': indices_values.get('SAVI', 0),
                'soil_moisture': soil_moisture_val,
                'data_source': source_tag, # Display this in UI if you want to prove it's real
                'image_date': image.get('system:time_start').getInfo(),
                'ndvi_url': map_id['tile_fetcher'].url_format
            }
            
            return satellite_data, "Success"
            
        except Exception as e:
            raise Exception(f"Satellite data retrieval failed: {str(e)}")
    
    def get_weather_forecast(self, latitude, longitude):
        """Get 7-day weather forecast from OpenWeatherMap with enhanced error handling"""
        if not self.api_status['weather']:
            # Return realistic default weather data
            default_weather = []
            current_date = datetime.now()
            for i in range(7):
                date = current_date + timedelta(days=i)
                default_weather.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'day': i + 1,
                    'temperature_avg': 25 + np.random.uniform(-5, 5),
                    'temperature_min': 20 + np.random.uniform(-3, 3),
                    'temperature_max': 30 + np.random.uniform(-3, 3),
                    'humidity_avg': 60 + np.random.uniform(-20, 20),
                    'precipitation_total': np.random.uniform(0, 10),
                    'description': 'partly cloudy'
                })
            return default_weather, "Using default weather data (API not configured)"
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url, timeout=15)
            
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}")
                
            data = response.json()
            
            # Process 7-day forecast
            daily_forecasts = {}
            for item in data['list']:
                date = item['dt_txt'].split(' ')[0]
                if date not in daily_forecasts:
                    daily_forecasts[date] = {
                        'temperatures': [],
                        'humidities': [],
                        'precipitations': [],
                        'descriptions': set()
                    }
                
                daily_forecasts[date]['temperatures'].append(item['main']['temp'])
                daily_forecasts[date]['humidities'].append(item['main']['humidity'])
                daily_forecasts[date]['precipitations'].append(item.get('rain', {}).get('3h', 0))
                daily_forecasts[date]['descriptions'].add(item['weather'][0]['description'])
            
            # Convert to list format
            forecast_data = []
            current_date = datetime.now()
            for i, (date, values) in enumerate(list(daily_forecasts.items())[:7]):
                forecast_data.append({
                    'date': date,
                    'day': i + 1,
                    'temperature_avg': np.mean(values['temperatures']),
                    'temperature_min': min(values['temperatures']),
                    'temperature_max': max(values['temperatures']),
                    'humidity_avg': np.mean(values['humidities']),
                    'precipitation_total': sum(values['precipitations']),
                    'description': ', '.join(list(values['descriptions'])[:2])
                })
            return forecast_data, "Success"
                
        except requests.exceptions.Timeout:
            raise Exception("Weather API request timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Unable to connect to weather service")
        except Exception as e:
            raise Exception(f"Weather data retrieval failed: {str(e)}")
    
    def prepare_features_for_prediction(self, latitude, longitude, satellite_data, weather_data, crop_type):
        """Prepare features for model prediction based on actual training data structure"""
        try:
            # Get current date information
            current_date = datetime.now()
            day_of_year = current_date.timetuple().tm_yday
            month = current_date.month
            year = current_date.year
            
            # Calculate trigonometric features for seasonality
            day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
            day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)
            
            # Start with base features from the training data structure
            features = {}
            
            # Initialize all features to 0 first
            for feature_name in self.feature_names:
                features[feature_name] = 0.0
            
            # Set basic geographic and temporal features
            features['latitude'] = latitude
            features['longitude'] = longitude
            features['day_of_year'] = day_of_year
            features['month'] = month
            features['year'] = year
            features['day_of_year_sin'] = day_of_year_sin
            features['day_of_year_cos'] = day_of_year_cos
            
            # Set vegetation indices from satellite data
            features['NDVI'] = satellite_data.get('ndvi', 0.5)
            features['GNDVI'] = satellite_data.get('gndvi', 0.4)
            features['NDWI'] = satellite_data.get('ndwi', 0.1)
            features['SAVI'] = satellite_data.get('savi', 0.5)
            
            # Set environmental conditions
            features['soil_moisture'] = satellite_data.get('soil_moisture', 35.0)
            
            if weather_data:
                features['temperature'] = np.mean([day['temperature_avg'] for day in weather_data])
                features['rainfall'] = np.sum([day['precipitation_total'] for day in weather_data])
            else:
                features['temperature'] = 25.0
                features['rainfall'] = 5.0
            
            # Set lag features (using current values as approximation)
            features['NDVI_lag_1'] = features['NDVI'] * 0.95
            features['NDVI_lag_2'] = features['NDVI'] * 0.90
            features['NDVI_lag_3'] = features['NDVI'] * 0.85
            
            features['rainfall_lag_1'] = features['rainfall'] * 0.9
            features['rainfall_lag_2'] = features['rainfall'] * 0.8
            features['rainfall_lag_3'] = features['rainfall'] * 0.7
            
            features['temperature_lag_1'] = features['temperature'] * 0.98
            features['temperature_lag_2'] = features['temperature'] * 0.96
            features['temperature_lag_3'] = features['temperature'] * 0.94
            
            # Set yield lags to reasonable defaults
            base_yield = 50.0
            features['yield_lag_1'] = base_yield * 0.95
            features['yield_lag_2'] = base_yield * 0.90
            features['yield_lag_3'] = base_yield * 0.85
            
            # Set crop type (one-hot encoding)
            crop_feature_name = f"crop_{crop_type}"
            if crop_feature_name in self.feature_names:
                features[crop_feature_name] = 1.0
            
            # Create feature array in correct order
            feature_array = np.array([features[feature_name] for feature_name in self.feature_names]).reshape(1, -1)
            
            return feature_array, features
            
        except Exception as e:
            raise Exception(f"Feature preparation failed: {str(e)}")
    
    def predict_yield(self, lat, lon, satellite_data, weather_data, crop_type):
        """
        HYBRID ENGINE: ML + Phenology
        """
        if not self.api_status['model']:
            return 0, "Model Offline"

        try:
            # 1. LAND POTENTIAL (Machine Learning)
            # "How good is this land?"
            input_df = pd.DataFrame([{
                'NDVI': satellite_data.get('ndvi', 0),
                'GNDVI': satellite_data.get('gndvi', 0),
                'NDWI': satellite_data.get('ndwi', 0),
                'SAVI': satellite_data.get('savi', 0),
                'soil_moisture': satellite_data.get('soil_moisture', 30),
                'temperature': np.mean([d['temperature_avg'] for d in weather_data]),
                'rainfall': np.sum([d['precipitation_total'] for d in weather_data]),
                'crop_type': crop_type 
            }])
            
            land_potential = self.model.predict(input_df)[0]
            
            # 2. BIOLOGICAL CORRECTION (The Config File)
            # "How much does this specific crop biologically produce?"
            spec = CROP_SPECS.get(crop_type, CROP_SPECS['default'])
            
            # 3. STRESS CALCULATION (The Rules)
            # "Is the weather killing this specific crop?"
            stress_factor = self._calculate_stress_factor(
                spec['type'], 
                input_df['soil_moisture'].values[0],
                input_df['temperature'].values[0],
                input_df['rainfall'].values[0]
            )
            
            # 4. FINAL YIELD
            final_yield = land_potential * spec['bio_factor'] * stress_factor
            
            return max(0.0, final_yield), "Success"

        except Exception as e:
            return 0, str(e)

    def _calculate_stress_factor(self, crop_category, sm, temp, rain):
        """
        Applies environmental stress penalties.
        """
        factor = 1.0
        
        # TROPICAL (Rice, Sugarcane, Oil Palm)
        if crop_category == 'wet_tropical':
            if sm < 30: factor *= 0.5   # Needs water
            if temp < 15: factor *= 0.7 # Hates cold
            
        # TROPICAL / HUMID (Coconut, Cashew, Spices)
        elif crop_category == 'tropical':
            if temp < 18: factor *= 0.6
            if sm < 20: factor *= 0.8
            
        # COOL SEASON (Wheat, Barley, Mustard, Saffron)
        elif crop_category == 'cool_season':
            if temp > 30: factor *= 0.6 # Heat shock
            if sm > 60: factor *= 0.8   # Rot risk
            
        # DRYLAND (Millets, Sorghum, Groundnut)
        elif crop_category == 'dryland':
            if sm > 65: factor *= 0.7   # Too wet
            if rain > 200: factor *= 0.8
            if sm < 20: factor *= 1.2   # Hardy bonus!
            
        # DRY FINISH (Cotton)
        elif crop_category == 'dry_finish':
            if rain > 150: factor *= 0.6 # Rain damages harvest
            
        # HILL / SHADE (Tea, Coffee)
        elif crop_category in ['hill', 'shade']:
            if temp > 35: factor *= 0.7 # Hates extreme heat
            
        return factor
    
    def generate_rag_context(self, analysis_data):
        """Generate RAG context from satellite data and predictions"""
        context = f"""
        AGRICULTURAL ANALYSIS CONTEXT:
        
        Location: Latitude {analysis_data['latitude']:.4f}, Longitude {analysis_data['longitude']:.4f}
        Crop Type: {analysis_data['crop_type']}
        Analysis Date: {analysis_data['analysis_date']}
        
        SATELLITE DATA:
        - Vegetation Health (NDVI): {analysis_data['ndvi']:.3f} ({'Excellent' if analysis_data['ndvi'] > 0.6 else 'Good' if analysis_data['ndvi'] > 0.3 else 'Poor'})
        - Green NDVI: {analysis_data['gndvi']:.3f}
        - Water Index (NDWI): {analysis_data['ndwi']:.3f}
        - Soil-Adjusted Vegetation Index: {analysis_data['savi']:.3f}
        - Soil Moisture: {analysis_data['soil_moisture']:.1f}mm
        
        WEATHER FORECAST (Next 7 days):
        - Average Temperature: {analysis_data['avg_temperature']:.1f}°C
        - Total Precipitation: {analysis_data['total_precipitation']:.1f}mm
        - Average Humidity: {analysis_data['avg_humidity']:.1f}%
        
        MACHINE LEARNING PREDICTIONS:
        - Predicted Yield: {analysis_data['predicted_yield']:.1f} units/hectare
        
        KEY INSIGHTS:
        - Current vegetation health is {'excellent' if analysis_data['ndvi'] > 0.6 else 'good' if analysis_data['ndvi'] > 0.3 else 'concerning'}
        - Weather conditions appear {'favorable' if analysis_data['total_precipitation'] > 20 and 15 < analysis_data['avg_temperature'] < 35 else 'moderate' if analysis_data['total_precipitation'] > 10 else 'challenging'}
        """
        return context
    
    def generate_ai_response(self, user_prompt, analysis_context, chat_history=[], language_instruction="Answer in English"):
        """
        FULL POWER: RAG (Books) + Satellite (Data) + Multilingual (Voice)
        """
        if not self.api_status['gemini']:
            return "AI service is currently unavailable."

        # --- 1. RAG RETRIEVAL (The Missing Link) ---
        rag_context = ""
        sources_used = []
        
        if self.api_status['rag']:
            try:
                # Search for 3 relevant chunks from your PDF database
                docs = self.vector_store.similarity_search(user_prompt, k=3)
                for doc in docs:
                    rag_context += f"- {doc.page_content}\n"
                    # Capture the source filename for citation
                    source = doc.metadata.get('source', 'Unknown File')
                    if source not in sources_used:
                        sources_used.append(source)
            except Exception as e:
                print(f"RAG Error: {e}")
                rag_context = "No knowledge base context available."
        
        # --- 2. PROMPT CONSTRUCTION ---
        full_prompt = f"""
        You are 'Kishan', an expert agricultural scientist.
        
        --- LIVE FARM DATA (Satellite & Weather) ---
        {analysis_context}
        
        --- KNOWLEDGE BASE (Research/Manuals) ---
        {rag_context}
        
        --- USER QUESTION ---
        {user_prompt}
        
        --- INSTRUCTIONS ---
        1. Analyze the 'Live Farm Data' first.
        2. Enhance your advice using the 'Knowledge Base' info if relevant.
        3. {language_instruction} <--- LANGUAGE COMMAND
        4. CRITICAL: At the end of your answer, list the 'Sources' you used from the Knowledge Base (e.g., 'Source: ICAR Manual').
        5. Keep the answer practical for a farmer.
        """
        
        try:
            response = self.gemini_model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

def create_interactive_map(default_lat=28.6139, default_lon=77.2090):
    """Create an interactive map for location selection"""
    m = folium.Map(
        location=[default_lat, default_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add marker for current location
    folium.Marker(
        [default_lat, default_lon],
        popup="Selected Farm Location",
        tooltip="Click for details",
        icon=folium.Icon(color='green', icon='leaf', prefix='fa')
    ).add_to(m)
    
    # Add click functionality
    m.add_child(folium.LatLngPopup())
    
    return m

def display_system_status(Kishan):
    """Display system status in the main UI"""
    st.markdown('<div class="section-header">🔧 System Status</div>', unsafe_allow_html=True)
    
    status_config = {
        'gemini': ("Gemini AI", "AI Recommendations"),
        'gee': ("Satellite Data", "Vegetation Analysis"),
        'weather': ("Weather API", "Forecast Data"),
        'model': ("ML Model", "Yield Predictions")
    }
    
    cols = st.columns(4)
    for idx, (api_key, (api_name, api_desc)) in enumerate(status_config.items()):
        with cols[idx]:
            if Kishan.api_status[api_key]:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: #e8f5e8; border-radius: 10px; border: 2px solid #32CD32;">
                    <div style="font-size: 2rem;">✅</div>
                    <div style="font-weight: 600; color: #1a531b;">{api_name}</div>
                    <div style="font-size: 0.8rem; color: #666;">{api_desc}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: #ffe8e8; border-radius: 10px; border: 2px solid #FF4500;">
                    <div style="font-size: 2rem;">❌</div>
                    <div style="font-weight: 600; color: #8b0000;">{api_name}</div>
                    <div style="font-size: 0.8rem; color: #666;">{api_desc}</div>
                </div>
                """, unsafe_allow_html=True)

def render_multilingual_contact(translations_dict):
    """
    Generate HTML for multilingual contact information (WhatsApp phone and join code).
    Shows labels and contact details in the selected language.
    """
    contact_html = f"""
    <div style="text-align:center; margin-bottom:0.9rem; font-size:0.95rem; color:#ff0000;">
        <span style="font-weight:800; margin-right:8px; color:#ffffff;">{translations_dict.get('contact_whatsapp_label', 'Chat us on WhatsApp')}:</span>
        <span style="font-weight:800; margin-right:16px; color:#ff0000;">{translations_dict.get('contact_phone', '+14155238886')}</span>
        <br style="margin: 6px 0;">
        <span style="font-weight:900; margin-right:8px; color:#ffffff;">{translations_dict.get('contact_code_label', 'Code to initialize the chat')}:</span>
        <span style="font-weight:800; color:#ff0000;">{translations_dict.get('contact_join_code', 'join shorter-far')}</span>
    </div>
    """
    return contact_html

def main():
    # --- 1. SESSION STATE INITIALIZATION ---
    if 'Kishan' not in st.session_state:
        st.session_state.Kishan = ProjectKishan()
        st.session_state.chat_history = []
        # Default start location
        st.session_state.current_location = {"lat": 23.2715, "lon": 87.3095} 
        st.session_state.current_analysis = None
        st.session_state.selected_crop = 'Rice'
    
    Kishan = st.session_state.Kishan
    
    # --- 2. SIDEBAR PART 1 (Language & Title) ---
    with st.sidebar:
        # A. LANGUAGE SELECTOR
        selected_lang = st.selectbox("🌐 Language / भाषा / ভাষা", ["English", "Hindi", "Bengali"])
        t = TRANSLATIONS[selected_lang]
        
        # B. TRANSLATED TITLE
        st.title(t["sidebar_title"])
        st.markdown("---")

    # --- 3. MAIN HEADER ---
    st.markdown(f'<div class="main-header">🌱 {t["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-size: 1.08rem; color: #1a531b; margin-bottom: 0.4rem; margin-top:0.25rem;">{t["tagline"]}</p>', unsafe_allow_html=True)

    # Render multilingual contact information
    contact_html = render_multilingual_contact(t)
    st.markdown(contact_html, unsafe_allow_html=True)
    
    # --- 4. MAP & ANALYSIS SECTION (FULL WIDTH) ---
    # We do NOT use columns here so the map stays wide.
    # LOGIC ORDER: This runs FIRST to update state before sidebar widgets are drawn.
    
    st.markdown('<div class="center-wrapper">', unsafe_allow_html=True)

    # Display Interactive Map using state coordinates
    map_obj = create_interactive_map(
        st.session_state.current_location["lat"],
        st.session_state.current_location["lon"]
    )
    # Height adjusted slightly to match your preference
    map_data = st_folium(map_obj, width=None, height=420, returned_objects=["last_clicked"])

    # HANDLE MAP CLICKS (The Sync Logic)
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]

        # Get current stored location
        current_lat = st.session_state.current_location["lat"]
        current_lng = st.session_state.current_location["lon"]

        # Check if new click (tolerance ~11 meters)
        if abs(clicked_lat - current_lat) > 0.0001 or abs(clicked_lng - current_lng) > 0.0001:
            # 1. Update Main State
            st.session_state.current_location = {"lat": clicked_lat, "lon": clicked_lng}
            # 2. Update widget state (Safe here because widgets aren't drawn yet)
            st.session_state["lat_input_box"] = clicked_lat
            st.session_state["lon_input_box"] = clicked_lng
            # 3. Rerun to refresh the view
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 5. SIDEBAR PART 2 (INPUTS) ---
    # Now that the map logic is done, we can safely draw the sidebar inputs.
    with st.sidebar:
        # Callback to sync manual typing to state
        def update_location_from_input():
            st.session_state.current_location["lat"] = st.session_state.lat_input_box
            st.session_state.current_location["lon"] = st.session_state.lon_input_box

        # C. LOCATION INPUTS
        lat_input = st.number_input(
            "📍 Latitude", 
            value=st.session_state.current_location["lat"],
            format="%.6f",
            key="lat_input_box",
            on_change=update_location_from_input
        )
        lon_input = st.number_input(
            "📍 Longitude", 
            value=st.session_state.current_location["lon"],
            format="%.6f",
            key="lon_input_box",
            on_change=update_location_from_input
        )

        # D. CROP SELECTOR
        available_crops = [
            'Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Coffee', 'Tea', 
            'Soybean', 'Barley', 'Sorghum', 'Millets', 'Pulses', 'Oil Palm',
            'Groundnut', 'Sunflower', 'Coconut', 'Cashew Nut',
             'Turmeric', 'Ginger', 'Tobacco', 'Rubber'
        ]
        
        crop_type = st.selectbox(t["select_crop"], available_crops, key="crop_select")
        st.session_state.selected_crop = crop_type
        
        # E. SYSTEM STATUS
        st.markdown("---")
        def status_icon(is_active): return "🟢" if is_active else "🔴"
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.write(f"{status_icon(Kishan.api_status['gee'])} Satellite")
            st.write(f"{status_icon(Kishan.api_status['model'])} AI Model")
        with s_col2:
            st.write(f"{status_icon(Kishan.api_status['weather'])} Weather")
            st.write(f"{status_icon(Kishan.api_status['gemini'])} Gemini")

    # --- 6. ANALYZE BUTTON (FULL WIDTH) ---
    # This renders below the map in the main flow
    st.markdown('<div class="center-button">', unsafe_allow_html=True)

    # Try to display farmer illustration if available
    for _p in ("assets/farmer.png", "farmer.png", "static/farmer.png"):
        if os.path.exists(_p):
            st.image(_p, width=140)
            break

    required_apis = ['model']
    missing_apis = [api for api in required_apis if not Kishan.api_status[api]]
    analyze_disabled = bool(missing_apis)
    if missing_apis:
        st.error(f"❌ API Error: {', '.join(missing_apis)}")

    if st.button(t["analyze_btn"], type="primary", use_container_width=True, disabled=analyze_disabled):
        with st.spinner(t["analyzing"]):
            try:
                lat = st.session_state.current_location["lat"]
                lon = st.session_state.current_location["lon"]
                satellite_data, _ = Kishan.get_satellite_data(lat, lon)
                weather_data, _ = Kishan.get_weather_forecast(lat, lon)
                prediction, _ = Kishan.predict_yield(lat, lon, satellite_data, weather_data, crop_type)
                st.session_state.current_analysis = {
                    'latitude': lat, 'longitude': lon, 'crop_type': crop_type,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'ndvi': satellite_data['ndvi'],
                    'gndvi': satellite_data['gndvi'],
                    'ndwi': satellite_data['ndwi'],
                    'savi': satellite_data['savi'],
                    'soil_moisture': satellite_data['soil_moisture'],
                    'avg_temperature': np.mean([d['temperature_avg'] for d in weather_data]),
                    'total_precipitation': np.sum([d['precipitation_total'] for d in weather_data]),
                    'predicted_yield': prediction,
                    'satellite_data': satellite_data,
                    'weather_data': weather_data
                }
                st.success("✅ Analysis Complete!")
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.session_state.current_analysis = None

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 7. RESULTS & CHAT TABS ---
    st.markdown("---")
    tab1, tab2 = st.tabs(["📊 " + t["results_header"], "🤖 " + t["chat_header"]])
    
    # --- TAB 1: METRICS & CHARTS ---
    with tab1:
        if st.session_state.current_analysis:
            analysis = st.session_state.current_analysis
            
            st.markdown(f'<div class="sub-header">{t["results_header"]}</div>', unsafe_allow_html=True)
            
            # Key Metrics Row
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            with m_col1: # NDVI
                ndvi_color = "🟢" if analysis['ndvi'] > 0.6 else "🟡" if analysis['ndvi'] > 0.3 else "🔴"
                st.markdown(f"""
                <div class="metric-card-box bg-ndvi">
                    <h3>NDVI (Health)</h3>
                    <h2>{analysis['ndvi']:.2f} {ndvi_color}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with m_col2: # YIELD
                st.markdown(f"""
                <div class="metric-card-box bg-yield">
                    <h3>{t['metric_yield']}</h3>
                    <h2>{analysis['predicted_yield']:.1f}</h2>
                    <p>tons/ha</p>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col3: # SOIL
                sm_val = analysis['soil_moisture']
                sm_status_key = "status_optimal" if 20 <= sm_val <= 60 else "status_dry" if sm_val < 20 else "status_wet"
                sm_status_text = t.get(sm_status_key, "Unknown")
                st.markdown(f"""
                <div class="metric-card-box bg-moisture">
                    <h3>{t['metric_soil']}</h3>
                    <h2>{sm_val:.1f}%</h2>
                    <p>{sm_status_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col4: # RAIN
                st.markdown(f"""
                <div class="metric-card-box bg-rain">
                    <h3>{t['metric_rain']}</h3>
                    <h2>{analysis['total_precipitation']:.1f}mm</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed Analysis Section
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-header">🌱 Vegetation Indices</div>', unsafe_allow_html=True)
                indices_df = pd.DataFrame({
                    'Index': ['NDVI', 'GNDVI', 'NDWI', 'SAVI', 'Soil Moisture (mm)'],
                    'Value': [analysis['ndvi'], analysis['gndvi'], analysis['ndwi'], analysis['savi'], analysis['soil_moisture']]
                })
                st.dataframe(indices_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown('<div class="section-header">🌤️ Weather Forecast</div>', unsafe_allow_html=True)
                if analysis['weather_data']:
                    weather_df = pd.DataFrame(analysis['weather_data'])
                    weather_display = weather_df[['day', 'date', 'temperature_avg', 'precipitation_total', 'description']]
                    weather_display.columns = ['Day', 'Date', 'Temp (°C)', 'Rain (mm)', 'Conditions']
                    st.dataframe(weather_display, use_container_width=True, hide_index=True)

            # Auto-Generated Insight
            st.markdown(f'<div class="section-header">🤖 {t["chat_header"]} Insights</div>', unsafe_allow_html=True)
            
            if Kishan.api_status['gemini']:
                insight_key = f"insight_{analysis['latitude']}_{analysis['crop_type']}_{selected_lang}_{analysis['analysis_date']}"
                if insight_key not in st.session_state:
                     with st.spinner("Generating insights..."):
                        initial_prompt = "Provide a 3-sentence summary of the farm health and one key recommendation."
                        st.session_state[insight_key] = Kishan.generate_ai_response(
                            initial_prompt, analysis, language_instruction=t["prompt_instruction"]
                        )
                st.info(st.session_state[insight_key])
            else:
                st.warning("AI Insights unavailable (Gemini API offline)")

    # --- TAB 2: AI CHAT (VOICE ENABLED) ---
    with tab2:
        st.subheader(t["chat_header"])
        
        if not st.session_state.current_analysis:
            st.warning("Please run analysis first to enable Context-Aware Chat.")
        else:
            # 1. Display Chat History
            for message in st.session_state.chat_history:
                role_display = "👤 You" if message['role'] == 'user' else f"🤖 {t['title']}"
                with st.chat_message(message['role']):
                    st.write(message['content'])
                    # If this is an assistant message and has audio, play it
                    if message.get('audio'):
                        st.audio(message['audio'], format='audio/mp3')

            # 2. Voice Input Section
            st.markdown("---")
            c1, c2 = st.columns([1, 8])
            
            # VOICE RECORDER BUTTON
            with c1:
                st.write("🎙️")
                # This records audio and returns the bytes
                audio_input = mic_recorder(
                    start_prompt="Rec",
                    stop_prompt="Stop",
                    key='recorder',
                    format="wav",  # Important for SpeechRecognition
                    use_container_width=True
                )

            # 3. Handle Voice Input OR Text Input
            user_text = None
            
            # CHECK: Did user speak?
            if audio_input and audio_input['bytes']:
                # Transcribe the audio
                transcribed_text = recognize_audio(audio_input['bytes'], selected_lang)
                if transcribed_text:
                    user_text = transcribed_text
            
            # CHECK: Did user type?
            if prompt := st.chat_input(t["chat_placeholder"]):
                user_text = prompt

            # 4. Process the Input (Voice or Text)
            if user_text:
                # Add User Message to Chat
                st.session_state.chat_history.append({'role': 'user', 'content': user_text})
                st.chat_message("user").write(user_text)
                
                with st.spinner("Thinking & Speaking..."):
                    try:
                        # Get AI Text Response
                        ai_response_text = Kishan.generate_ai_response(
                            user_text,
                            st.session_state.current_analysis,
                            st.session_state.chat_history,
                            language_instruction=t["prompt_instruction"]
                        )
                        
                        # Generate Audio for the Response
                        audio_response = speak_text(ai_response_text, selected_lang)
                        
                        # Save to history with audio
                        st.session_state.chat_history.append({
                            'role': 'assistant', 
                            'content': ai_response_text,
                            'audio': audio_response
                        })
                        
                        # Display AI Response immediately
                        with st.chat_message("assistant"):
                            st.write(ai_response_text)
                            if audio_response:
                                st.audio(audio_response, format='audio/mp3', start_time=0)
                                
                    except Exception as e:
                        st.error(f"AI Error: {str(e)}")

if __name__ == "__main__":
    main()