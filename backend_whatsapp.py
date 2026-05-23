from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os
import requests
import traceback
from pydub import AudioSegment
import speech_recognition as sr

# --- 1. IMPORT YOUR CLASS SAFELY ---
# Make sure your app.py has the "if __name__ == '__main__':" block at the bottom!
try:
    from app3 import ProjectKishan 
    # Initialize the Brain once
    Kishan = ProjectKishan()
    print("✅ Project Kishan Brain Loaded Successfully!")
except Exception as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    # We continue so the server starts, but it will fail on request

app = Flask(__name__)

# --- MEMORY ---
USER_SESSIONS = {} 

@app.route("/whatsapp", methods=['POST'])
def whatsapp_webhook():
    resp = MessagingResponse()
    msg = resp.message()
    
    try:
        # 1. Get Incoming Data
        sender_id = request.values.get('From', '')
        incoming_msg = request.values.get('Body', '').lower()
        num_media = int(request.values.get('NumMedia', 0))
        latitude = request.values.get('Latitude')
        longitude = request.values.get('Longitude')

        print(f"📩 Received from {sender_id}: Lat={latitude}, Text='{incoming_msg}', Media={num_media}")

        # --- CASE A: LIVE LOCATION ---
        if latitude and longitude:
            USER_SESSIONS[sender_id] = {
                'lat': float(latitude),
                'lon': float(longitude)
            }
            msg.body("✅ Location Secured! Now ask me: 'How is my soil?' or send a Voice Note.")
            return str(resp)

        # --- CHECK SESSION ---
        if sender_id not in USER_SESSIONS:
            msg.body("📍 I need your location first! Tap (+) -> Location -> Share Live Location.")
            return str(resp)

        # --- CASE B: PROCESS INPUT (VOICE or TEXT) ---
        user_query = ""
        
        # If Voice Note
        if num_media > 0:
            media_type = request.values.get('MediaContentType0')
            if 'audio' in media_type:
                media_url = request.values.get('MediaUrl0')
                msg.body("🎧 Listening to your voice note...")
                # Download and Convert
                user_query = process_audio(media_url)
                if not user_query:
                    msg.body("❌ I couldn't hear that properly. Please try again.")
                    return str(resp)
        else:
            # If Text
            user_query = incoming_msg

        if not user_query:
            msg.body("❌ Please send text or audio.")
            return str(resp)

        # --- CASE C: RUN AI ANALYSIS ---
        # This is where it usually crashes if keys are missing
        user_data = USER_SESSIONS[sender_id]
        lat, lon = user_data['lat'], user_data['lon']
        
        # Get Real Data
        satellite, _ = Kishan.get_satellite_data(lat, lon)
        weather, _ = Kishan.get_weather_forecast(lat, lon)
        
        # Prepare Context
        analysis_context = {
            'ndvi': satellite.get('ndvi', 0),
            'soil_moisture': satellite.get('soil_moisture', 0),
            'latitude': lat,
            'longitude': lon,
            'weather_data': weather
        }
        
        # Ask Gemini
        ai_reply = Kishan.generate_ai_response(
            user_query, 
            analysis_context,
            chat_history=[],
            language_instruction="Reply in the same language as the user. Keep it short (under 50 words) for WhatsApp."
        )
        
        msg.body(ai_reply)

    except Exception as e:
        # --- THE ERROR REPORTER ---
        # If anything crashes, it tells you WHY on WhatsApp
        error_message = f"⚠️ System Error: {str(e)}"
        print(traceback.format_exc()) # Print full error to your PC terminal
        msg.body(error_message)

    return str(resp)

def process_audio(media_url):
    try:
        # Download
        r = requests.get(media_url)
        with open("temp.ogg", "wb") as f: f.write(r.content)
        
        # Convert OGG -> WAV (Requires FFmpeg installed on system)
        sound = AudioSegment.from_ogg("temp.ogg")
        sound.export("temp.wav", format="wav")
        
        # Transcribe
        r = sr.Recognizer()
        with sr.AudioFile("temp.wav") as source:
            audio = r.record(source)
            # Detects Hindi/Bengali automatically
            text = r.recognize_google(audio)
            print(f"🗣️ Transcribed: {text}")
            return text
    except Exception as e:
        print(f"Audio Error: {e}")
        return None

if __name__ == "__main__":
    app.run(debug=True, port=5000)