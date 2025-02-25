import requests

VOICEVOX_URL = "http://127.0.0.1:50021"

# Coba tes endpoint lain
try:
    response = requests.get(f"{VOICEVOX_URL}/speakers")
    if response.status_code == 200:
        print("✅ VoiceVox API berjalan dengan baik.")
    else:
        print(f"❌ VoiceVox API tidak merespons dengan baik: {response.status_code}")
except Exception as e:
    print(f"❌ Tidak bisa terhubung ke VoiceVox: {e}")
