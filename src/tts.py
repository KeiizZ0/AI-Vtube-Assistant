import requests
import numpy as np
import soundfile as sf
import pyaudio
import io

VOICEVOX_URL = "http://127.0.0.1:50021"
VOICE_ID = 68  # Sesuaikan dengan config.yaml

def text_to_speech(text):
    """Mengubah teks menjadi suara menggunakan VoiceVox API dan langsung memutarnya."""
    try:
        # 1️⃣ Buat audio query ke VoiceVox
        params = {"text": text, "speaker": VOICE_ID}
        res1 = requests.post(f"{VOICEVOX_URL}/audio_query", params=params)
        if res1.status_code != 200:
            print(f"❌ Gagal mendapatkan audio query dari VoiceVox! Status: {res1.status_code}")
            return False

        audio_query = res1.json()

        # 2️⃣ Hasilkan suara dari VoiceVox
        res2 = requests.post(f"{VOICEVOX_URL}/synthesis", json=audio_query, params={"speaker": VOICE_ID})
        if res2.status_code != 200:
            print(f"❌ Gagal melakukan sintesis suara! Status: {res2.status_code}")
            return False

        # 3️⃣ Load audio ke buffer tanpa menyimpan file
        audio_data = io.BytesIO(res2.content)
        audio_array, samplerate = sf.read(audio_data, dtype="int16")

        # 4️⃣ Putar audio langsung ke speaker
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=samplerate,
                        output=True)

        print("🔊 Memutar suara langsung ke speaker...")
        stream.write(audio_array.tobytes())

        stream.stop_stream()
        stream.close()
        p.terminate()

        print("🎶 Suara selesai diputar!")
        return True

    except Exception as e:
        print(f"❌ Error dalam proses TTS: {e}")
        return False

# Contoh penggunaan
if __name__ == "__main__":
    text = "こんにちは、ダーリン！今日はどんな気分ですか？"
    text_to_speech(text)
