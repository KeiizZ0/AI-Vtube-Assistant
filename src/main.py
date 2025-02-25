import asyncio
import requests
import os
import yaml
import whisper
import torch
import sounddevice as sd
import numpy as np
import queue
import wave
import tempfile
from pydub import AudioSegment
from pydub.playback import play

from ai import GoogleAIChat

class TextToSpeech:
    def __init__(self, config):
        """Inisialisasi VoiceVox TTS dengan sesi HTTP dan optimalisasi GPU."""
        self.config = config["voicevox"]
        self.voice_id = self.config["voice_id"]
        self.base_url = self.config["base_url"]
        self.session = requests.Session()

        print("✅ VoiceVox TTS siap digunakan.")

    async def speak(self, text):
        """Mengubah teks menjadi suara dan langsung memutarnya tanpa menghentikan event loop utama."""
        if not text.strip():
            return

        print(f"🗣️ AI Waifu berkata: {text}")

        try:
            response = self.session.post(
                f"{self.base_url}/audio_query",
                params={"text": text, "speaker": self.voice_id}
            )

            if response.status_code != 200:
                print(f"❌ Error TTS (Audio Query): {response.status_code} - {response.text}")
                return

            query_data = response.json()
            query_data["outputSamplingRate"] = 24000  
            query_data["outputStereo"] = False        

            synthesis_response = self.session.post(
                f"{self.base_url}/synthesis",
                params={"speaker": self.voice_id},
                json=query_data
            )

            if synthesis_response.status_code != 200:
                print(f"❌ Error TTS (Synthesis): {synthesis_response.status_code} - {synthesis_response.text}")
                return

            temp_filename = "output.wav"

            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            with open(temp_filename, "wb") as f:
                f.write(synthesis_response.content)

            audio = AudioSegment.from_wav(temp_filename)
            play(audio)  
            os.remove(temp_filename)

            await asyncio.sleep(1.5)  

        except Exception as e:
            print(f"❌ Error TTS: {e}")

class SpeechRecognizer:
    def __init__(self, config):
        """Inisialisasi model Whisper AI."""
        self.config = config["whisper"]
        self.model_name = self.config["model"]
        self.device = "cuda" if torch.cuda.is_available() and self.config["device"] == "cuda" else "cpu"

        print(f"🔍 Memuat model Whisper: {self.model_name} di perangkat {self.device}")
        self.model = whisper.load_model(self.model_name, device=self.device)
        self.q = queue.Queue()
        self.device_index = self.get_microphone_device()

    def get_microphone_device(self):
        """Menampilkan daftar mikrofon yang tersedia dan memilih satu secara manual."""
        print("🎤 Daftar perangkat audio yang tersedia:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if "Microphone" in device["name"] or "CABLE Output" in device["name"]:
                print(f"✅ Menggunakan mikrofon: {device['name']} (Index {i})")
                return i
        return 0

    def callback(self, indata, frames, time, status):
        """Memproses audio input dari mikrofon dan memasukkan ke queue."""
        if status:
            print(f"⚠️ Mikrofon Error: {status}")
        self.q.put(np.array(indata, dtype=np.int16))  # ✅ FIX: Konversi buffer ke NumPy array

    def record_audio(self, duration=5):
        """Merekam audio selama beberapa detik dan menyimpannya dalam file sementara."""
        print("🎙️ Merekam audio...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_filename = temp_wav.name

        with wave.open(temp_filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)

            with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16', channels=1, callback=self.callback, device=self.device_index):
                for _ in range(int(16000 / 4000 * duration)):
                    data = self.q.get()
                    wf.writeframes(data)

        print("✅ Audio berhasil direkam.")
        return temp_filename

    def recognize(self):
        """Mendengarkan input suara dan mengonversinya ke teks menggunakan Whisper."""
        audio_file = self.record_audio()
        print("📝 Memproses audio dengan Whisper...")
        result = self.model.transcribe(audio_file)
        text = result["text"].strip()
        print(f"📝 Hasil STT: {text}")
        os.remove(audio_file)  
        return text


class MainApp:
    def __init__(self, config):
        """ Inisialisasi AI, STT, dan TTS """
        self.config = config
        self.ai = GoogleAIChat(config)  
        self.stt = SpeechRecognizer(config)  
        self.tts = TextToSpeech(config)  
        self.running = True  

    async def chat_mode(self):
        """ Mode Chat: AI merespons input teks pengguna secara terus-menerus """
        print("\n✉️ Mode Chat diaktifkan! Ketik pesan di bawah ini ('exit' untuk keluar).")

        while self.running:
            try:
                user_input = input("\nKamu (ketik pesan): ")

                if user_input.lower().strip() == "exit":
                    print("👋 Sampai jumpa!")
                    self.running = False
                    break  

                response = await self.ai.get_response(user_input)
                response_text = response.get("text", "Maaf, aku tidak bisa menjawab itu.")

                print(f"🤖 Ganyu: {response_text}")

                await self.tts.speak(response_text)  

            except Exception as e:
                print(f"❌ Error Chat Mode: {e}")

    async def voice_mode(self):
        """ Mode Voice: AI merespons suara pengguna secara terus-menerus """
        print("\n🎙️ Mode Voice diaktifkan! Bicara ke mikrofon.")

        while self.running:
            try:
                print("\n🎤 AI Waifu mendengarkan... Bicara sekarang!")
                user_input = self.stt.recognize()

                if not user_input:
                    print("⏳ Tidak ada suara yang terdeteksi, coba lagi...")
                    continue

                print(f"👂 Terdengar: {user_input}")

                response = await self.ai.get_response(user_input)
                response_text = response.get("text", "Maaf, aku tidak bisa menjawab itu.")

                print(f"🤖 Ganyu: {response_text}")

                await self.tts.speak(response_text)  

            except Exception as e:
                print(f"❌ Error Voice Mode: {e}")

    async def run(self):
        """ Memilih Mode Interaksi """
        print("\n🎤 Pilih Mode Interaksi:")
        print("1️⃣ Mode Chat (Ketik pesan)")
        print("2️⃣ Mode Voice (Gunakan suara)")

        choice = input("Masukkan pilihan (1/2): ").strip()
        if choice == "1":
            await self.chat_mode()
        elif choice == "2":
            await self.voice_mode()
        else:
            print("Pilihan tidak valid. Program berhenti.")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    app = MainApp(config)
    asyncio.run(app.run())
