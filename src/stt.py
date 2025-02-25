import whisper
import sounddevice as sd
import numpy as np
import logging
import soundfile as sf
import torch  # Untuk cek apakah GPU digunakan

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Cek apakah GPU tersedia untuk Whisper
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"🚀 Menggunakan device: {DEVICE}")

# Load model Whisper (gunakan "small" untuk lebih cepat, atau "medium" untuk lebih akurat)
MODEL_SIZE = "small"  # Bisa diganti: "tiny", "base", "small", "medium", "large"
model = whisper.load_model(MODEL_SIZE, device=DEVICE)

def record_audio(duration=5, samplerate=16000):
    """Merekam audio dari mikrofon selama beberapa detik."""
    logging.info(f"🎤 Merekam suara selama {duration} detik...")
    try:
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
        sd.wait()
        return np.squeeze(audio_data), samplerate
    except Exception as e:
        logging.error(f"❌ Error saat merekam suara: {e}")
        return None, None

def transcribe_audio():
    """Merekam audio dan mengubahnya menjadi teks menggunakan Whisper."""
    try:
        audio, samplerate = record_audio()
        if audio is None:
            return None
        
        logging.info("📝 Mengonversi suara ke teks...")
        
        # Simpan audio ke file WAV
        audio_path = "temp_audio.wav"
        sf.write(audio_path, audio, samplerate)

        # Transkripsi dengan Whisper
        result = model.transcribe(audio_path)
        text = result["text"].strip()
        
        logging.info(f"✅ Teks yang dikenali: {text}")
        return text

    except Exception as e:
        logging.error(f"❌ Error saat mendengar suara: {e}")
        return None

# Uji coba
if __name__ == "__main__":
    print("🎙️ Mulai berbicara, aku akan mendengar...")
    text = transcribe_audio()
    if text:
        print(f"🗣️ Kamu mengatakan: {text}")
    else:
        print("❌ Tidak dapat mengenali suara.")
