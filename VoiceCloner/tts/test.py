# save as google_stt.py
import subprocess
import speech_recognition as sr
import os

# Convert MP3 to WAV using ANY available method
def convert_to_wav(mp3_file, wav_file):
    try:
        # Try ffmpeg if installed
        subprocess.run(['ffmpeg', '-i', mp3_file, '-ar', '16000', '-ac', '1', wav_file], check=True)
        return True
    except:
        try:
            # Try pydub as backup
            from pydub import AudioSegment
            AudioSegment.from_mp3(mp3_file).export(wav_file, format="wav")
            return True
        except:
            return False

# If conversion fails, use microphone instead
if not convert_to_wav("output.mp3", "temp.wav"):
    print("Using microphone instead...")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now:")
        audio = r.listen(source)
        text = r.recognize_google(audio)
        print("TEXT:", text)
else:
    # Transcribe the WAV file
    r = sr.Recognizer()
    with sr.AudioFile("temp.wav") as source:
        audio = r.record(source)
        text = r.recognize_google(audio)
        print("TEXT:", text)
    
    # Clean up
    os.remove("temp.wav")