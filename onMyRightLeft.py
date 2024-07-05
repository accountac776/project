from pydub import AudioSegment
import wave
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr
import assemblyai as aai
from gtts import gTTS
aai.settings.api_key = "9c0ad83e80bc4a199acfec799c5891b3"
import os

# mytext = 'There are' + objects + "on your" + zone
mytext = "This is a certified hood classic."

language = 'en'
myobj = gTTS(text=mytext, lang=language, slow=False)


myobj.save("messageOfComputer.mp3")
os.system("afplay messageOfComputer.mp3")

def recordVoice():
    seconds = 5 #şimdilik böyle dedim
    #audioInformation
    FRAMES_PER_BUFFER = 3200
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()
    stream=p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
    )
    frames = []
    print("Start speaking!")
    for i in range(0, int(RATE/FRAMES_PER_BUFFER*seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
    stream.stop_stream()
    print("Time is over, processing order.")
    stream.close()
    p.terminate()
    obj = wave.open("messageOfUser.wav", "wb")
    obj.setnchannels(CHANNELS)
    obj.setsampwidth(p.get_sample_size(FORMAT))
    obj.setframerate(RATE)
    obj.writeframes(b"".join(frames))
    obj.close()
recordVoice()

def theTranscriber():
    sr.__version__
    r = sr.Recognizer()
    randomNoise = sr.AudioFile("messageOfUser.wav")
    with randomNoise as source:
        audio = r.record(source)
    type(audio)
    print(r.recognize_google(audio))
theTranscriber()


def check_direction(word):
    if "left" in word:
        print("left")
    elif "right" in word:
        print("right")
    else:
        print("Neither left nor right in the string.")