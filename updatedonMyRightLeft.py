import tkinter as tk
import pyaudio
import wave
import os
from gtts import gTTS
import speech_recognition as sr
from main import YOLOv8Live

class VoiceRecorder:
    def __init__(self, yolo_instance):
        self.yolo_instance = yolo_instance
        self.FRAMES_PER_BUFFER = 3200
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

    def record_voice(self):
        seconds = 5
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.FRAMES_PER_BUFFER
        )
        frames = []
        print("Recording is beginning!")
        for i in range(0, int(self.RATE / self.FRAMES_PER_BUFFER * seconds)):
            data = stream.read(self.FRAMES_PER_BUFFER)
            frames.append(data)
        stream.stop_stream()
        print("Time is over, processing order.")
        stream.close()
        p.terminate()

        obj = wave.open("messageOfUser.wav", "wb")
        obj.setnchannels(self.CHANNELS)
        obj.setsampwidth(p.get_sample_size(self.FORMAT))
        obj.setframerate(self.RATE)
        obj.writeframes(b"".join(frames))
        obj.close()

    def create_button(self):
        window = tk.Tk()
        window.title("Voice Recorder")

        button = tk.Button(window, text="Record Voice", command=self.on_button_click)
        button.pack(pady=20)

        window.mainloop()

    def on_button_click(self):
        self.record_voice()
        self.yolo_instance.detection()

    def run(self):
        self.create_button()
