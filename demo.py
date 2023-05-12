import tkinter as tk
import pyaudio
import tensorflow.keras as keras
import numpy as np
import librosa
import threading
# import warnings
# warnings.filterwarnings("ignore")

class Speech_Recognizer:
    def __init__(self):
        self.CHUNK_SIZE = 1024
        self.SAMPLE_RATE = 48000
        self.labels = ['speaker1', 'speaker2', 'speaker3', 'speaker4', 'speaker5']
        self.model = keras.models.load_model('weights/CNN.h5', compile=False)
        self.stop_recording = False
        self.t = threading.Thread(target=self.start)
        self.lock = threading.Lock()
        self.initGUI()
        
    def initGUI(self):
        self.root = tk.Tk()
        self.root.title('Speaker Recognition')

        self.label_speaker = tk.Label(self.root, text='Press "START" to begin')
        self.label_speaker.pack()

        self.button_start = tk.Button(self.root, text='START', command=self.t.start)
        self.button_start.pack()

        self.button_stop = tk.Button(self.root, text='STOP', command=self.stop)
        self.button_stop.pack()

    def center_window(self, width, height):
        screenwidth = self.root.winfo_screenwidth()
        screenheight = self.root.winfo_screenheight()
        size = f'{width}x{height}+{int((screenwidth - width) / 2)}+{int((screenheight - height) / 2)}'
        self.root.geometry(size)

    def extract_features(self, signal, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
        if len(signal) >= sample_rate:
            # ensure consistency of the length of the signal
            signal = signal[:sample_rate]
        # Convert the audio data to MFCC features
        MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.array(MFCCs.T.tolist())

    def start(self):
        self.button_start["state"] = "disable"
        self.stop_recording = False
        audio = pyaudio.PyAudio()
        while True:
            stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=self.SAMPLE_RATE, input=True, frames_per_buffer=self.CHUNK_SIZE)
            self.lock.acquire()
            if self.stop_recording:
                self.button_stop["text"] = "繼續"
                self.label_speaker['text'] = '暫停'
                self.lock.release()
                continue
            self.button_stop["text"] = '暫停'

            # Read audio data from the stream
            audio_data = stream.read(self.SAMPLE_RATE, exception_on_overflow=False)

            # Convert the audio data to a numpy array
            audio_samples = np.frombuffer(audio_data, dtype=np.float32)

            # Extract the MFCC features from the audio data
            features = self.extract_features(audio_samples, self.SAMPLE_RATE)

            # Reshape the features to match the input shape of the model
            features = features[np.newaxis, ..., np.newaxis]

            # Make a prediction using the model
            prediction = self.model.predict(features)
            predicted_label_index = np.argmax(prediction)
            predicted_label = self.labels[predicted_label_index]

            # Update the label text
            self.label_speaker['text'] = f'The speaker is: {predicted_label}, {prediction[0][predicted_label_index]*100:.3f}'
            self.label_speaker.pack()

            # Update the GUI
            self.root.update()
            self.lock.release()

    def stop(self):
        if self.stop_recording:
            self.stop_recording = False
            self.label_speaker['text'] = 'recording...'
        else:
            self.stop_recording = True
            print("stop")

    def run(self):
        self.center_window(400, 200)

        # Start GUI event loop
        self.root.mainloop()

if __name__ == '__main__':
    app = Speech_Recognizer()
    app.run()