import tkinter as tk
import pyaudio
import tensorflow.keras as keras
import numpy as np
import librosa

import warnings
warnings.filterwarnings("ignore")

def center_window(root, width, height):
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    size = f'{width}x{height}+{int((screenwidth - width) / 2)}+{int((screenheight - height) / 2)}'
    root.geometry(size)

def extract_features(audio_data, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
    # 將音訊資料轉為 MFCC 特徵
    MFCCs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return MFCCs.T.tolist()

def button_start():
    CHUNK_SIZE = 1024
    SAMPLE_RATE = 48000
    global label_speaker

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open an input stream
    stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    # Define the labels for the speakers
    labels = ['speaker1', 'speaker2', 'speaker3', 'speaker4', 'speaker5']

    # Load model
    model = keras.models.load_model('CNN.h5', compile=False)

    while True:
        # Read audio data from the stream
        audio_data = stream.read(4800*3)

        # Convert the audio data to a numpy array
        audio_samples = np.frombuffer(audio_data, dtype=np.float32)

        # Extract the MFCC features from the audio data
        features = extract_features(audio_samples, SAMPLE_RATE)

        # Reshape the features to match the input shape of the model
        features = np.expand_dims(features, axis=0)
        
        # Make a prediction using the model
        prediction = model.predict(features)
        predicted_label_index = np.argmax(prediction)
        predicted_label = labels[predicted_label_index]

        # Update the label text
        label_speaker['text'] = 'The speaker is: ' + predicted_label

        # Update the GUI
        label_speaker.pack()
        root.update()

# Initialize the GUI
root = tk.Tk()
root.title('GUI')
center_window(root, 600, 400)

# Create a label to display the speaker's name
label_speaker = tk.Label(root, text='')
label_speaker.pack()

button = tk.Button(root, text='START', command=button_start)
button.pack()

root.mainloop()