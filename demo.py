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

# Define button to load model
def load_model():
    global model
    global label_speaker
    model = keras.models.load_model('weights\CNN.h5', compile=False)
    label_speaker['text'] = 'Model loaded successfully'

def extract_features(audio_data, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
    # 將音訊資料轉為 MFCC 特徵
    MFCCs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return MFCCs.T.tolist()
    
def start():
    global model
    global label_speaker
    global stop_recording
    CHUNK_SIZE = 1024
    SAMPLE_RATE = 48000

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open an input stream
    stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    # Define the labels for the speakers
    labels = ['speaker1', 'speaker2', 'speaker3', 'speaker4', 'speaker5']

    stop_recording = False
    while True:
        if stop_recording:
            break
        # Read audio data from the stream
        audio_data = stream.read(SAMPLE_RATE)

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

def stop():
    global stop_recording
    stop_recording = True

if __name__ == '__main__':
    # Initialize the GUI
    root = tk.Tk()
    root.title('Speaker Recognition')
    root.configure(bg="#FFFFFF")
    
    # Define label for speaker prediction
    label_speaker = tk.Label(root, text='Press "Start" to begin')

    # Define button to load model
    button_load = tk.Button(root, text='Load Model', command=load_model)

    # Define button to start prediction
    button_start = tk.Button(root, text='START', command=start)

    # Define button to stop prediction
    button_stop = tk.Button(root, text='STOP', command=stop)

    center_window(root, 400, 200)

    # Pack GUI elements
    label_speaker.pack()
    button_load.pack()
    button_start.pack()
    button_stop.pack()

    # Start GUI event loop
    root.mainloop()