import pyaudio
import numpy as np
import scipy.fftpack
import time
import sys

# Audio Parameters
CHUNK = 2048 # samples per chunk
FORMAT = pyaudio.paInt16 # 16-bit resolution
CHANNELS = 1 # Mono
RATE = 44100 # Commonly used sampling rate (44.1 hz)

# Parameters for printing frequency ion place
MOVE_CURSOR_UP = "\033[1A"
ERASE = "\x1b[2x"

# Initialize PyAudio object
p = pyaudio.PyAudio()

# Open stream for capturing audio
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK,
    input_device_index=None,
    output_device_index=None,
)

def get_dominat_frequency(audio_data):
    # Apply Fast Fourier Transformation to the audio data
    fft_data = np.abs(scipy.fftpack.fft(audio_data))
    frequencies = scipy.fftpack.fftfreq(len(fft_data), 1.0/RATE)

    # Consider only the positive frequencies for efficiency
    positive_frequencies = frequencies[:len(frequencies)//2]
    positive_fft_data = fft_data[:len(fft_data)//2]

    # Find the peak frequencies
    return positive_frequencies[np.argmax(positive_fft_data)]


# Program loop
try:
    while True:
        # Read chunk of data from microphone
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Get dominant frequency in the chunk
        dominat_frequency = get_dominat_frequency(audio_data)

        # Print frequency in place
        print(f'Frequency: {dominat_frequency:.2f} Hz')
        print(MOVE_CURSOR_UP + ERASE, end="")

except KeyboardInterrupt:
    print("\nStopping...")

stream.stop_stream()
stream.close()

p.terminate()
