import sounddevice as sd
import soundfile as sf
import numpy as np
import subprocess
import os
import time
import queue

class LiveRecorder:
    def __init__(self, sample_rate=16000, channels=1, block_duration=2.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = int(sample_rate * block_duration)
        self.queue = queue.Queue()
        self.running = False

    def is_silent(self, chunk, threshold=0.01):
        """
        Returns True if the audio chunk is considered silent.
        :param chunk: np.ndarray audio data, shape = (n, 1)
        :param threshold: float - RMS energy threshold for silence
        """
        chunk = np.squeeze(chunk).astype(np.float32)
        rms = np.sqrt(np.mean(chunk ** 2))  # Root Mean Square (energy)
        if rms < threshold:
            return True
        else:
            print('Silence', flush=True)
            return False
        

    def _callback(self, indata, frames, time_info, status):
        if self.running:
            if not self.is_silent(indata):
                self.queue.put(indata.copy())

    def start(self):
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.stream.stop()
        self.stream.close()
