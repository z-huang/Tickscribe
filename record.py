import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import subprocess
import os
import time


class Recorder:
    def __init__(self, samplerate=64000, channels=1, subtype='PCM_16'):
        self.samplerate = samplerate
        self.channels = channels
        self.subtype = subtype
        self._running = False
        self._frames = []

    def _callback(self, indata, frames, time_info, status):
        if self._running:
            # 複製一份，否則後面會跑掉
            self._frames.append(indata.copy())

    def start(self):
        """Start recording"""
        self._frames = []
        self._running = True
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        """Stop recording"""
        self._running = False
        # 確保抓完最後一批資料
        time.sleep(0.1)
        self.stream.stop()
        self.stream.close()

    def save_wav(self, filename: str):
        """Store audio as WAV file"""
        if not filename.lower().endswith('.wav'):
            filename += '.wav'
        data = np.concatenate(self._frames, axis=0)
        sf.write(filename, data, self.samplerate, subtype=self.subtype)
        print(f"Saved WAV: {filename}")
        return filename

if __name__ == '__main__':
    rec = Recorder(samplerate=64000, channels=1)

    input("Press Enter to start recording…")
    rec.start()
    print("…Recording..., press Enter to stop")
    input()
    rec.stop()

    wav = rec.save_wav("my_recording.wav")
