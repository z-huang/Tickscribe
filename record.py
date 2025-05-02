import sounddevice as sd
import numpy as np
import time
import queue

class LiveRecorder:
    def __init__(self, sample_rate=16000, channels=1, block_duration=1.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = int(sample_rate * block_duration)
        self.queue = queue.Queue()
        self.running = False

        # 自動閾值參數
        self.noise_level = init_threshold     # 背景噪音能量估計
        self.sensitivity = sensitivity        # 靜音判定乘數
        self.adapt_rate = adapt_rate          # EMA 更新速率
        self.min_threshold = min_threshold    # 閾值下限，避免變得過低

    def is_silent(self, chunk: np.ndarray) -> bool:
        """
        用動態閾值判斷靜音，並在靜音時更新背景噪音估計
        """
        chunk = np.squeeze(chunk).astype(np.float32)
        rms = np.sqrt(np.mean(chunk ** 2))  # Root Mean Square (energy)
        # print(rms)
        if rms < threshold:
            print('[Debug] Silence', flush=True)
            return True
        else:
            return False
        

    def _callback(self, indata, frames, time_info, status):
        if self.running:
            if not self.is_silent(indata):
                # 非靜音才推進 queue
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
