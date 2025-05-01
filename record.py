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
        # 等待 buffer flush
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

    def get_segments(self, segment_seconds: float = 2.0):
        """
        将录好的 frames 拼成一个大 array，然后每 segment_seconds 秒切一段，
        返回一个 list，每个元素都是形状 (段长采样数, channels) 的 NumPy array。
        """
        # 拼成一个长 array
        data = np.concatenate(self._frames, axis=0)
        # 每段多少采样点
        seg_len = int(self.samplerate * segment_seconds)
        segments = [
            data[i : i + seg_len]
            for i in range(0, len(data), seg_len)
        ]
        return segments

    def save_mp4(self, wav_path: str, mp4_path: str = None):
        if mp4_path is None:
            mp4_path = os.path.splitext(wav_path)[0] + '.mp4'
        subprocess.run([
            'ffmpeg', '-y',
            '-i', wav_path,
            '-c:a', 'aac',
            '-b:a', '192k',
            mp4_path
        ], check=True)
        print(f"Saved MP4: {mp4_path}")
        return mp4_path


if __name__ == '__main__':
    rec = Recorder(samplerate=64000, channels=1)

    input("Press Enter to start recording…")
    rec.start()
    print("…Recording..., press Enter to stop")
    input()
    rec.stop()

    # 先保存 WAV
    wav = rec.save_wav("my_recording.wav")

    # 再按 2 秒一段切片
    segments = rec.get_segments(segment_seconds=2.0)
    print(f"共切出 {len(segments)} 段，每段 shape = {segments[0].shape}")

    # （可选）把每段保存成独立文件
    for idx, seg in enumerate(segments, 1):
        fname = f"segment_{idx:03d}.wav"
        sf.write(fname, seg, rec.samplerate, subtype=rec.subtype)
        print(f"Saved {fname}")
