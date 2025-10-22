# -*- coding: utf-8 -*-
"""
voice_control.py — Auto mic discovery + sentence-level aggregation (stitching)
faster-whisper (GPU/CPU) → command parsing → on_command callback.

Key features
------------
- Auto-pick an input device & supported sample rate (fallbacks).
- Preload model before opening mic to avoid initial overflows.
- Tunable PortAudio buffers: latency_s, block_ms.
- Sentence-level aggregation:
  * stitch_chunks=True → accumulate partials, finalize on silence/timeout.
  * speech_end_silence_ms, max_phrase_seconds.
- initial_prompt_once & ignore_prompt_echo to avoid "热词回声".
- No external audio writers; use stdlib `wave`.

Dependencies
------------
pip install faster-whisper sounddevice numpy
"""
from __future__ import annotations

import re
import time
import queue
import threading
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple

import numpy as np
import sounddevice as sd
import wave

try:
    from faster_whisper import WhisperModel  # heavy import
except Exception as _e:
    WhisperModel = None  # type: ignore


# ---------------------------
# Configuration
# ---------------------------

@dataclass
class VoiceConfig:
    # Preferred audio capture settings (auto-fallback will try alternatives)
    samplerate: int = 16_000                 # preferred; will auto-fallback if unsupported
    channels: int = 1                        # preferred; will auto-fallback to {1,2}
    device_index: Optional[int] = None       # preferred device; None = auto pick an input device
    chunk_seconds: float = 1.3               # ASR window size（低延时建议 1.2~1.4）
    overlap_seconds: float = 0.2             # tail kept to reduce word cut-off
    min_dbfs: float = -35.0                  # chunks quieter than this are ignored（提高可减回声）

    # PortAudio buffer tuning
    latency_s: float = 0.35                  # requested I/O latency (sec); higher = fewer overflows
    block_ms: float = 120.0                  # callback block size in ms (e.g., 100~150ms)

    # Whisper/faster-whisper
    whisper_model_name: str = "small"        # tiny/base/small/medium/large-v3/large-v3-turbo/distil-large-v3
    language: str = "zh"                     # "zh" or ""(auto)
    initial_prompt: str = ""                 # e.g. "升高 降低 前进 后退 停止 Z轴 底盘 夹爪 归零"
    beam_size: int = 1
    condition_on_previous_text: bool = False
    use_vad: bool = True
    vad_min_silence_ms: int = 600            # 更稳的静音阈值（命令场景建议 500~700）

    # Prompt usage & echo control
    initial_prompt_once: bool = True         # 只在首次有效语音使用 initial_prompt
    ignore_prompt_echo: bool = True          # 文本几乎等于热词清单则忽略

    # Sentence-level aggregation
    stitch_chunks: bool = True               # 先拼句，后触发（强烈推荐开启）
    speech_end_silence_ms: int = 2000         # 连续静音多久算一句话结束
    max_phrase_seconds: float = 5.0          # 句子最长超时（兜底）

    # Device / precision
    device: str = "cpu"                      # "cpu" or "cuda"
    compute_type: str = "int8"               # CPU: int8; GPU: float16 or int8_float16

    # Parser / dispatch
    command_cooldown_s: float = 1.0          # min seconds between dispatches
    print_partial: bool = True               # print recognized text to console

    # Model loading
    preload_model: bool = True               # preload model before opening mic to avoid overflow

    # Optional keyword gating (if non-empty, only texts containing any keyword are considered)
    keywords: List[str] = field(default_factory=list)


# ---------------------------
# Utility
# ---------------------------

def dbfs(x: np.ndarray) -> float:
    """Return dBFS of a float32 waveform in [-1, 1]."""
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * np.log10(min(max(rms, 1e-9), 1.0))


def list_input_devices() -> List[Tuple[int, dict]]:
    """Return list of (index, info) for input-capable devices."""
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            devices.append((i, d))
    return devices


# ---------------------------
# Command Parser
# ---------------------------

class CommandParser:
    """Regex rules → canonical command string."""

    def __init__(self) -> None:
        # Order matters; first match wins.
        self.rules: List[tuple[re.Pattern, str]] = [
            (re.compile(r"(紧急|立刻)?停(止)?|别动|停止|stop", re.I), "stop"),
            (re.compile(r"(升高|上升|抬高|升一点|向上|up)", re.I), "lift_up"),
            (re.compile(r"(降低|下降|放下|降一点|向下|down)", re.I), "lift_down"),
            (re.compile(r"(?:向|像|相)右(旋转|转)|turn right", re.I), "turn_right"),
            (re.compile(r"(?:向|像|相)左(旋转|转)|turn left",  re.I), "turn_left"),
            (re.compile(r"(前进|往前|向前|go|forward)", re.I), "forward"),
            (re.compile(r"(后退|往后|向后|back|backward)", re.I), "backward"),
        ]

    def parse(self, text: str) -> Optional[str]:
        for pat, cmd in self.rules:
            if pat.search(text):
                return cmd
        return None


# ---------------------------
# Engine
# ---------------------------

class VoiceEngine:
    """
    Capture from microphone → faster-whisper transcribe → (stitch) → parse → on_command(cmd)
    Auto-discovers a working input device & sample rate if not provided.
    """

    def __init__(self, cfg: VoiceConfig, on_command: Optional[Callable[[str], None]] = None):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed or failed to import.")
        self.cfg = cfg
        self.on_command = on_command or (lambda cmd: print(f"→ 执行动作: {cmd}"))
        self.parser = CommandParser()

        # Runtime state
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_evt = threading.Event()
        self._stream: Optional[sd.InputStream] = None
        self._asr_thread: Optional[threading.Thread] = None
        self._model: Optional[WhisperModel] = None  # type: ignore
        self._last_dispatch_ts = 0.0
        self._last_overflow_log_ts = 0.0

        # Prompt usage & stitching state
        self._prompt_used = False
        self._phrase_text = ""
        self._speech_active = False
        self._last_voice_ts = 0.0
        self._phrase_start_ts = 0.0

    # ---- Public API ----

    def start(self) -> None:
        """Start microphone stream and ASR loop thread."""
        if self._asr_thread and self._asr_thread.is_alive():
            return
        self._stop_evt.clear()

        # Preload model first to avoid CPU stall after starting mic
        if self.cfg.preload_model:
            self._lazy_model()

        self._open_stream()
        self._asr_thread = threading.Thread(target=self._asr_loop, name="asr-loop", daemon=True)
        self._asr_thread.start()
        print("🎤 语音控制已启动（Ctrl+C 停止）")

    def stop(self) -> None:
        """Stop everything."""
        self._stop_evt.set()
        try:
            if self._asr_thread:
                self._asr_thread.join(timeout=2.0)
        finally:
            self._close_stream()
        print("🛑 已停止语音控制")

    # ---- Internals ----

    def _open_stream(self) -> None:
        """Find a working input device & sample rate; open the stream."""

        def _cb(indata, frames, t, status):
            # Throttle status prints to avoid spam
            if status:
                now = time.time()
                if now - self._last_overflow_log_ts > 2.0:
                    print(status)
                    self._last_overflow_log_ts = now
            # indata shape: (frames, channels), float32 in [-1, +1]
            mono = np.mean(indata, axis=1).astype(np.float32)
            self._audio_q.put(mono)

        # Build device candidates
        candidates_dev: List[int] = []
        tried_summary: List[str] = []

        if self.cfg.device_index is not None:
            candidates_dev = [self.cfg.device_index]
        else:
            # Prefer PortAudio default input
            try:
                default_in = sd.default.device[0]
                if isinstance(default_in, int) and default_in >= 0:
                    candidates_dev.append(default_in)
            except Exception:
                pass
            # Append all input-capable devices
            for i, _info in list_input_devices():
                if i not in candidates_dev:
                    candidates_dev.append(i)

        if not candidates_dev:
            raise RuntimeError("No input-capable audio devices found.")

        last_err = None
        for dev in candidates_dev:
            try:
                d_info = sd.query_devices(dev, 'input')
            except Exception as e:
                last_err = e
                continue

            # Build samplerate candidates
            sr_cand: List[int] = []
            if self.cfg.samplerate:
                sr_cand.append(int(self.cfg.samplerate))
            dflt_sr = int(d_info.get("default_samplerate") or 0)
            if dflt_sr and dflt_sr not in sr_cand:
                sr_cand.append(dflt_sr)
            for sr in [48000, 44100, 32000, 16000, 22050]:
                if sr not in sr_cand:
                    sr_cand.append(sr)

            # Build channel candidates
            ch_cand: List[int] = []
            if self.cfg.channels:
                ch_cand.append(int(self.cfg.channels))
            if 1 not in ch_cand:
                ch_cand.append(1)
            if int(d_info.get("max_input_channels", 0)) >= 2 and 2 not in ch_cand:
                ch_cand.append(2)

            # Compute blocksize frames
            def blocksize_for(sr: int) -> int:
                return max(128, int(sr * (self.cfg.block_ms / 1000.0)))

            for sr in sr_cand:
                for ch in ch_cand:
                    try:
                        stream = sd.InputStream(
                            samplerate=sr,
                            channels=ch,
                            dtype="float32",
                            device=dev,
                            callback=_cb,
                            blocksize=blocksize_for(sr),
                            latency=self.cfg.latency_s,   # increase buffers inside PortAudio
                        )
                        stream.start()
                        # Success: adopt the working settings
                        self._stream = stream
                        if sr != self.cfg.samplerate:
                            print(f"⚠️  采样率 {self.cfg.samplerate} 不可用，已改用 {sr}")
                            self.cfg.samplerate = sr
                        if ch != self.cfg.channels:
                            print(f"⚠️  通道数 {self.cfg.channels} 不可用，已改用 {ch}")
                            self.cfg.channels = ch
                        if dev != self.cfg.device_index:
                            name = d_info.get('name', str(dev))
                            print(f"✅  使用设备 #{dev}: {name}")
                            self.cfg.device_index = dev
                        print(f"ℹ️  采样率={sr}, 通道={ch}, block={blocksize_for(sr)} 帧, latency={self.cfg.latency_s}s")
                        return
                    except sd.PortAudioError as e:
                        last_err = e
                        tried_summary.append(f"dev#{dev}({d_info.get('name','?')}), ch={ch}, sr={sr} -> {e}")
                        continue

        # If we get here, no combination worked
        lines = "\n  ".join(tried_summary[-10:])  # show last 10 attempts
        raise RuntimeError(f"Failed to open audio input stream. Recent attempts:\n  {lines}") from last_err

    def _close_stream(self) -> None:
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None

    def _lazy_model(self) -> WhisperModel:
        if self._model is None:
            print(f"⏳ 加载模型: {self.cfg.whisper_model_name} ({self.cfg.device}, {self.cfg.compute_type}) ...")
            self._model = WhisperModel(
                self.cfg.whisper_model_name,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
            )
            print("✅ 模型就绪")
        return self._model

    def _asr_loop(self) -> None:
        buf = np.zeros(0, dtype=np.float32)
        sr = self.cfg.samplerate
        chunk = int(sr * self.cfg.chunk_seconds)
        tail = int(sr * self.cfg.overlap_seconds)

        model = self._lazy_model()

        while not self._stop_evt.is_set():
            try:
                chunk_in = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                # 长静音期间也需要检查句末
                self._maybe_finalize_on_silence()
                continue

            buf = np.concatenate([buf, chunk_in])
            if buf.size < chunk:
                # 即便不够一个窗口，也可能达到句末
                #self._maybe_finalize_on_silence()
                continue

            clip = buf[:chunk]
            # Slice next buffer keeping a small tail
            buf = buf[chunk - tail:]

            now = time.time()
            # Loudness估计（用于聚合与门限）
            level = dbfs(clip)

            # 静音：如果正在说话，检查是否到达句末
            if level < self.cfg.min_dbfs:
                self._maybe_finalize_on_silence(now=now)
                continue
            else:
                # 只要能量达标，就认为在说话，刷新时间戳（防止等 ASR 期间早收）
                if not self._speech_active:
                    self._phrase_start_ts = now
                self._speech_active = True
                self._last_voice_ts = now

            # Optional keyword gating（快探一次，非必须）
            if self.cfg.keywords:
                probe = "".join(self._quick_transcribe(model, clip))
                if not any(k in probe for k in self.cfg.keywords):
                    self._maybe_finalize_on_silence(now=now)
                    continue

            # 写临时 WAV（16-bit PCM）
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                x = np.clip(clip, -1.0, 1.0)
                x_i16 = (x * 32767.0).astype(np.int16)
                with wave.open(f.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(x_i16.tobytes())

                # 仅在未用过且启用“只用一次”时传入 initial_prompt
                hint = (self.cfg.initial_prompt or None)
                if self.cfg.initial_prompt_once and self._prompt_used:
                    hint = None

                segments, _info = model.transcribe(
                    f.name,
                    language=(self.cfg.language or None),
                    initial_prompt=hint,
                    beam_size=self.cfg.beam_size,
                    condition_on_previous_text=self.cfg.condition_on_previous_text,
                    vad_filter=self.cfg.use_vad,
                    vad_parameters=dict(min_silence_duration_ms=int(self.cfg.vad_min_silence_ms)),
                )

            text = "".join(seg.text for seg in segments).strip()
            if not text:
                self._maybe_finalize_on_silence(now=now)
                continue

            # 标记 initial_prompt 已使用
            if hint:
                self._prompt_used = True

            # 聚合：进入“说话中”状态并累加文本
            self._speech_active = True
            self._last_voice_ts = now
            if not self._phrase_text:
                self._phrase_start_ts = now

            if self.cfg.stitch_chunks:
                self._phrase_text += ((" " if self._phrase_text else "") + text)
                # 兜底：句子过长直接提交
                if (now - self._phrase_start_ts) >= self.cfg.max_phrase_seconds:
                    self._finalize_phrase(reason="timeout")
                    continue
                # 打印中间结果（不触发命令）
                if self.cfg.print_partial:
                    print(time.strftime("[%H:%M:%S] "), "实时识别:",self._phrase_text, f"({level:.1f} dBFS)")
                continue

            # 不聚合：旧行为（逐段触发）
            if self.cfg.print_partial:
                print(time.strftime("[%H:%M:%S] "), text, f"({level:.1f} dBFS)")
            self._trigger_if_match(text, now)

    # ---- helpers ----

    def _maybe_finalize_on_silence(self, now: Optional[float] = None) -> None:
        """If we are in speech and silence lasts long enough, finalize current phrase."""
        if not self.cfg.stitch_chunks or not self._speech_active:
            return
        now = now or time.time()
        if (now - self._last_voice_ts) * 1000 >= self.cfg.speech_end_silence_ms:
            self._finalize_phrase(reason="silence")

    def _finalize_phrase(self, reason: str) -> None:
        """在句末提交：解析整句并触发一次。"""
        text = self._phrase_text.strip()
        self._phrase_text = ""
        the_phrase_started = self._phrase_start_ts
        self._phrase_start_ts = 0.0
        self._speech_active = False
        if not text:
            return

        # 提示词回声过滤
        if self.cfg.ignore_prompt_echo and self.cfg.initial_prompt:
            tw = set(text.split())
            pw = set(self.cfg.initial_prompt.split())
            overlap = len(tw & pw) / max(1, len(tw))
            if overlap >= 0.8:
                if self.cfg.print_partial:
                    print(f"忽略提示词回声（{reason}）:", text)
                return

        if self.cfg.print_partial:
            dur = time.time() - (the_phrase_started or time.time())
            print(f"句末（{reason}，{dur:.2f}s）:", text)

        now = time.time()
        self._trigger_if_match(text, now)

    def _trigger_if_match(self, text: str, now: float) -> None:
        cmd = self.parser.parse(text)
        if cmd and (now - self._last_dispatch_ts) >= self.cfg.command_cooldown_s:
            self._last_dispatch_ts = now
            try:
                self.on_command(cmd)
            except Exception as e:
                print("on_command 处理异常：", e)

    def _quick_transcribe(self, model: WhisperModel, clip: np.ndarray) -> List[str]:
        """Fast probe transcription for keyword gating. Uses small beam and no VAD."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            x = np.clip(clip, -1.0, 1.0)
            x_i16 = (x * 32767.0).astype(np.int16)
            with wave.open(f.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.cfg.samplerate)
                wf.writeframes(x_i16.tobytes())
            segs, _ = model.transcribe(
                f.name,
                language=(self.cfg.language or None),
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
        return [s.text for s in segs]


# ---------------------------
# Demo
# ---------------------------

if __name__ == "__main__":
    def demo_on_command(cmd: str) -> None:
        print(f"→ 执行动作: {cmd}")

    cfg = VoiceConfig(
        # whisper_model_name="small",
        # language="zh",
        initial_prompt = (
            "指令词：上升、下降、前进、后退、停止、向左转、向右转；"
            "单位：毫米、厘米、度；数字：0、1、2、3、5、10、20、30、90；"
        )
        # device="cpu",             # "cuda" for GPU
        # compute_type="int8",      # "float16" (GPU) or "int8_float16"
        # samplerate=16000,         # preferred; auto-fallback will try others
        # channels=1,               # preferred
        # device_index=None,        # auto-pick device
        # chunk_seconds=1.3,
        # overlap_seconds=0.3,
        # latency_s=0.35,
        # block_ms=120.0,
        # min_dbfs=-30.0,
        # stitch_chunks=True,
        # speech_end_silence_ms=600,
        # max_phrase_seconds=3.0,
        # initial_prompt_once=True,
        # ignore_prompt_echo=True,
    )
    engine = VoiceEngine(cfg, on_command=demo_on_command)

    try:
        engine.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()