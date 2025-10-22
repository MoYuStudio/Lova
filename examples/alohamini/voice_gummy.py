#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_gummy_fuzzy.py — 容错更强的语音→动作：
- “保持N秒”更宽松：只要句子里既出现动作意图（forward/back/turn left/...）又出现时长（one second / 1 sec / 1s），就执行保持；
- 兼容标点/顺序（如 "Forward, for one second." / "one second turn left"）。
- 其它改动沿用：固定热词、统一数字归一化、去掉裸 left/right、旋转优先。
"""
from __future__ import annotations
import os
import re
import time
import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import sys
import subprocess
import shlex
from pathlib import Path

import numpy as np
import sounddevice as sd

# ---------- 可选的高质量重采样 ----------
try:
    from scipy.signal import resample_poly  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------- DashScope (Gummy) ----------
import dashscope
from dashscope.audio.asr import (
    TranslationRecognizerChat,
    TranslationRecognizerCallback,
    TranscriptionResult,
    TranslationResult,
)

# 可用则支持热词，否则自动降级
_VOCAB_AVAILABLE = True
try:
    from dashscope.audio.asr import VocabularyService
except Exception:
    _VOCAB_AVAILABLE = False
    VocabularyService = None  # type: ignore

# ===================== replay相关 =====================
def _project_root() -> Path:
    """自动定位到项目根目录"""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "examples" / "alohamini").exists():
            return p
    return here.parent

def run_replay(dataset: str, episode: int):
    """在当前conda环境异步执行replay_bi.py"""
    root = _project_root()
    py = sys.executable  # 当前conda环境的python路径
    cmd = [
        py,
        "examples/alohamini/replay_bi.py",
        "--dataset", dataset,
        "--episode", str(episode),
    ]
    print(f"[VOICE-REPLAY] Launching: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.Popen(cmd, cwd=root)

# ===================== 固定热词常量 =====================
HOTWORDS_CONST: List[str] = [
    # 中文动作与单位
    "上升","下降","前进","后退","左移","右移","左转","右转","停止",
    "毫米","厘米","米","秒","秒钟","锤他",
    # 英文同义词/单位/时长
    "up","down","forward","back",
    "turn left","turn right","rotate left","rotate right",
    "move left","move right","strafe left","strafe right",
    "millimeter","millimeters","centimeter","centimeters","meter","meters",
    "second","seconds","sec","s","for",
    # 常见数字词
    "zero","oh","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
    "seventeen","eighteen","nineteen","twenty","thirty","forty","fifty",
    "sixty","seventy","eighty","ninety","hundred","half","quarter",
]
VOCAB_PREFIX_CONST = "gummyam"

# ===================== 工具函数 =====================

def dbfs(x: np.ndarray) -> float:
    eps = 1e-12
    rms = max(eps, float(np.sqrt(np.mean(np.square(x.astype(np.float64))))))
    return 20.0 * math.log10(rms + eps)


def float32_to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def resample_to_16k(x: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == 16000 or len(x) == 0:
        return x.astype(np.float32, copy=False)
    if _HAS_SCIPY:
        from math import gcd
        g = gcd(src_sr, 16000)
        up, down = 16000 // g, src_sr // g
        y = resample_poly(x.astype(np.float32, copy=False), up, down)
        return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)
    new_len = int(round(len(x) * (16000.0 / float(src_sr))))
    if new_len <= 1:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    xnew = np.linspace(0.0, 1.0, num=new_len, endpoint=False, dtype=np.float64)
    y = np.interp(xnew, xp, x.astype(np.float64))
    return y.astype(np.float32, copy=False)


# ===================== 配置 =====================

@dataclass
class VoiceConfig:
    # 本地音频
    target_sr: int = 16000
    channels: int = 1
    chunk_seconds: float = 0.05
    overlap_seconds: float = 0.01
    frame_bytes: int = 3200

    # 能量门限（绝对 + 相对）
    min_dbfs: float = -30.0
    rel_db_margin_db: float = 7.0
    env_track_alpha: float = 0.9

    # 分句
    speech_end_silence_ms: int = 1000
    max_phrase_seconds: float = 15.0

    # Gummy
    model: str = "gummy-chat-v1"
    gummy_max_end_silence_ms: int = 1200
    print_partial: bool = True

    # 热词（默认启用固定常量）
    vocabulary_id: Optional[str] = None
    vocabulary_prefix: Optional[str] = VOCAB_PREFIX_CONST
    hotwords: Optional[List[str]] = field(default_factory=lambda: HOTWORDS_CONST.copy())

    # 输出
    emit_text_cmd: bool = True
    verbose_vol: bool = True

    # 速度标定
    xy_speed_cmd: float = 0.20
    theta_speed_cmd: float = 500.0


# ===================== 热词服务 =====================

class VocabularyManager:
    """热词表管理器：自动查找/创建/更新/清理"""
    def __init__(self, target_model: str, prefix: str):
        if not _VOCAB_AVAILABLE:
            raise RuntimeError("当前 dashscope 版本不支持 VocabularyService")
        self.svc = VocabularyService()
        self.target_model = target_model
        self.prefix = prefix
        self.vocabulary_id: Optional[str] = None
    
    def _clear_all_vocabularies(self):
        """清空所有热词表"""
        print("🧹 开始清空所有热词表...")
        try:
            vocab_list = self.svc.list_vocabularies()
            for vocab in vocab_list:
                try:
                    vid = vocab.get('vocabulary_id') or vocab.get('id')
                    if vid:
                        self.svc.delete_vocabulary(vid)
                        print(f"  ✓ 已删除热词表: {vid}")
                except Exception as e:
                    print(f"  ✗ 删除失败: {e}")
            print(f"🧹 清空完成，共删除 {len(vocab_list)} 个热词表")
        except Exception as e:
            print(f"⚠️ 清空热词表时出错: {e}")
    
    def _create_new_vocabulary(self, vocab: List[dict]) -> str:
        """创建新的热词表"""
        try:
            # 确保prefix安全
            safe_prefix = "".join(ch for ch in self.prefix.lower() if ch.isalnum())[:9] or "v1"
            
            res = self.svc.create_vocabulary(
                target_model=self.target_model,
                prefix=safe_prefix,
                vocabulary=vocab
            )
            
            # 提取vocabulary_id
            if isinstance(res, dict):
                vid = res.get("vocabulary_id") or res.get("id") or res.get("output", {}).get("vocabulary_id")
            else:
                vid = str(res)
            
            if vid:
                print(f"✓ 创建新热词表成功: {vid}")
                return vid
            else:
                raise RuntimeError("创建热词表返回了空ID")
                
        except Exception as e:
            error_msg = str(e)
            print(f"✗ 创建热词表失败: {error_msg}")
            
            # 检查是否是429超额错误
            if "429" in error_msg or "quota" in error_msg.lower() or "limit" in error_msg.lower():
                print("⚠️ 检测到配额超限，尝试清空所有热词表后重新创建...")
                self._clear_all_vocabularies()
                
                # 重新尝试创建
                safe_prefix = "".join(ch for ch in self.prefix.lower() if ch.isalnum())[:9] or "v1"
                res = self.svc.create_vocabulary(
                    target_model=self.target_model,
                    prefix=safe_prefix,
                    vocabulary=vocab
                )
                
                if isinstance(res, dict):
                    vid = res.get("vocabulary_id") or res.get("id") or res.get("output", {}).get("vocabulary_id")
                else:
                    vid = str(res)
                
                if vid:
                    print(f"✓ 清空后重新创建成功: {vid}")
                    return vid
                else:
                    raise RuntimeError("清空后重新创建失败")
            else:
                raise
    
    def _find_existing_vocabulary(self) -> Optional[str]:
        """查找已存在的相同prefix的热词表"""
        try:
            vocab_list = self.svc.list_vocabularies(prefix=self.prefix, page_index=0, page_size=10)
            if isinstance(vocab_list, list) and len(vocab_list) > 0:
                for item in vocab_list:
                    status = item.get("status") or item.get("state") or "OK"
                    vid = item.get("vocabulary_id") or item.get("id")
                    if status and status.upper() == "OK" and vid:
                        return vid
        except Exception as e:
            print(f"⚠️ 查找现有热词表时出错: {e}")
        return None
    
    def _hotwords_to_vocab_format(self, hotwords: List[str]) -> List[dict]:
        """将热词列表转换为API需要的格式"""
        vocab = []
        for word in hotwords:
            if not isinstance(word, str) or not word.strip():
                continue
            # 简单判断是否是中文
            if any('\u4e00' <= c <= '\u9fff' for c in word):
                vocab.append({"text": word, "lang": "zh"})
            else:
                vocab.append({"text": word, "lang": "en"})
        return vocab
    
    def ensure_vocabulary(self, hotwords: List[str]) -> Optional[str]:
        """
        确保热词表存在并更新
        1. 先查找是否有相同prefix的热词表
        2. 如果有，尝试更新
        3. 如果更新失败或找不到，清空所有热词表并重新创建
        4. 如果创建时遇到429错误，也会自动清空后重新创建
        """
        # 转换为API格式
        vocab = self._hotwords_to_vocab_format(hotwords)
        if not vocab:
            print("⚠️ 热词列表为空，跳过热词表创建")
            return None
        

        print("📋 当前所有热词表列表：")
        try:
            all_vocabs = self.svc.list_vocabularies()
            if all_vocabs and len(all_vocabs) > 0:
                for idx, v in enumerate(all_vocabs, 1):
                    vid = v.get('vocabulary_id') or v.get('id')
                    status = v.get('status') or v.get('state') or 'UNKNOWN'
                    prefix = v.get('prefix') or 'N/A'
                    created = v.get('created_time') or v.get('create_time') or 'N/A'
                    print(f"  [{idx}] ID: {vid}")
                    print(f"      Prefix: {prefix}, Status: {status}, Created: {created}")
            else:
                print("  （无热词表）")
        except Exception as e:
                print(f"  ⚠️ 获取热词表列表失败: {e}")
        
        print(f"📝 准备更新 {len(vocab)} 个热词...")
        
        # 查找现有热词表
        existing_id = self._find_existing_vocabulary()
        
        if existing_id:
            print(f"🔍 找到现有热词表: {existing_id}")
            try:
                # 尝试更新
                self.svc.update_vocabulary(existing_id, vocab)
                print(f"✓ 热词表更新成功: {existing_id}")
                self.vocabulary_id = existing_id
                return existing_id
                
            except Exception as e:
                error_msg = str(e)
                print(f"✗ 更新失败: {error_msg}")
                
                # 检查是否是找不到该ID或429错误
                if "not found" in error_msg.lower() or "404" in error_msg or \
                   "429" in error_msg or "quota" in error_msg.lower() or "limit" in error_msg.lower():
                    
                    if "not found" in error_msg.lower() or "404" in error_msg:
                        print("⚠️ 热词表不存在，清空所有热词表并重新创建...")
                    else:
                        print("⚠️ 检测到配额超限，清空所有热词表并重新创建...")
                    
                    self._clear_all_vocabularies()
                    vocabulary_id = self._create_new_vocabulary(vocab)
                    self.vocabulary_id = vocabulary_id
                    return vocabulary_id
                else:
                    print(f"⚠️ 更新热词表时遇到未知错误: {e}")
                    raise
        
        # 没有现有热词表，直接创建
        else:
            print("🆕 未找到现有热词表，创建新的...")
            vocabulary_id = self._create_new_vocabulary(vocab)
            self.vocabulary_id = vocabulary_id
            return vocabulary_id


def ensure_vocabulary_id(prefix: Optional[str], words: Optional[List[str]], target_model: str) -> Optional[str]:
    """
    便捷函数：确保热词表ID存在
    包含完整的错误处理和自动清理逻辑
    """
    if not prefix or not words:
        return None
    if not _VOCAB_AVAILABLE:
        print("⚠️ 当前 dashscope 版本不支持 VocabularyService；已跳过热词创建。")
        return None
    
    try:
        manager = VocabularyManager(target_model, prefix)
        return manager.ensure_vocabulary(words)
    except Exception as e:
        print(f"⚠️ 热词表创建/更新失败: {e}")
        return None


# ===================== Gummy 单句会话封装 =====================

class _GummyOneShot(TranslationRecognizerCallback):
    def __init__(self, cfg: VoiceConfig, vocabulary_id: Optional[str] = None):
        self.cfg = cfg
        self._vid = vocabulary_id
        self._cli: Optional[TranslationRecognizerChat] = None
        self._final_text: str = ""
        self._opened = threading.Event()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._err: Optional[str] = None

    def on_open(self):
        self._opened.set()

    def on_event(self, request_id, transcription_result: TranscriptionResult, translation_result: TranslationResult, usage):
        if transcription_result is not None and transcription_result.text:
            with self._lock:
                self._final_text = transcription_result.text

    def on_error(self, result):
        with self._lock:
            self._err = f"Gummy error: {result}"
        self._closed.set()

    def on_complete(self):
        self._closed.set()

    def on_close(self):
        self._closed.set()

    def start(self):
        self._cli = TranslationRecognizerChat(
            model=self.cfg.model,
            format="pcm",
            sample_rate=16000,
            transcription_enabled=True,
            callback=self,
            max_end_silence=self.cfg.gummy_max_end_silence_ms,
            vocabulary_id=self._vid if self._vid else None,
        )
        self._cli.start()
        self._opened.wait(timeout=5.0)

    def send_audio(self, pcm_bytes: bytes) -> bool:
        if not self._cli:
            return False
        return self._cli.send_audio_frame(pcm_bytes)

    def stop(self):
        if self._cli:
            self._cli.stop()
        self._closed.wait(timeout=5.0)

    @property
    def final_text(self) -> str:
        with self._lock:
            return (self._final_text or "").strip()

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._err


# ===================== 数字归一化 =====================

_UNIT_MM = {
    # 中文
    "毫米": 1.0, "厘米": 10.0, "米": 1000.0,
    # 英文
    "mm": 1.0, "millimeter": 1.0, "millimeters": 1.0,
    "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0,
    "m": 1000.0, "meter": 1000.0, "meters": 1000.0,
}

_EN_UNITS = {"zero":0,"oh":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}
_EN_TEENS = {"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19}
_EN_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
_EN_NUM_WORD = (
    r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|half|quarter)"
)
_NUM_PAT_ENFREE = rf"(?:{_EN_NUM_WORD}(?:[-\s]{_EN_NUM_WORD}){{0,4}})"
_CN_DIG = {"零":0,"〇":0,"○":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}

def _en_to_float(tok: str) -> Optional[float]:
    t = (tok or "").strip().lower()
    if not t:
        return None
    t = t.replace("-", " ")
    if t in ("half", "a half", "half a"):
        return 0.5
    if t in ("quarter", "a quarter"):
        return 0.25
    if " point " in t:
        left, right = t.split(" point ", 1)
        iv = _en_to_float(left)
        if iv is None:
            return None
        frac = 0.0; mul = 0.1
        for w in right.split():
            if w in _EN_UNITS:
                frac += _EN_UNITS[w] * mul; mul *= 0.1
            elif w in ("zero","oh"):
                mul *= 0.1
            else:
                return None
        return iv + frac
    if t.endswith(" and a half"):
        base = _en_to_float(t[: -len(" and a half")]); return (base + 0.5) if base is not None else None
    if t.endswith(" and a quarter"):
        base = _en_to_float(t[: -len(" and a quarter")]); return (base + 0.25) if base is not None else None
    parts = [w for w in t.split() if w not in ("and",)]
    if not parts: return None
    total = 0; current = 0; i = 0
    while i < len(parts):
        w = parts[i]
        if w in _EN_UNITS: current += _EN_UNITS[w]
        elif w in _EN_TEENS: current += _EN_TEENS[w]
        elif w in _EN_TENS:
            val = _EN_TENS[w]
            if i + 1 < len(parts) and parts[i+1] in _EN_UNITS:
                val += _EN_UNITS[parts[i+1]]; i += 1
            current += val
        elif w == "hundred":
            current = 100 if current == 0 else current * 100
        else: return None
        i += 1
    total += current
    if total == 0 and t in ("zero","oh"): return 0.0
    return float(total) if total != 0 else None

def _cn_to_float(tok: str) -> Optional[float]:
    tok = (tok or "").strip()
    if not tok: return None
    try: return float(tok)
    except Exception: pass
    if "点" in tok:
        left, right = tok.split("点", 1)
        lv = _cn_to_float(left) if left else 0.0
        rv = 0.0; base = 0.1
        for ch in right:
            if ch in _CN_DIG: rv += _CN_DIG[ch]*base; base *= 0.1
        return (lv or 0.0) + rv
    if "十" in tok:
        parts = tok.split("十")
        tens = _CN_DIG.get(parts[0], 1) if parts[0] else 1
        units = _CN_DIG.get(parts[1], 0) if len(parts) > 1 else 0
        return float(tens*10 + units)
    if tok == "半": return 0.5
    if all(ch in _CN_DIG for ch in tok):
        val = 0
        for ch in tok: val = val*10 + _CN_DIG[ch]
        return float(val)
    return None

def normalize_number(text: str) -> Optional[float]:
    if not text: return None
    t = text.lower()
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", t)
    if m:
        try: return float(m.group(1))
        except Exception: pass
    m = re.search(r"[零〇○一二两三四五六七八九十点半]+", text)
    if m:
        v = _cn_to_float(m.group(0))
        if v is not None: return v
    cands = list(re.finditer(_NUM_PAT_ENFREE, t))
    if cands:
        cands.sort(key=lambda mm: (mm.start(), -(mm.end()-mm.start())), reverse=True)
        for mm in cands:
            v = _en_to_float(mm.group(0))
            if v is not None: return v
    return None


# ====== 即时命令解析（不带“保持秒数”） ======
def parse_command(s: str) -> Dict[str, Any]:
    s = (s or "").strip().lower()
    out: Dict[str, Any] = {}
    if any(k in s for k in ["停止","急停","stop","停"]): return {"__stop": True}

    # 旋转（即时一帧）
    if any(k in s for k in ["左转","向左转","turn left","rotate left"]):
        n = normalize_number(s); out["theta.vel"] = +abs(n) if n is not None else 0.0
    if any(k in s for k in ["右转","向右转","turn right","rotate right"]):
        n = normalize_number(s); out["theta.vel"] = -abs(n) if n is not None else 0.0

    # 平移（不接受裸 left/right 与“向左/向右”）
    if any(k in s for k in ["前进","向前","forward","go forward","ahead"]):
        n = normalize_number(s); unit = next((u for u in _UNIT_MM if u in s), None)
        out["x.vel"] = + (n * _UNIT_MM[unit]) / 1000.0 if unit and n is not None else +0.0
    if any(k in s for k in ["后退","向后","倒退","back","backward","go back"]):
        n = normalize_number(s); unit = next((u for u in _UNIT_MM if u in s), None)
        out["x.vel"] = - (n * _UNIT_MM[unit]) / 1000.0 if unit and n is not None else -0.0
    if any(k in s for k in ["左移","向左平移","move left","strafe left"]):
        n = normalize_number(s); unit = next((u for u in _UNIT_MM if u in s), None)
        out["y.vel"] = + (n * _UNIT_MM[unit]) / 1000.0 if unit and n is not None else +0.0
    if any(k in s for k in ["右移","向右平移","move right","strafe right"]):
        n = normalize_number(s); unit = next((u for u in _UNIT_MM if u in s), None)
        out["y.vel"] = - (n * _UNIT_MM[unit]) / 1000.0 if unit and n is not None else -0.0

    # 升降（相对 → 在 _handle_final_text 转为绝对）
    if any(k in s for k in ["上升","升高","上移","up","raise","lift up"]):
        n = normalize_number(s) or 0.0; unit = next((u for u in _UNIT_MM if u in s), "毫米")
        out["lift_axis.height_mm"] = + (n * _UNIT_MM[unit])
    if any(k in s for k in ["下降","降低","下移","down","lower"]):
        n = normalize_number(s) or 0.0; unit = next((u for u in _UNIT_MM if u in s), "毫米")
        out["lift_axis.height_mm"] = - (n * _UNIT_MM[unit])


    if ("锤他" in s) or ("chui ta" in s) or ("hammer him" in s):
        out["__replay"] = {
            "dataset": "liyitenga/record_20251015131957",
            "episode": 0,
    }
    return out


# ====== “保持 N 秒”解析（更宽松） ======
_EN_SEC = r"(?:seconds?|sec|s)\b"
_CN_SEC = r"(?:秒钟|秒)\b"
_SEC_ANY = fr"(?:{_CN_SEC}|{_EN_SEC})"
_NUM_PAT = rf"([-+]?\d+(?:\.\d+)?|[零〇○一二两三四五六七八九十点半]+|{_NUM_PAT_ENFREE})"

def _extract_secs_anywhere(s: str) -> Optional[float]:
    """从句子任意位置提取时长（数字 + 秒单位），允许标点/空白间隔。"""
    t = (s or "").lower()
    # 1) 直接数值 + 单位（允许中间有标点/空白）
    m = re.search(fr"{_NUM_PAT}[\s,.;:-]*{_SEC_ANY}", t)
    if m:
        v = normalize_number(m.group(1))
        if v is not None:
            return max(0.1, float(v))
    # 2) 英文先单位后数字（很少见，但兼容 "seconds one"）
    m = re.search(fr"{_SEC_ANY}[\s,.;:-]*{_NUM_PAT}", t)
    if m:
        v = normalize_number(m.group(1))
        if v is not None:
            return max(0.1, float(v))
    return None

def _parse_hold(s: str) -> Optional[Dict[str, Any]]:
    """更容错：只要句子包含一个明确意图 + 任意位置的时长，就触发保持。意图优先级：旋转 > 平移左右 > 前进/后退"""
    s = (s or "").strip().lower()
    secs = _extract_secs_anywhere(s)
    if secs is None:
        return None

    # 明确意图触发词
    if any(k in s for k in ["左转","向左转","turn left","rotate left"]):
        return {"kind":"rotate_left","secs":secs}
    if any(k in s for k in ["右转","向右转","turn right","rotate right"]):
        return {"kind":"rotate_right","secs":secs}
    if any(k in s for k in ["左移","向左平移","move left","strafe left"]):
        return {"kind":"left","secs":secs}
    if any(k in s for k in ["右移","向右平移","move right","strafe right"]):
        return {"kind":"right","secs":secs}
    if any(k in s for k in ["前进","向前","forward","go forward","ahead"]):
        return {"kind":"forward","secs":secs}
    if any(k in s for k in ["后退","向后","倒退","back","backward","go back"]):
        return {"kind":"backward","secs":secs}

    return None


def _kind_to_cmd(kind: str, cfg: VoiceConfig) -> Dict[str, float]:
    v = cfg.xy_speed_cmd; w = cfg.theta_speed_cmd
    if kind == "forward": return {"x.vel": +v, "y.vel": 0.0, "theta.vel": 0.0}
    if kind == "backward": return {"x.vel": -v, "y.vel": 0.0, "theta.vel": 0.0}
    if kind == "left": return {"x.vel": 0.0, "y.vel": +v, "theta.vel": 0.0}
    if kind == "right": return {"x.vel": 0.0, "y.vel": -v, "theta.vel": 0.0}
    if kind == "rotate_left": return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": +w}
    if kind == "rotate_right": return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": -w}
    return {}


# ===================== 语音主引擎 =====================
class VoiceEngine:
    def __init__(self, cfg: VoiceConfig):
        self.cfg = cfg
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key: raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY")
        dashscope.api_key = api_key

        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._stop_evt = threading.Event()
        self._worker: Optional[threading.Thread] = None

        self._env_db: Optional[float] = None
        self._last_vol_print = 0.0

        self._speech_active = False
        self._cloud: Optional[_GummyOneShot] = None
        self._last_voice_ts = 0.0
        self._phrase_start = None

        self.device_sr = 16000
        self._stream: Optional[sd.InputStream] = None

        # 输出缓存
        self._one_shot_action: Dict[str, float] = {}
        self._now_height_mm: float = 0.0
        self._hold_until: float = 0.0
        self._held_cmd: Dict[str, float] = {}

        # 固定热词：始终尝试创建/复用；失败则降级为 N
        self._vocabulary_id: Optional[str] = cfg.vocabulary_id
        self._vocab_words_cnt = len(cfg.hotwords) if (cfg.hotwords is not None) else 0
        if not self._vocabulary_id:
            if cfg.vocabulary_prefix and cfg.hotwords:
                self._vocabulary_id = ensure_vocabulary_id(cfg.vocabulary_prefix, cfg.hotwords, cfg.model)
        self._vocab_enabled = bool(self._vocabulary_id)

    def set_height_mm(self, h: float): self._now_height_mm = float(h)

    def _audio_cb(self, indata: np.ndarray, frames: int, time_info, status):
        mono = indata[:, 0].copy()
        try: self._q.put_nowait(mono)
        except queue.Full: pass

    def start(self):
        try:
            self._stream = sd.InputStream(
                samplerate=None, channels=self.cfg.channels, dtype="float32",
                blocksize=int(self.cfg.chunk_seconds * 16000), callback=self._audio_cb,
            ); self._stream.start(); self.device_sr = int(round(self._stream.samplerate))
        except Exception:
            self._stream = sd.InputStream(
                samplerate=16000, channels=self.cfg.channels, dtype="float32",
                blocksize=int(self.cfg.chunk_seconds * 16000), callback=self._audio_cb,
            ); self._stream.start(); self.device_sr = 16000

        self._cloud = _GummyOneShot(self.cfg, self._vocabulary_id); self._cloud.start()
        self._worker = threading.Thread(target=self._run, daemon=True); self._worker.start()

        vocab_str = "Y" if self._vocab_enabled else "N"
        hot_cnt = f", hotwords={self._vocab_words_cnt}" if self._vocab_words_cnt else ""
        why = ""
        if not self._vocab_enabled:
            if not _VOCAB_AVAILABLE: why = "（dashscope 无 VocabularyService）"
            elif not self.cfg.vocabulary_prefix or not self.cfg.hotwords: why = "（未配置前缀/热词）"
            else: why = "（服务端拒绝或网络问题）"
        vid_str = f", vocab_id={self._vocabulary_id}" if self._vocab_enabled else ""
        print(f"🎤 语音已开启：device_sr={self.device_sr}Hz → resample→16000Hz, "
              f"model={self.cfg.model}, rel_gate={self.cfg.rel_db_margin_db}dB, "
              f"vocab={vocab_str}{hot_cnt}{vid_str}{why}")

    def stop(self):
        self._stop_evt.set()
        if self._worker: self._worker.join(timeout=2.0)
        if self._cloud: self._cloud.stop()
        if self._stream: self._stream.stop(); self._stream.close()

    def get_action_nowait(self) -> Dict[str, float]:
        now = time.time(); act: Dict[str, float] = {}
        if self._held_cmd and now < self._hold_until:
            act.update(self._held_cmd)
        else:
            if self._held_cmd and now >= self._hold_until:
                self._held_cmd.clear(); self._hold_until = 0.0
        if self._one_shot_action:
            act.update(self._one_shot_action); self._one_shot_action.clear()
        return act

    def _run(self):
        chunk = max(1, int(self.device_sr * self.cfg.chunk_seconds))
        tail = int(self.device_sr * self.cfg.overlap_seconds)
        silence_needed = self.cfg.speech_end_silence_ms / 1000.0
        buf = np.zeros(0, dtype=np.float32)
        while not self._stop_evt.is_set():
            try:
                piece = self._q.get(timeout=0.2); buf = np.concatenate([buf, piece])
            except queue.Empty:
                pass
            while len(buf) >= chunk:
                clip = buf[:chunk]; buf = buf[chunk - tail:] if tail > 0 else buf[chunk:]
                sig16 = resample_to_16k(clip, self.device_sr)
                level = dbfs(sig16)

                if getattr(self, "_env_db", None) is None or (level < self.cfg.min_dbfs):
                    a = self.cfg.env_track_alpha
                    self._env_db = level if getattr(self, "_env_db", None) is None else (a*self._env_db + (1.0-a)*level)
                now = time.time()
                if self.cfg.verbose_vol and (now - getattr(self, "_last_vol_print", 0.0) >= 1.0):
                    env = self._env_db if getattr(self, "_env_db", None) is not None else level
                    thr = max(self.cfg.min_dbfs, (env if env else level) + self.cfg.rel_db_margin_db)
                    print(f"[VOL] 当前帧 {level:.1f} dBFS | 背景 {env:.1f} dBFS | 门限 >= {thr:.1f}")
                    self._last_vol_print = now

                env = self._env_db if getattr(self, "_env_db", None) is not None else level
                rel_gate = (env if env is not None else level) + self.cfg.rel_db_margin_db
                gate = max(self.cfg.min_dbfs, rel_gate)
                is_voice = (level >= gate)

                if is_voice:
                    if not getattr(self, "_speech_active", False):
                        self._speech_active = True; self._phrase_start = now
                        self._cloud = self._cloud or _GummyOneShot(self.cfg, getattr(self, "_vocabulary_id", None))
                        if self._cloud and self._cloud.error:
                            self._cloud = _GummyOneShot(self.cfg, getattr(self, "_vocabulary_id", None)); self._cloud.start()
                    self._last_voice_ts = now
                    if self._cloud: self._cloud.send_audio(float32_to_pcm16(sig16))
                else:
                    if getattr(self, "_speech_active", False) and (now - getattr(self, "_last_voice_ts", now)) >= silence_needed:
                        self._speech_active = False
                        if self._cloud:
                            self._cloud.stop(); txt = self._cloud.final_text; err = self._cloud.error; self._cloud = None
                            self._handle_final_text(txt, err)
                        self._cloud = _GummyOneShot(self.cfg, getattr(self, "_vocabulary_id", None)); self._cloud.start()

                if getattr(self, "_speech_active", False) and getattr(self, "_phrase_start", None) and (now - self._phrase_start) > self.cfg.max_phrase_seconds:
                    self._speech_active = False
                    if self._cloud:
                        self._cloud.stop(); txt = self._cloud.final_text; err = self._cloud.error; self._cloud = None
                        self._handle_final_text(txt, err)
                    self._cloud = _GummyOneShot(self.cfg, getattr(self, "_vocabulary_id", None)); self._cloud.start()

    def _handle_final_text(self, text: str, err: Optional[str]):
        if err: print("[ASR] 错误：", err); return
        text = (text or "").strip()
        if not text: print("[ASR] 空文本。"); return
        print(f"[ASR] {text}")

        # ① 更宽松的“保持 N 秒”
        hold = _parse_hold(text)
        if hold is not None:
            kind = hold["kind"]; secs = float(hold["secs"])
            cmd = _kind_to_cmd(kind, self.cfg)
            self._held_cmd = cmd; self._hold_until = time.time() + secs
            self._one_shot_action = dict(cmd)
            print(f"{str(cmd)} 持续{secs:.1f}秒"); return

        # ② 即时口令
        parsed = parse_command(text)


        if "__replay" in parsed:
            params = parsed["__replay"] or {}
            dataset = str(params.get("dataset", "liyitenga/record_20251015131957"))
            episode = int(params.get("episode", 0))
            import sys, subprocess, shlex
            cmd = [sys.executable, "examples/alohamini/replay_bi.py",
                "--dataset", dataset, "--episode", str(episode)]
            print(f"[ASR] 触发锤他 → 执行: {' '.join(shlex.quote(c) for c in cmd)}")
            subprocess.Popen(cmd, cwd="/home/worker/lerobot2a")  # ← 改成你的项目根目录
        # 急停
        if parsed.get("__stop"):
            self._held_cmd.clear(); self._hold_until = 0.0
            base_cmd = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
            z_cmd = {"lift_axis.height_mm": self._now_height_mm}
            if self.cfg.emit_text_cmd: print(str(base_cmd), end=""); print(str(z_cmd))
            self._one_shot_action = dict(base_cmd); return

        # 相对高度→绝对
        if "lift_axis.height_mm" in parsed:
            delta = float(parsed["lift_axis.height_mm"])
            parsed["lift_axis.height_mm"] = self._now_height_mm + delta


        # 兜底速度
        if "theta.vel" in parsed and parsed["theta.vel"] == 0.0:
            parsed["theta.vel"] = self.cfg.theta_speed_cmd * (1.0 if "turn left" in text.lower() or "左转" in text else -1.0 if ("turn right" in text.lower() or "右转" in text) else 1.0)
        if "x.vel" in parsed and parsed["x.vel"] == 0.0:
            if any(k in text.lower() for k in ["前进","向前","forward","go forward","ahead"]): parsed["x.vel"] = +self.cfg.xy_speed_cmd
            elif any(k in text.lower() for k in ["后退","向后","倒退","back","backward","go back"]): parsed["x.vel"] = -self.cfg.xy_speed_cmd
        if "y.vel" in parsed and parsed["y.vel"] == 0.0:
            if any(k in text.lower() for k in ["左移","向左平移","move left","strafe left"]): parsed["y.vel"] = +self.cfg.xy_speed_cmd
            elif any(k in text.lower() for k in ["右移","向右平移","move right","strafe right"]): parsed["y.vel"] = -self.cfg.xy_speed_cmd

        base_cmd = {k: float(parsed[k]) for k in ("x.vel","y.vel","theta.vel") if k in parsed}
        z_cmd = {"lift_axis.height_mm": float(parsed["lift_axis.height_mm"])} if "lift_axis.height_mm" in parsed else {}

        if self.cfg.emit_text_cmd:
            printable_base = {"x.vel": base_cmd.get("x.vel", 0.0),
                              "y.vel": base_cmd.get("y.vel", 0.0),
                              "theta.vel": base_cmd.get("theta.vel", 0.0)}
            printable_z = {"lift_axis.height_mm": z_cmd.get("lift_axis.height_mm", self._now_height_mm)}
            print(str(printable_base), end=""); print(str(printable_z))

        

        self._one_shot_action.clear(); self._one_shot_action.update(base_cmd); self._one_shot_action.update(z_cmd)


# ===================== CLI =====================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--min-dbfs", type=float, default=-30.0)
    p.add_argument("--rel-db", type=float, default=7.0)
    p.add_argument("--model", type=str, default="gummy-chat-v1")
    p.add_argument("--xy-speed-cmd", type=float, default=0.10, help="前/后/左/右 时的 x/y.vel 大小（按你控制栈单位）")
    p.add_argument("--theta-speed-cmd", type=float, default=50.0, help="左/右转时的 theta.vel 大小（按你控制栈单位）")

    args = p.parse_args()
    cfg = VoiceConfig(
        min_dbfs=args.min_dbfs,
        rel_db_margin_db=args.rel_db,
        model=args.model,
        xy_speed_cmd=args.xy_speed_cmd,
        theta_speed_cmd=args.theta_speed_cmd,
    )

    eng = VoiceEngine(cfg); eng.start()
    try:
        while True: time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()