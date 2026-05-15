"""
market_memory.py — Case-Based Reasoning via feature vector similarity.

จำรูปแบบตลาดที่เคยเจอมาแล้ว และดึงประวัติเพื่อช่วยตัดสินใจ trade ใหม่.

หลักการทำงาน:
  1. ทุก trade ที่ปิด → บันทึก (entry_features, direction, profit, atr) ลง memory
  2. ก่อน trade ใหม่ → หา K patterns ที่คล้ายที่สุดด้วย cosine similarity
  3. คำนวณ "memory_confidence" จาก win rate ของ K nearest neighbors
  4. ส่ง confidence boost กลับไปให้ prediction ensemble

Memory Properties:
  - Max 500 entries (rolling window, oldest dropped when full)
  - Similarity metric: cosine similarity on normalized feature vectors
  - K = 10 nearest neighbors
  - Minimum similarity threshold: 0.75 (ไม่เอา pattern ที่ต่างกันมาก)
  - Pattern decay: ลด weight ของ patterns เก่ากว่า 72 ชั่วโมง

Expected Output:
  MemoryMatch(n=8, win_rate=0.75, avg_return_r=1.2, confidence_boost=0.06)
  → แสดงว่า 8 patterns ที่คล้าย win 75%, คาด return 1.2R → เพิ่ม conf 6%
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger('AI-Trade')

_MEM_PATH = Path('models') / 'market_memory.json'
_MAX_MEM  = 500
_K        = 10        # K nearest neighbors
_MIN_SIM  = 0.70      # minimum cosine similarity to count as "match"
_DECAY_H  = 72        # patterns older than this get half weight
_N_FEATS  = 46        # ต้องตรงกับ _N_TAB_FEATS ใน ai_model.py


@dataclass
class MemoryMatch:
    n_matches:       int
    win_rate:        float
    avg_return_r:    float
    confidence_boost: float   # additional confidence to add to prediction (0–0.10)
    top_similarity:  float    # similarity score of best match
    regime_match:    str      # regime ของ best match

    def summary(self) -> str:
        return (
            f"Memory: n={self.n_matches} win={self.win_rate:.1%} "
            f"avg_R={self.avg_return_r:+.2f} boost={self.confidence_boost:+.3f}"
        )


@dataclass
class _MemoryEntry:
    features:  List[float]
    direction: str          # 'BUY' | 'SELL'
    profit:    float        # actual profit/loss
    atr:       float        # ATR at entry (for R normalization)
    regime:    str          # 'TREND' | 'RANGE' | 'HIGH_VOL'
    ts:        str          # ISO timestamp


class MarketMemory:
    """
    Case-based reasoning memory สำหรับ XAUUSD trading.

    ใช้งาน:
      memory = MarketMemory()

      # หลัง trade ปิด:
      memory.store(features, 'BUY', profit=50.0, atr=2.5, regime='TREND')

      # ก่อน trade ใหม่ เพื่อดึง historical confidence:
      match = memory.recall(current_features, query_direction='BUY')
      if match.confidence_boost > 0.03:
          confidence += match.confidence_boost
    """

    def __init__(self):
        self._entries: List[_MemoryEntry] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _MEM_PATH.exists():
            return
        try:
            raw = json.loads(_MEM_PATH.read_text())
            self._entries = [
                _MemoryEntry(**e) for e in raw.get('entries', [])
            ]
            logger.info(f"Market memory loaded: {len(self._entries)} patterns")
        except Exception as exc:
            logger.debug(f"Memory load failed: {exc}")
            self._entries = []

    def save(self) -> None:
        try:
            _MEM_PATH.parent.mkdir(exist_ok=True)
            data = {
                'entries': [
                    {
                        'features':  e.features,
                        'direction': e.direction,
                        'profit':    e.profit,
                        'atr':       e.atr,
                        'regime':    e.regime,
                        'ts':        e.ts,
                    }
                    for e in self._entries
                ]
            }
            tmp = _MEM_PATH.with_suffix('.tmp')
            tmp.write_text(json.dumps(data))
            tmp.replace(_MEM_PATH)
        except Exception as exc:
            logger.debug(f"Memory save failed: {exc}")

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(
        self,
        features: np.ndarray,
        direction: str,
        profit:    float,
        atr:       float = 1.0,
        regime:    str   = 'TREND',
    ) -> None:
        """บันทึก trade ที่ปิดแล้วลง memory."""
        feats = features.flatten()[:_N_FEATS].tolist()
        if len(feats) < _N_FEATS:
            feats += [0.0] * (_N_FEATS - len(feats))

        entry = _MemoryEntry(
            features  = feats,
            direction = direction,
            profit    = float(profit),
            atr       = float(max(atr, 0.01)),
            regime    = regime,
            ts        = datetime.now().isoformat(),
        )
        self._entries.append(entry)

        # Rolling window — drop oldest
        if len(self._entries) > _MAX_MEM:
            self._entries = self._entries[-_MAX_MEM:]

        logger.debug(
            f"Memory stored: {direction} P&L={profit:+.2f} regime={regime} "
            f"total={len(self._entries)}"
        )

    # ── Cosine similarity ─────────────────────────────────────────────────────

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ── Time decay weight ─────────────────────────────────────────────────────

    @staticmethod
    def _time_weight(ts_str: str) -> float:
        """Patterns from recent hours count more (half-life = _DECAY_H hours)."""
        try:
            ts    = datetime.fromisoformat(ts_str)
            hours = (datetime.now() - ts).total_seconds() / 3600.0
            return max(0.10, 1.0 / (1.0 + hours / _DECAY_H))
        except Exception:
            return 0.5

    # ── Recall ────────────────────────────────────────────────────────────────

    def recall(
        self,
        query_features: np.ndarray,
        query_direction: str = '',
        regime: str = '',
    ) -> MemoryMatch:
        """
        ค้นหา K patterns ที่คล้ายที่สุดและคำนวณ memory_confidence.

        query_direction : 'BUY' | 'SELL' | '' (ใช้ทุก direction)
        regime          : 'TREND' | 'RANGE' | 'HIGH_VOL' | '' (ไม่กรอง)
        """
        null = MemoryMatch(0, 0.5, 0.0, 0.0, 0.0, '')
        if len(self._entries) < 5:
            return null

        q = query_features.flatten()[:_N_FEATS]
        if len(q) < _N_FEATS:
            q = np.pad(q, (0, _N_FEATS - len(q)))
        q = q.astype(np.float32)

        # Build similarity list
        sims: List[Tuple[float, _MemoryEntry]] = []
        for e in self._entries:
            # Direction filter (optional)
            if query_direction and e.direction != query_direction:
                continue
            # Regime filter — prefer same regime but allow mismatches
            regime_bonus = 0.05 if (regime and e.regime == regime) else 0.0

            ef = np.array(e.features, dtype=np.float32)
            sim = self._cosine_sim(q, ef) + regime_bonus
            if sim >= _MIN_SIM:
                decay = self._time_weight(e.ts)
                sims.append((sim * decay, e))

        if not sims:
            return null

        # Sort by similarity, take top K
        sims.sort(key=lambda x: x[0], reverse=True)
        top_k = sims[:_K]

        # Compute weighted win rate and average R
        total_w  = 0.0
        win_w    = 0.0
        sum_r    = 0.0
        best_sim = top_k[0][0]
        best_regime = top_k[0][1].regime

        for sim_w, e in top_k:
            r_return = e.profit / (e.atr * 10)  # normalize to R units
            win      = 1.0 if e.profit > 0 else 0.0
            total_w += sim_w
            win_w   += sim_w * win
            sum_r   += sim_w * r_return

        if total_w < 1e-10:
            return null

        win_rate   = win_w / total_w
        avg_r      = sum_r / total_w
        n_matches  = len(top_k)

        # Confidence boost: +0 to +0.10 based on win rate above 50%
        #   50% win rate → 0 boost
        #   75% win rate → +0.05 boost
        #   90% win rate → +0.08 boost
        edge = max(0.0, win_rate - 0.50)
        conf_boost = min(0.10, edge * 0.40)

        # Penalize if win rate below 50% (historical pattern suggests against us)
        if win_rate < 0.50:
            conf_boost = max(-0.05, (win_rate - 0.50) * 0.20)

        return MemoryMatch(
            n_matches        = n_matches,
            win_rate         = win_rate,
            avg_return_r     = float(avg_r),
            confidence_boost = float(conf_boost),
            top_similarity   = float(best_sim),
            regime_match     = best_regime,
        )

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self._entries:
            return {'total': 0, 'win_rate': 0.5, 'avg_profit': 0.0,
                    'regime_dist': {}, 'direction_dist': {}}

        wins = sum(1 for e in self._entries if e.profit > 0)
        profits = [e.profit for e in self._entries]

        # Regime distribution
        regime_dist: dict = {}
        for e in self._entries:
            regime_dist[e.regime] = regime_dist.get(e.regime, 0) + 1

        # Direction distribution
        dir_dist: dict = {}
        for e in self._entries:
            dir_dist[e.direction] = dir_dist.get(e.direction, 0) + 1

        return {
            'total':        len(self._entries),
            'win_rate':     round(wins / len(self._entries), 4),
            'avg_profit':   round(float(np.mean(profits)), 2),
            'total_profit': round(float(np.sum(profits)), 2),
            'regime_dist':  regime_dist,
            'direction_dist': dir_dist,
        }

    def __len__(self) -> int:
        return len(self._entries)
