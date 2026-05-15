"""
rl_agent.py — Deep Q-Network (DQN) Reinforcement Learning agent.

เรียนรู้จาก P&L จริงของแต่ละ trade ที่ปิดไปแล้ว และปรับปรุง action policy
ให้ดีขึ้นเรื่อยๆ โดยไม่ต้องรอ retrain ทุก 6 ชั่วโมง

Architecture:
  State   : 46-dim feature vector (เหมือนกับ tabular ensemble)
  Actions : 3 → HOLD=0, BUY=1, SELL=2
  Reward  : normalized P&L / ATR  (ปรับด้วย Sharpe-like scaling)
  Network : 3-layer MLP (46→128→64→3) + BatchNorm + Dropout
  Training: Experience Replay (buffer=2000) + Target Network (soft update)
  Epsilon : decays 0.90 → 0.05 over 500 steps (exploration → exploitation)

Integration:
  - ให้ soft vote เพิ่มเติมใน prediction ensemble (weight grows with accuracy)
  - Weight เริ่มที่ 0, ค่อยๆ โตถึง max 0.20 (20%) ของ final probability
  - ไม่เคย override safety checks ของ risk manager
  - Fallback เป็น Q-table เมื่อ PyTorch ไม่ได้ติดตั้ง
"""

import json
import logging
import random
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger('AI-Trade')

_RL_DIR       = Path('models')
_RL_NET_PATH  = _RL_DIR / 'rl_dqn.pt'
_RL_META_PATH = _RL_DIR / 'rl_meta.json'

_N_ACTIONS = 3          # HOLD=0, BUY=1, SELL=2
_N_FEATS   = 46         # ต้องตรงกับ _N_TAB_FEATS ใน ai_model.py
_REPLAY_SIZE = 2000
_BATCH_SIZE  = 64
_GAMMA       = 0.95     # discount factor
_LR          = 5e-4
_TARGET_UPDATE = 50     # update target network every N steps
_EPS_START   = 0.90
_EPS_END     = 0.05
_EPS_DECAY   = 500      # steps to reach EPS_END
_MAX_WEIGHT  = 0.20     # max contribution to final ensemble probability

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    logger.info("PyTorch unavailable — RL agent using Q-table fallback")


# ── Neural Q-Network ──────────────────────────────────────────────────────────

if _TORCH_OK:
    class _QNet(nn.Module):
        """3-layer MLP Q-network."""

        def __init__(self, n_feats: int = _N_FEATS, n_actions: int = _N_ACTIONS):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_feats, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, n_actions),
            )

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            return self.net(x)


# ── Q-Table fallback ──────────────────────────────────────────────────────────

class _QTable:
    """
    Simple discretized Q-table fallback when PyTorch is unavailable.
    State discretized into (regime × trend × confidence_band) = 18 buckets.
    """

    def __init__(self):
        self._q = np.zeros((18, _N_ACTIONS), dtype=np.float32)
        self._counts = np.zeros(18, dtype=np.int32)
        self._lr = 0.1

    def _state_idx(self, features: np.ndarray) -> int:
        """Map 46-dim features → one of 18 discrete states."""
        regime_enc  = float(features[37]) if len(features) > 37 else 0.5
        rsi_norm    = float(features[0])  if len(features) > 0  else 0.5
        macd_hist   = float(features[3])  if len(features) > 3  else 0.0

        r = 0 if regime_enc < 0.33 else (1 if regime_enc < 0.67 else 2)
        t = 0 if rsi_norm < 0.40 else (1 if rsi_norm < 0.60 else 2)
        m = 0 if macd_hist < -0.001 else (1 if macd_hist < 0.001 else 2)
        return r * 6 + t * 2 + m

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return self._q[self._state_idx(features)].copy()

    def update(self, features: np.ndarray, action: int, reward: float,
               next_features: np.ndarray, done: bool = True) -> None:
        s = self._state_idx(features)
        s2 = self._state_idx(next_features)
        target = reward + (0 if done else _GAMMA * self._q[s2].max())
        self._q[s, action] += self._lr * (target - self._q[s, action])
        self._counts[s] += 1


# ── Experience Replay Buffer ──────────────────────────────────────────────────

class _ReplayBuffer:
    def __init__(self, max_size: int = _REPLAY_SIZE):
        self._buf: deque = deque(maxlen=max_size)

    def push(self, state, action: int, reward: float,
             next_state, done: bool) -> None:
        self._buf.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, min(batch_size, len(self._buf)))
        states, actions, rewards, nexts, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.stack(nexts),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ── RL Agent ──────────────────────────────────────────────────────────────────

class RLAgent:
    """
    DQN agent สำหรับ XAUUSD trading.

    ใช้งาน:
      agent = RLAgent()

      # ทุก cycle (ก่อน open trade):
      action, q_vals = agent.select_action(features)  # 0=HOLD, 1=BUY, 2=SELL

      # หลัง trade ปิด:
      agent.record_outcome(entry_features, action, profit, next_features)

      # ดึง soft-probability สำหรับ ensemble:
      bull_prob = agent.bull_probability(features)
    """

    def __init__(self):
        self._step        = 0
        self._n_updates   = 0
        self._correct     = 0
        self._total       = 0

        self._replay      = _ReplayBuffer()
        self._last_state: Optional[np.ndarray] = None
        self._last_action: int = 0

        if _TORCH_OK:
            self._online  = _QNet()
            self._target  = _QNet()
            self._target.load_state_dict(self._online.state_dict())
            self._target.eval()
            self._optim   = optim.Adam(self._online.parameters(), lr=_LR, weight_decay=1e-5)
            self._loss_fn = nn.SmoothL1Loss()
            self._qtable  = None
        else:
            self._online  = None
            self._target  = None
            self._qtable  = _QTable()

        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if _RL_META_PATH.exists():
                meta = json.loads(_RL_META_PATH.read_text())
                self._step      = int(meta.get('step',      0))
                self._n_updates = int(meta.get('n_updates', 0))
                self._correct   = int(meta.get('correct',   0))
                self._total     = int(meta.get('total',     0))
            if _TORCH_OK and _RL_NET_PATH.exists() and self._online is not None:
                state = torch.load(_RL_NET_PATH, map_location='cpu', weights_only=True)
                self._online.load_state_dict(state)
                self._target.load_state_dict(state)
                self._target.eval()
                logger.info(
                    f"RL agent loaded | steps={self._step} "
                    f"updates={self._n_updates} acc={self.accuracy:.1%}"
                )
        except Exception as exc:
            logger.debug(f"RL load failed: {exc}")

    def save(self) -> None:
        try:
            _RL_DIR.mkdir(exist_ok=True)
            meta = {
                'step':      self._step,
                'n_updates': self._n_updates,
                'correct':   self._correct,
                'total':     self._total,
            }
            _RL_META_PATH.write_text(json.dumps(meta))
            if _TORCH_OK and self._online is not None:
                torch.save(self._online.state_dict(), _RL_NET_PATH)
        except Exception as exc:
            logger.debug(f"RL save failed: {exc}")

    # ── Epsilon schedule ──────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        decay = max(0, _EPS_DECAY - self._step) / _EPS_DECAY
        return _EPS_END + (_EPS_START - _EPS_END) * decay

    # ── Accuracy & weight ─────────────────────────────────────────────────────

    @property
    def accuracy(self) -> float:
        return self._correct / self._total if self._total > 0 else 0.5

    @property
    def weight(self) -> float:
        """
        Weight grows from 0 → max 0.20 as agent accumulates experience.
        Needs 50 trade outcomes before contributing; grows linearly to max at 500.
        """
        if self._total < 50:
            return 0.0
        base = min(_MAX_WEIGHT, (self._total - 50) / 2250 * _MAX_WEIGHT)
        # Accuracy bonus: scale up if better than random
        acc_factor = max(0.0, (self.accuracy - 0.50) * 2.0)
        return min(_MAX_WEIGHT, base * (0.5 + acc_factor))

    # ── Q-values ──────────────────────────────────────────────────────────────

    def _q_values(self, features: np.ndarray) -> np.ndarray:
        if _TORCH_OK and self._online is not None:
            self._online.eval()
            with torch.no_grad():
                t = torch.FloatTensor(features).unsqueeze(0)
                return self._online(t).squeeze(0).numpy()
        elif self._qtable is not None:
            return self._qtable.q_values(features)
        return np.zeros(_N_ACTIONS)

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(
        self, features: np.ndarray, explore: bool = True
    ) -> Tuple[int, np.ndarray]:
        """
        Epsilon-greedy action selection.

        Returns:
          action  — 0=HOLD, 1=BUY, 2=SELL
          q_vals  — Q-values for all 3 actions
        """
        feats = features.flatten()[:_N_FEATS]
        if len(feats) < _N_FEATS:
            feats = np.pad(feats, (0, _N_FEATS - len(feats)))

        q_vals = self._q_values(feats)

        if explore and random.random() < self.epsilon:
            action = random.randint(0, _N_ACTIONS - 1)
        else:
            action = int(np.argmax(q_vals))

        self._last_state  = feats.copy()
        self._last_action = action
        self._step += 1
        return action, q_vals

    # ── Reward shaping ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_reward(profit: float, atr: float, direction: str, action: int) -> float:
        """
        Reward shaping:
          - Profitable trade in correct direction: positive
          - Loss trade: negative
          - HOLD action when market moved strongly: small penalty
          - Sharpe-like normalization by ATR
        """
        atr = max(atr, 1.0)
        r_per_atr = profit / (atr * 10)   # normalize to R units (XAUUSD: ~10pts per ATR)

        if action == 0:  # HOLD
            # Penalize holding when a big move happened
            return max(-0.5, min(0.0, -abs(r_per_atr) * 0.3))

        expected_bull = (action == 1)
        actual_bull   = (profit > 0) if direction in ('BUY', '') else (profit <= 0)

        if expected_bull == actual_bull:
            return min(2.0, max(0.1, r_per_atr))   # correct direction
        else:
            return max(-2.0, min(-0.1, r_per_atr))  # wrong direction

    # ── Recording trade outcomes ──────────────────────────────────────────────

    def record_outcome(
        self,
        entry_features: np.ndarray,
        action: int,
        profit: float,
        next_features: np.ndarray,
        direction: str = '',
        atr: float = 1.0,
    ) -> None:
        """
        เรียกทุกครั้งที่ trade ปิด — อัพเดท Q-network ด้วย real P&L.

        entry_features : features ณ วันที่เปิด trade
        action         : action ที่ RL ให้ (1=BUY, 2=SELL, 0=HOLD)
        profit         : actual profit/loss จาก MT5
        next_features  : features ณ ปัจจุบัน (after trade close)
        direction      : 'BUY' หรือ 'SELL' จาก main engine
        atr            : ATR ณ ตอนเปิด trade สำหรับ normalization
        """
        feats = entry_features.flatten()[:_N_FEATS]
        if len(feats) < _N_FEATS:
            feats = np.pad(feats, (0, _N_FEATS - len(feats)))

        next_f = next_features.flatten()[:_N_FEATS]
        if len(next_f) < _N_FEATS:
            next_f = np.pad(next_f, (0, _N_FEATS - len(next_f)))

        reward = self._compute_reward(profit, atr, direction, action)

        # Update accuracy tracker
        self._total += 1
        if (action == 1 and profit > 0) or (action == 2 and profit <= 0):
            self._correct += 1

        # Push to replay buffer
        self._replay.push(feats, action, reward, next_f, done=True)

        # Train on replay if buffer is big enough
        self._train_step()

        logger.debug(
            f"RL update | action={['HOLD','BUY','SELL'][action]} "
            f"profit={profit:+.2f} reward={reward:+.3f} "
            f"weight={self.weight:.3f} acc={self.accuracy:.1%}"
        )

    def _train_step(self) -> None:
        """One mini-batch gradient update."""
        if len(self._replay) < max(64, _BATCH_SIZE):
            return
        if _TORCH_OK and self._online is not None:
            self._train_dqn()
        elif self._qtable is not None:
            self._train_qtable()

    def _train_dqn(self) -> None:
        states, actions, rewards, nexts, dones = self._replay.sample(_BATCH_SIZE)

        s  = torch.FloatTensor(states)
        a  = torch.LongTensor(actions).unsqueeze(1)
        r  = torch.FloatTensor(rewards)
        ns = torch.FloatTensor(nexts)
        d  = torch.FloatTensor(dones)

        self._online.train()
        q_curr = self._online(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            # Double DQN: online selects action, target evaluates value
            best_next = self._online(ns).argmax(1, keepdim=True)
            q_next    = self._target(ns).gather(1, best_next).squeeze(1)
            q_target  = r + _GAMMA * q_next * (1.0 - d)

        loss = self._loss_fn(q_curr, q_target)
        self._optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online.parameters(), 1.0)
        self._optim.step()

        # Soft target network update (τ = 0.01)
        self._n_updates += 1
        if self._n_updates % _TARGET_UPDATE == 0:
            for p, tp in zip(self._online.parameters(), self._target.parameters()):
                tp.data.copy_(0.01 * p.data + 0.99 * tp.data)

    def _train_qtable(self) -> None:
        if len(self._replay) < 4:
            return
        batch = self._replay.sample(min(16, len(self._replay)))
        states, actions, rewards, nexts, dones = batch
        for i in range(len(states)):
            self._qtable.update(states[i], int(actions[i]), float(rewards[i]),
                                nexts[i], bool(dones[i]))

    # ── Ensemble probability output ────────────────────────────────────────────

    def bull_probability(self, features: np.ndarray) -> Optional[float]:
        """
        แปลง Q-values เป็น bullish probability [0, 1] สำหรับ ensemble.

        ใช้ softmax บน Q(BUY) และ Q(SELL):
          bull_prob = exp(Q_buy) / (exp(Q_buy) + exp(Q_sell) + exp(Q_hold))
        """
        if self.weight == 0:
            return None
        feats = features.flatten()[:_N_FEATS]
        if len(feats) < _N_FEATS:
            feats = np.pad(feats, (0, _N_FEATS - len(feats)))

        q = self._q_values(feats)
        q_exp = np.exp(q - q.max())   # numerically stable softmax
        probs = q_exp / q_exp.sum()
        # bull_prob = P(BUY) + 0.5 * P(HOLD) [proportional split]
        bull_prob = float(probs[1] + 0.5 * probs[0])
        return float(np.clip(bull_prob, 0.0, 1.0))

    # ── Status summary ────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            'step':       self._step,
            'n_updates':  self._n_updates,
            'n_outcomes': self._total,
            'accuracy':   round(self.accuracy, 4),
            'weight':     round(self.weight, 4),
            'epsilon':    round(self.epsilon, 4),
            'buf_size':   len(self._replay),
            'backend':    'dqn' if _TORCH_OK else 'qtable',
        }
