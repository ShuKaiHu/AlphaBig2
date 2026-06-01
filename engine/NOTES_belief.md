# Belief model — 階段性準確度觀念 (重要)

## 核心觀念 (使用者提出, 2026-06)

Belief model（推斷對手手牌）的準確度**必須分早期/晚期看，不能整局平均**：

- **早期（剛發牌）**：沒人出過牌，資訊量幾乎為 0 → belief 理論上**不可能準**，
  接近亂猜是正常的，不是 bug。
- **晚期（大量牌已出）**：看得到每家分別出了什麼牌、pass 過什麼 →
  資訊量大 → belief **應該要很準**。

而且**晚期正是 determinization 最關鍵的時候**（殘局精算），所以晚期 belief
準不準，直接決定線上 MCTS 在關鍵時刻的強度。

## 評估方法（正確做法）

不要報整局平均 lift（會被早期稀釋）。要**分階段 bucket**：
以「還有多少張牌未知（= 對手手上的牌）」當階段指標：
- early: >27 unknown
- mid:   15–27 unknown
- late:  <15 unknown

指標：`precision@k`（belief 排名前 k 的牌命中對手真實持牌的比例，
k = 該對手真實手牌數），對照 `base = k/unknown`（亂猜），看 `lift = prec/base`。
- lift 1.0 = 無用；>1.3 = 明顯有預測力。

工具：`probe_belief.py`（已分階段）。

## 量測紀錄

**deploy_best_mcts.pt（belief weight=0.1，舊）：**
| phase | lift |
|-------|------|
| early | 1.01x |
| mid   | 1.04x |
| late  | 1.07x |

→ 趨勢正確（晚期較高）但幅度太小 = under-trained（0.1 權重被 policy/value 壓過）。

## 假設與下一步

**假設**：把 belief 權重提高（0.1→0.4）重訓後，**晚期 lift 應大幅上升**，
早期維持 ~1.0（早期本就無資訊，學不出來也合理）。

**若假設成立 → 實作 belief 引導 determinization：**
- 晚期：用 belief 機率分布採樣對手手牌（取代均勻隨機）
- 早期：直接用均勻隨機（belief 沒用，省算力）
- 對應修改：`alpha_big2_wrapper.py` 的 `_sample_opponent_hands`

**若晚期 lift 仍上不去 → 退路**：belief-independent 的「多重 determinization 平均」
（每步猜 N 副牌各跑 MCTS，平均）—— robust 但較貴。
