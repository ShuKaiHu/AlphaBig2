# Iteration Plan (Policy / Value Alternating Training)

本文件定義 `AlphaBig2` 目前的穩健迭代流程，目標是避免模型被覆蓋掉、避免訓練退步時污染基線，並讓每一輪結果可追蹤、可比較。

---

## 1. 目前已確認的事

### 已驗證有效

- `ValueModel` 在「四家 heuristic + 全資訊（oracle）」設定下可以穩定學到訊號
  - `val_mae` 約 `0.05 ~ 0.07`（對 `[-1,1]` target）
- 固定 `ValueModel` 後，單獨訓練 `PolicyModel` 可以明顯改善 player1 的表現
  - 實戰 `avg_reward` 與 `win_rate` 有提升

### 已驗證不理想

- `Policy + Value` 同時訓練（一起更新）容易不穩定，甚至退步
- 單看 `policy_loss / p_top1` 不足以判斷是否真的變強
  - 應以 `eval avg_reward` 為主 KPI

---

## 2. 三個核心模型與角色

- `belief`：目前 phase2/政策訓練常用 `oracle` 取代，之後可再回頭迭代
- `policy`：主決策模型，目標是提升 player1 的 `avg_reward`
- `value`：MCTS leaf evaluation 用，先穩定、再慢慢迭代

---

## 3. 模型存檔結構（已採用）

分類資料夾：

- `ML_SKHU/models/belief/`
- `ML_SKHU/models/policy/`
- `ML_SKHU/models/value/`

訓練工作目錄（暫存 / pipeline 直接讀寫）：

- `ML_SKHU/models/new_three/`
- `ML_SKHU/models/new_three_cycle/`

說明檔：

- `ML_SKHU/models/checkpoint_notes.md`

---

## 4. 存檔策略（重要）

### 原則

- **不要直接覆蓋 best 模型**
- 每次訓練結果先存「時間戳記版本」
- 通過升級門檻後，再更新 `best/current`

### 命名規則（建議）

#### Policy

- 時間戳記版本：
  - `ML_SKHU/models/policy/policy_iter03_YYYYMMDD_HHMMSS.pt`
  - `ML_SKHU/models/policy/policy_fixed_value_oracle_YYYYMMDD_HHMMSS.pt`
- 固定別名（只在通過門檻時更新）：
  - `ML_SKHU/models/policy/policy_current.pt`
  - `ML_SKHU/models/policy/policy_best.pt`

#### Value

- 時間戳記版本：
  - `ML_SKHU/models/value/value_iter03_YYYYMMDD_HHMMSS.pt`
  - `ML_SKHU/models/value/value_heuristic_fullinfo_YYYYMMDD_HHMMSS.pt`
- 固定別名（只在通過門檻時更新）：
  - `ML_SKHU/models/value/value_current.pt`
  - `ML_SKHU/models/value/value_best.pt`

#### Belief（未來）

- 時間戳記版本：
  - `ML_SKHU/models/belief/belief_iter03_YYYYMMDD_HHMMSS.pt`
- 固定別名：
  - `ML_SKHU/models/belief/belief_current.pt`
  - `ML_SKHU/models/belief/belief_best.pt`

---

## 5. 升級條件（Gate Criteria）

## 5.1 Policy 升級條件（主 KPI）

### 主指標

- `player1 avg_reward`（`eval_three_models`, 建議 `games=500`）

### 輔指標

- `player1 win_rate`

### 升級規則（建議）

- 若 `avg_reward(new) > avg_reward(current_best) + 0.3`：
  - 接受新 policy（更新 `policy_current.pt` / `policy_best.pt`）
- 若差距在 `[-0.3, +0.3]`：
  - 看 `win_rate` 是否明顯更高（例如 `+0.02` 以上）再決定
- 若 `avg_reward` 變差：
  - 不升級，只保留 timestamp 檔案作紀錄

說明：`0.3` 是抗噪音門檻，可依實際波動調整到 `0.5`

## 5.2 Value 升級條件（先用雙指標）

### 指標 A（監督學習）

- `val_mae`（固定驗證集）

### 指標 B（實戰效果）

- 固定 policy 下的小型 MCTS eval（例如 `games=100~200`）
- 觀察 `player1 avg_reward`

### 升級規則（建議）

- `val_mae` 明顯改善（例如 `>= 0.005`）
  - 或在固定 policy 的實戰 eval 有改善
- 若兩者都無改善：
  - 不升級，只保留 timestamp 檔案

---

## 6. Value 訓練資料來源策略（關鍵）

問題：`ValueModel` 訓練時要用 `heuristic` 還是 `policy` 對局資料？

### 答案：分階段 + 混合（推薦）

#### 初期（bootstrap）

- `heuristic` 為主
- 目標：學到穩定基本價值結構

#### 中期開始

- 混合 `heuristic + current policy`
- 目標：讓 value 貼近目前 policy 的狀態分布

### 建議混合比例

- Iter 1 ~ 2：`heuristic 80% + policy 20%`
- Iter 3 ~ 5：`heuristic 50% + policy 50%`
- Iter 6+：`heuristic 20% + policy 80%`

備註：
- 若 policy 突然退步，暫時提高 heuristic 比例，穩住 value

---

## 7. 每一輪迭代流程（推薦）

以下流程以「先 policy、再 value」為主：

### Round N

1. **Policy Training (fixed value)**
   - belief: frozen / `oracle`
   - value: fixed（使用 `value_current.pt`）
   - train policy only

2. **Policy Eval (before/after)**
   - `games=500`
   - 記錄：
     - player1 `avg_reward`
     - player1 `win_rate`

3. **Policy Gate**
   - 若達升級條件：
     - 存 timestamp
     - 更新 `policy_current.pt`
     - 若是歷史最佳則更新 `policy_best.pt`
   - 否則：
     - 不覆蓋 current/best

4. **Value Training (fixed policy distribution source)**
   - 先從 `heuristic + oracle` 或混合資料開始
   - 之後逐步提高 policy 生成資料比例

5. **Value Eval**
   - `val_mae`
   -（可選）固定 policy 下小型實戰測試

6. **Value Gate**
   - 達標才更新 `value_current.pt` / `value_best.pt`

---

## 8. KPI 優先級（避免看錯指標）

### Policy（主）

1. `player1 avg_reward`（最重要）
2. `player1 win_rate`
3. `p_top1`
4. `target_expected_value_gap`
5. `policy_loss`

說明：
- `policy_loss / p_top1` 可能不變，但實戰 reward 仍可進步

### Value（主）

1. `val_mae`
2.（建議補）固定 policy 下的小型實戰效果
3. `value_loss`（單 batch 僅作參考）

---

## 9. 目前可用的基線模型（已知）

### Value baseline（固定 value 訓練來源）

- `ML_SKHU/models/value/value_heuristic_fullinfo.pt`

### Policy checkpoint（目前表現不錯）

- `ML_SKHU/models/policy/policy_fixed_value_oracle_20260223_003500.pt`
- 詳細訓練結果紀錄：
  - `ML_SKHU/models/checkpoint_notes.md`

---

## 10. 下一步實作建議（可選）

1. 建立 `policy_current.pt / policy_best.pt` 與 `value_current.pt / value_best.pt` 自動更新腳本
2. 建立 `ITER_LOG.csv`（每輪記錄 before/after 指標）
3. 建立「混合資料版 value 訓練腳本」
4. 後續再把 `belief` 拉回來進入三模型完整迭代

---

## 11. 操作原則（簡短版）

- 先穩定 `value`
- 固定 `value` 訓練 `policy`
- 每輪都做 `eval before/after`
- 只用 `avg_reward` 決定 policy 是否升級
- 一律存時間戳，不直接覆蓋 best

