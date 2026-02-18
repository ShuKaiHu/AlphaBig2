# Big 2 自我對戰強化學習 AI

這個專案實作了「大老二（Big 2）」的遊戲環境與規則。Big 2 是 4 人不完全資訊紙牌遊戲，出牌型態複雜（單張、對子、三條、兩對、順子、同花、葫蘆等），需要長期規劃與對手意圖推測。

## 特色
- Tkinter GUI 可直接遊玩
- 內附規則說明與圖片資源
- 兩條訓練管線：
  - policy-only（PPO/entropy/early-stop）
  - belief + policy_value + MCTS（不完全資訊）

## 快速開始

### 1) 直接遊玩 GUI
```bash
python generateGUI.py
```

## 訓練與評估

### A) policy-only 管線（不含 belief / MCTS）
```bash
./run_policy_reward_pipeline.sh
```
會產生每輪模型並在最後列出各輪對 heuristic 的平均 reward。

### B) belief + policy_value + MCTS 管線
```bash
./run_mcts_pipeline.sh
```
這條管線會：
1. 重新訓練 belief + policy_value
2. 對 heuristic 評估（MCTS）
3. 以 full-info 模型作為上限測試（若存在）

### Full-info / Unknown 測試
```bash
# 完全開放資訊（full_info_known）
.venv/bin/python -m ML_SKHU.policy_only.eval_reward \
  --policy-ckpt policy_best/policy_only_fullinfo_vs_heuristic.pt \
  --full-info-known --games 500 --opponent heuristic

# 不開放資訊（unknown）
.venv/bin/python -m ML_SKHU.policy_only.eval_reward \
  --policy-ckpt policy_best/policy_only_no_fullinfo_vs_heuristic.pt \
  --games 500 --opponent heuristic
```

### 自訂 cross_eval（四類對手）
```bash
.venv/bin/python -m ML_SKHU.policy_only.custom_cross_eval \
  --model-a policy_best/policy_only_no_fullinfo_vs_heuristic.pt \
  --model-b policy_best/policy_only_fullinfo_vs_heuristic.pt \
  --info-mode unknown \
  --games 500
```

## 專案結構
- `generateGUI.py`: GUI 對戰介面
- `big2Game.py` / `gameLogic.py`: 遊戲規則與狀態轉換
- `enumerateOptions.py`: 出牌選項枚舉
- `ML_SKHU/policy_only/*`: policy-only 訓練與評估
- `ML_SKHU/train.py`: belief + policy_value + MCTS 訓練
- `ML_SKHU/eval.py`: MCTS 評估

## 相關連結
- 規則說明：`rules.md`
- 線上版本（可能需要等待啟動）：https://big2-rl-4ba753215e7b.herokuapp.com/game/
- 訓練細節論文（arXiv）：https://arxiv.org/abs/1808.10442

## 更新說明（2023/10）
Heroku 線上版本曾經下線一段時間，現已重新部署在上述連結。若日後不再維護，作者也釋出可在本機遊玩的完整伺服器專案：
https://github.com/henrycharlesworth/big2_server/
