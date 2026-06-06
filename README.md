# AlphaBig2 — 大老二 self-play 強化學習 AI

4 人不完全資訊紙牌遊戲「大老二」的 AlphaZero 式自我對戰 AI。
本 repo 是 **model 端**(訓練 + 模型);線上對戰的 executor/connector 在另一個
repo **`Big2VisionAgent-claude`**(driver 用本 repo 的模型在真實網頁上打)。

---

## 架構一覽

```
遊戲核心(不動)                 engine/ 現役框架                      部署
─────────────              ───────────────────              ──────────
big2Game.py     ┐          model.py    Big2Net 網路          checkpoints/
gameLogic.py    ├─ 規則 →   features.py 特徵編碼(+dominance)   ├ best.pt    (現役 v6)
enumerateOptions.py┘        mcts.py     多人 max^n MCTS         ├ deploy_best_mcts.pt (+7.1 備援)
                           env.py      Gym 式環境             └ saved/     (GitHub 存檔)
                           dominance.py 牌力支配(純邏輯)
                           self_play.py 4 人自我對戰
                           replay_buffer.py
                           trainer.py   訓練主迴圈
                           evaluator.py vs heuristic 評估
```

### 模型 = 一個網路、四個頭(`engine/model.py` `Big2Net`)
```
歷史出牌 → GRU(128) ┐
靜態特徵(302/306) ├→ 拼接 → 4×ResBlock trunk(256)
                   ┘        ├ policy head → 14739 動作機率
                            ├ value head  → 4 維(每家期望分數)★多人,非2人零和
                            └ belief head → 對手手牌(輔助,訓練用)
```

### 5 個運作零件
| 零件 | 作用 |
|------|------|
| **policy** | 出牌「直覺」(先驗機率) |
| **value(4維)** | 預測 4 家各自最終得分 — 修掉「2人零和」核心 bug |
| **belief** | 猜對手手牌(輔助;晚期較準,早期弱) |
| **dominance** | 精確算「我這張會不會被壓」(opt-in 特徵,見下) |
| **MCTS** | 用 policy 當先驗、value 當評估,往前推演修正直覺 |

---

## 訓練

```bash
# 標準配方(產生 baseline +7.1)
python -m engine.trainer --iterations 500 --self-play 50 --sims 50 \
  --bc-warmup 10 --bc-mix 0.15 --entropy-coef 0.0 --policy-temp 0.7 \
  --belief-coef 0.1 --eval-freq 10 --eval-games 100 --torch-threads 3

# 加上 dominance 特徵(v6):前面加 BIG2_DOMINANCE=1,並指定獨立 checkpoint 目錄
BIG2_DOMINANCE=1 python -m engine.trainer ... --checkpoint-dir engine/checkpoints_v6
```
關鍵旗標:`--league`(對手池)、`--belief-coef`、`--policy-temp`、`--entropy-coef`。

### dominance 特徵開關(重要)
`BIG2_DOMINANCE=1` 會在 `encode_static` 加 4 維 dominance 特徵(STATIC_DIM 302→306)。
- **預設關閉** → 與 +7.1 baseline 相容
- **用 dominance 訓練的模型(v6)務必也用 `BIG2_DOMINANCE=1` 載入/評估**,否則維度不符
  (wrapper 有防呆會大聲報錯)

---

## 評估 / 分析工具
| 工具 | 用途 |
|------|------|
| `eval_reward.py CKPT --games N --mcts S` | 對 heuristic 的 reward(perfect-info)|
| `eval_determinization.py CKPT --dets D --sims S [--belief]` | 不完全資訊評估(determinization)|
| `probe_belief.py CKPT` | belief 準確度(分早/中/晚期)|
| `trace_v6_game.py [seed]` | **逐步追蹤一局**:手牌→dominance→policy→value→belief→MCTS→出牌 |
| `sanity_multiplayer.py` | 4 維 value / max^n backup 的回歸測試 |
| `report.py` | 訓練進度報告(讀 logs/train.log)|

---

## 現況(2026-06)
- **核心 bug 已修**:4 人 max^n MCTS(原本誤用 2 人零和 backup);executor 零失敗。
- **部署模型**:`checkpoints/best.pt` = v6(dominance);`saved/` 有 v6 + baseline 兩個乾淨存檔。
- **強度**:對簡單 heuristic ≈ +7(MCTS);**對真人 ≈ 打平**(self-play 收斂在約 amateur)。
- **v6(dominance)**:平均分 ≈ baseline,但線上對真人**出完率↑、災難場↓**(分布較佳)。
- 細節研究日誌見 **`engine/NOTES_belief.md`**(所有診斷、教訓、實驗結論)。

### 突破天花板的待辦方向(NOTES 有詳述)
1. 約束式 determinization(用 pass 揭露的 void 排除不可能世界)
2. league / 真人棋譜模仿學習
3. 更大網路 / 更大算力(AlphaZero-scale)
