# Big 2 自我對戰強化學習 AI

這個專案實作了「大老二（Big 2）」的遊戲環境與規則。Big 2 是 4 人不完全資訊紙牌遊戲，出牌型態複雜（單張、對子、三條、兩對、順子、同花、葫蘆等），需要長期規劃與對手意圖推測。

## 特色
- Tkinter GUI 可直接遊玩
- 內附規則說明與圖片資源

## 快速開始

### 1) 直接遊玩 GUI
```bash
python generateGUI.py
```

## 專案結構
- `generateGUI.py`: GUI 對戰介面
- `big2Game.py` / `gameLogic.py`: 遊戲規則與狀態轉換
- `enumerateOptions.py`: 出牌選項枚舉

## 相關連結
- 規則說明：`rules.md`
- 線上版本（可能需要等待啟動）：https://big2-rl-4ba753215e7b.herokuapp.com/game/
- 訓練細節論文（arXiv）：https://arxiv.org/abs/1808.10442

## 更新說明（2023/10）
Heroku 線上版本曾經下線一段時間，現已重新部署在上述連結。若日後不再維護，作者也釋出可在本機遊玩的完整伺服器專案：
https://github.com/henrycharlesworth/big2_server/
