"""Step-by-step trace of ONE v6 game from player 1's perspective.

Shows, at each of player 1's decisions:
  - hand, table state, opponent counts
  - dominance features (the v6-specific 'will I get over-trumped?' signal)
  - the network's RAW policy (its instinct, top candidates)
  - the 4-dim value estimate (expected score per player)
  - what MCTS chose AFTER search (visit distribution) — the flywheel in action

Run:  BIG2_DOMINANCE=1 python trace_v6_game.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch

from engine.model import Big2Net
from engine.env import Big2Env, PASS_IDX
from engine.features import encode_static, encode_history_steps, STATIC_DIM, DOMINANCE_ON
from engine.mcts import MCTS
from engine.self_play import _heuristic_action
import enumerateOptions as eo

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
SIMS = 200
np.random.seed(SEED); torch.manual_seed(SEED)

SUIT = {1: "♣", 2: "♦", 3: "♥", 4: "♠"}
RANK = {1:"3",2:"4",3:"5",4:"6",5:"7",6:"8",7:"9",8:"10",9:"J",10:"Q",11:"K",12:"A",13:"2"}
def card(cid): return SUIT[(cid-1)%4+1] + RANK[(cid-1)//4+1]
def cards(cs): return " ".join(card(int(c)) for c in cs)

def describe(action):
    if action == PASS_IDX: return "PASS"
    cs, n = eo.getOptionNC(action)
    label = {1:"單",2:"對",5:"五張"}.get(n, f"{n}張")
    return f"出{label} [{cards(cs)}]"

assert DOMINANCE_ON, "請用 BIG2_DOMINANCE=1 執行 (v6 是 306 維)"
ck = torch.load("engine/checkpoints/saved/v6_dominance_deploy.pt", map_location="cpu")
model = Big2Net(); model.load_state_dict(ck); model.eval()
print(f"v6 模型載入 (STATIC_DIM={STATIC_DIM}, dominance=ON)  seed={SEED}  MCTS sims={SIMS}\n")

env = Big2Env(); env.reset()
mcts = MCTS(model, n_simulations=SIMS)
turn = 0
while not env.done:
    p = env.current_player
    g = env.game
    if p != 1:
        a = _heuristic_action(env)
        if a != PASS_IDX:
            cs, _ = eo.getOptionNC(a)
            print(f"  對手P{p}: {describe(a)}")
        else:
            print(f"  對手P{p}: PASS")
        env.step(a); continue

    turn += 1
    static = encode_static(g, 1); history = encode_history_steps(g)
    valid = env.get_valid_actions()
    # one forward pass → policy logits + 4-dim value + belief logits (3 opp × 52)
    with torch.no_grad():
        sf = torch.FloatTensor(static).unsqueeze(0)
        hs = torch.FloatTensor(history).unsqueeze(0)
        vm = torch.FloatTensor(valid).unsqueeze(0)
        logits, val_t, belief_logits = model.forward(sf, hs, vm)
    probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    value = val_t.squeeze(0).numpy()
    belief = torch.sigmoid(belief_logits).squeeze(0).numpy().reshape(3, 52)  # opp × card0..51
    dom = static[-4:]  # the 4 dominance features (last dims)

    print("="*64)
    print(f"【我的決策 #{turn}】  控制權={'我主導(lead)' if g.control==1 else '跟牌(follow)'}")
    print(f"  我的手牌({g.currentHands[1].size}): {cards(sorted(g.currentHands[1]))}")
    print(f"  對手剩牌: P2={g.currentHands[2].size} P3={g.currentHands[3].size} P4={g.currentHands[4].size}")
    if g.control == 0 and g.goIndex > 1 and (g.goIndex-1) in g.handsPlayed:
        print(f"  桌上要壓過: [{cards(g.handsPlayed[g.goIndex-1].hand)}]")
    # dominance features (v6 的核心)
    print(f"  ★dominance特徵: 有無敵單張={dom[0]:.0f} | 無敵單張數={dom[1]*5:.0f} | "
          f"最大對是nuts={dom[2]:.0f} | 對手可炸/同花順壓我={dom[3]:.0f}")
    print(f"  ★value估計(4家期望分,tanh): 我={value[0]:+.2f} 下家={value[1]:+.2f} "
          f"對家={value[2]:+.2f} 上家={value[3]:+.2f}")
    # belief: 網路猜每個對手手牌 (top-3 機率), ✓=真的有
    my_set = set(int(c) for c in g.currentHands[1])
    all_hands = set(int(c) for p in range(1,5) for c in g.currentHands[p])
    unseen = sorted(all_hands - my_set)
    seats = {0:("下家P2",2),1:("對家P3",3),2:("上家P4",4)}
    print("  ★belief(網路猜對手手牌, top3機率 ✓=真的有):")
    for oi,(name,pp) in seats.items():
        truth = set(int(c) for c in g.currentHands[pp])
        ranked = sorted(unseen, key=lambda c: -belief[oi][c-1])[:3]
        shown = " ".join(f"{card(c)}{belief[oi][c-1]*100:.0f}%{'✓' if c in truth else '✗'}" for c in ranked)
        print(f"      {name}: {shown}")
    # raw policy 前 4
    vi = np.flatnonzero(valid)
    non_pass = [a for a in vi if a != PASS_IDX]
    if not non_pass:
        print("  → 壓不過,只能 PASS")
        env.step(PASS_IDX); continue
    top_raw = sorted(vi, key=lambda a: -probs[a])[:4]
    print(f"  網路直覺(raw policy) top4:")
    for a in top_raw:
        print(f"      {probs[a]*100:5.1f}%  {describe(a)}")
    # MCTS 搜索
    action, visits = mcts.run(env, temperature=0.0)
    tot = visits.sum()
    top_mcts = sorted(np.flatnonzero(visits), key=lambda a: -visits[a])[:4]
    print(f"  MCTS搜索後(訪問次數,共{int(tot)}) top4:")
    for a in top_mcts:
        flag = " ←選這個" if a == action else ""
        print(f"      {visits[a]/tot*100:5.1f}%  {describe(a)}{flag}")
    env.step(action)

# 結果
r = g.rewards
print("="*64)
print(f"\n【結束】我最終得分 = {r[0]:+.0f}  (四家: 我={r[0]:+.0f} 下家={r[1]:+.0f} "
      f"對家={r[2]:+.0f} 上家={r[3]:+.0f})")
