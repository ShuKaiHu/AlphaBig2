import tkinter
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import big2Game
import enumerateOptions

from ML_SKHU.belief import BeliefModel
from ML_SKHU.policy_value import PolicyValueModel
from ML_SKHU.features import encode_belief_input, encode_full_info_with_belief, belief_targets

mainGame = big2Game.big2Game()

# Model paths (player1)
BELIEF_CKPT = "policy_best/belief_only_margin0.8_accknown_20260219_191635.pt"
POLICY_CKPT = "policy_best/policy_only_no_fullinfo_vs_heuristic_20260219_205951.pt"
POLICY_IS_POLICY_ONLY = True
DEVICE = "cpu"

belief_model = None
policy_value_model = None

def _load_models():
    global belief_model, policy_value_model
    # Belief
    b_in_dim = encode_belief_input(big2Game.big2Game(), 1).shape[0]
    belief_model = BeliefModel(b_in_dim).to(DEVICE)
    try:
        belief_model.load_state_dict(torch.load(BELIEF_CKPT, map_location=DEVICE))
        belief_model.eval()
    except Exception as e:
        print(f"[warn] belief ckpt incompatible with current input dim: {e}")
        belief_model = None
    # Policy/Value
    if POLICY_IS_POLICY_ONLY:
        p_in_dim = encode_belief_input(big2Game.big2Game(), 1).shape[0]
    else:
        dummy = big2Game.big2Game()
        dummy_belief = np.zeros((52, 3), dtype=np.float32)
        p_in_dim = int(encode_full_info_with_belief(dummy, 1, dummy_belief).shape[0])
    policy_value_model = PolicyValueModel(p_in_dim, hidden_dim=256 if POLICY_IS_POLICY_ONLY else 512).to(DEVICE)
    try:
        policy_value_model.load_state_dict(torch.load(POLICY_CKPT, map_location=DEVICE))
        policy_value_model.eval()
    except Exception as e:
        print(f"[warn] policy ckpt incompatible with current input dim: {e}")
        policy_value_model = None

try:
    _load_models()
except Exception as e:
    print(f"[warn] model load failed: {e}")

top=tkinter.Tk()

top.resizable(True, True)

dX = 17
hideOtherCards = 1
playersGo = 1
control = 0
X_OFFSET = 200

currSampledOption = -1

#load images
cardImages = {}
cardImages2 = {}
cardImages3 = {}
cardImages4 = {}
cardImages180 = {}
resample_mode = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
for i in range(1,53):
    string = "cardImages/" + str(i) + ".png"
    cardImages[i] = Image.open(string)
    cardImages[i] = cardImages[i].resize((63,91), resample_mode)
    cardImages2[i] = cardImages[i].rotate(270, expand=True)
    cardImages3[i] = cardImages[i]
    cardImages4[i] = cardImages[i].rotate(90, expand=True)
    cardImages180[i] = cardImages[i].rotate(180, expand=True)
    cardImages[i] = ImageTk.PhotoImage(cardImages[i])
    cardImages2[i] = ImageTk.PhotoImage(cardImages2[i])
    cardImages3[i] = ImageTk.PhotoImage(cardImages3[i])
    cardImages4[i] = ImageTk.PhotoImage(cardImages4[i])
    cardImages180[i] = ImageTk.PhotoImage(cardImages180[i])
backOfCard = Image.open("cardImages/back.jpg")
backOfCard = backOfCard.resize((63,91), resample_mode)
backOfCardRotated = backOfCard.rotate(90, expand=True)
backOfCardRotated180 = backOfCard.rotate(180, expand=True)
backOfCardRotated270 = backOfCard.rotate(270, expand=True)
backOfCard = ImageTk.PhotoImage(backOfCard)
backOfCardRotated = ImageTk.PhotoImage(backOfCardRotated)
backOfCardRotated180 = ImageTk.PhotoImage(backOfCardRotated180)
backOfCardRotated270 = ImageTk.PhotoImage(backOfCardRotated270)
    
player1Hand = {}
player2Hand = {}
player3Hand = {}
player4Hand = {}
for i in range(13):
    player1Hand[i] = tkinter.Label(top)
    player2Hand[i] = tkinter.Label(top)
    player3Hand[i] = tkinter.Label(top)
    player4Hand[i] = tkinter.Label(top)
    
currentOption = {}
prevHand1 = {}
prevHand2 = {}
prevHand3 = {}
for i in range(5):
    currentOption[i] = tkinter.Label(top)
    prevHand3[i] = tkinter.Label(top)
    prevHand2[i] = tkinter.Label(top)
    prevHand1[i] = tkinter.Label(top)
    

top.geometry("1600x1000")
top.title("Big 2 - Game Visualization and Testing")

def updatePrevHands():
    sX = 520 + X_OFFSET
    sY = 435
    cX = sX
    cY = sY
    #clear it
    for i in range(5):
        prevHand3[i].config(image='')
        prevHand2[i].config(image='')
        prevHand1[i].config(image='')
    #3rd most recent
    if mainGame.goIndex >= 4:
        thirdMostRecent = mainGame.handsPlayed[mainGame.goIndex-3].hand
        for i in range(len(thirdMostRecent)):
            prevHand3[i].config(image=cardImages[thirdMostRecent[i]])
            prevHand3[i].place(x=cX, y=cY)
            prevHand3[i].lift()
            cX += dX
    cX = sX
    cY = sY + 1.5*dX
    if mainGame.goIndex >= 3:
        secondMostRecent = mainGame.handsPlayed[mainGame.goIndex-2].hand
        for i in range(len(secondMostRecent)):
            prevHand2[i].config(image=cardImages[secondMostRecent[i]])
            prevHand2[i].place(x=cX, y=cY)
            prevHand2[i].lift()
            cX += dX
    cX = sX
    cY = sY + 3*dX
    if mainGame.goIndex >= 2:
        mostRecent = mainGame.handsPlayed[mainGame.goIndex-1].hand
        for i in range(len(mostRecent)):
            prevHand1[i].config(image=cardImages[mostRecent[i]])
            prevHand1[i].place(x=cX, y=cY)
            prevHand1[i].lift()
            cX += dX

def updateCurrentOption(hand, passing=0):
    #clear it
    for i in range(5):
        currentOption[i].config(image='')
    if passing == 1:
        return
    sX = 1000 + X_OFFSET
    sY = 115
    for i in range(len(hand)):
        currentOption[i].config(image=cardImages[hand[i]])
        currentOption[i].place(x=sX,y=sY)
        currentOption[i].lift()
        sX += dX

def updatePlayerHand(hand, player):
    if player==1:
        #remove current hand
        for i in range(13):
            player1Hand[i].config(image='')
        sX = 430 + X_OFFSET
        sY = 680
        #update hand
        for i in range(len(hand)):
            player1Hand[i].config(image=cardImages[hand[i]])
            player1Hand[i].place(x=sX,y=sY)
            player1Hand[i].lift()
            sX += dX
    elif player==2:
        #remove current hand
        for i in range(13):
            player2Hand[i].config(image='')
        sX = 70 + X_OFFSET
        sY = 600
        #update hand
        for i in range(len(hand)):
            if hideOtherCards == 0:
                player2Hand[i].config(image=cardImages2[hand[i]])
            else:
                player2Hand[i].config(image=backOfCardRotated)
            player2Hand[i].place(x=sX, y=sY)
            player2Hand[i].lift()
            sY -= dX
    elif player==3:
        for i in range(13):
            player3Hand[i].config(image='')
        sX = 430 + X_OFFSET
        sY = 280
        for i in range(len(hand)):
            if hideOtherCards == 0:
                player3Hand[i].config(image=cardImages3[hand[i]])
            else:
                player3Hand[i].config(image=backOfCard)
            player3Hand[i].place(x=sX,y=sY)
            player3Hand[i].lift()
            sX += dX
    elif player==4:
        for i in range(13):
            player4Hand[i].config(image='')
        sX = 1000 + X_OFFSET
        sY = 600
        for i in range(len(hand)):
            if hideOtherCards == 0:
                player4Hand[i].config(image=cardImages2[hand[i]])
            else:
                player4Hand[i].config(image=backOfCardRotated270)
            player4Hand[i].place(x=sX, y=sY)
            player4Hand[i].lift()
            sY -= dX
            
def changeShowHands():
    global hideOtherCards
    hideOtherCards = (hideOtherCards + 1)%2
    updatePlayerHand(mainGame.currentHands[1],1)
    updatePlayerHand(mainGame.currentHands[2],2)
    updatePlayerHand(mainGame.currentHands[3],3)
    updatePlayerHand(mainGame.currentHands[4],4)
    
def playSelectedOption():
    index = int(listBox.curselection()[0])
    option = availableOptions[index]
    if isinstance(option,int):
        if option == -1:
            mainGame.updateGame(-1)
        else:
            print("something wrong with this option. It's an integer not equal to -1")
    else:
        # option is a list of card IDs (1..52)
        mainGame.updateGame(option, len(option))
            
    updateScreen()
    
def playSampledOption():
    global currSampledOption
    if currSampledOption == -1:
        return
    else:
        mainGame.step(currSampledOption)
        updateScreen()
            
def updateScreen():
    global availableOptions
    global control
    global currSampledOption
    availableOptions = updateOptions()
    updatePlayerHand(mainGame.currentHands[1],1)
    updatePlayerHand(mainGame.currentHands[2],2)
    updatePlayerHand(mainGame.currentHands[3],3)
    updatePlayerHand(mainGame.currentHands[4],4)
    updatePrevHands()
    control = mainGame.control
    movePlayerCircle(mainGame.playersGo)
    for i in range(5):
        currentOption[i].config(image='')
    updateValue()
    updateModelInsights()
    probNegLogValue.set("")
    sampledOptionValue.set("")
    currSampledOption=-1
    #index = int(nnPlayerSelect.curselection()[0])
    #value = nnPlayerSelect.get(index)
    #if value == "Player 1":
    #    updateNeuralNetwork(1)
    #elif value == "Player 2":
    #    updateNeuralNetwork(2)
    #elif value == "Player 3":
    #    updateNeuralNetwork(3)
    #elif value == "Player 4":
    #    updateNeuralNetwork(4)
    
def updateValue():
    valueValue.set("N/A")


def _action_to_string(action):
    if action == enumerateOptions.passInd:
        return "pass"
    cards, nC = enumerateOptions.getOptionNC(action)
    if nC == 0:
        return "pass"
    return " ".join(_card_label(int(c)) for c in cards)


def _card_label(card_id):
    value = int(np.ceil(card_id / 4.0))
    if value <= 7:
        rank = str(value + 2)
    elif value == 8:
        rank = "10"
    elif value == 9:
        rank = "J"
    elif value == 10:
        rank = "Q"
    elif value == 11:
        rank = "K"
    elif value == 12:
        rank = "A"
    else:
        rank = "2"
    suit_idx = (card_id - 1) % 4
    suit = ["C", "D", "H", "S"][suit_idx]
    return f"{rank}{suit}"


def updateModelInsights():
    if belief_model is None or policy_value_model is None:
        modelInfoValue.set("Model: not loaded")
        beliefInfoValue.set("")
        policyInfoValue.set("")
        policyRecValue.set("")
        for i in range(beliefTopN):
            beliefP2Cards[i].config(image='')
            beliefP3Cards[i].config(image='')
            beliefP4Cards[i].config(image='')
        return
    if mainGame.playersGo != 1:
        modelInfoValue.set("Model: waiting (player1 only)")
        beliefInfoValue.set("")
        policyInfoValue.set("")
        policyRecValue.set("")
        for i in range(beliefTopN):
            beliefP2Cards[i].config(image='')
            beliefP3Cards[i].config(image='')
            beliefP4Cards[i].config(image='')
        return
    player = 1
    belief_in = encode_belief_input(mainGame, player)
    with torch.no_grad():
        b_in = torch.from_numpy(belief_in).float().unsqueeze(0).to(DEVICE)
        b_logits = belief_model(b_in)
        b_probs = torch.softmax(b_logits, dim=-1).cpu().numpy()[0]

    targets, mask = belief_targets(mainGame, player)
    valid = mask > 0
    if valid.sum() > 0:
        top2 = np.partition(b_probs, -2, axis=1)
        margins = top2[:, -1] - top2[:, -2]
        valid_m = margins[valid]
        unknown_rate = float((valid_m < 0.5).mean())
        known_rate = 1.0 - unknown_rate
        avg_margin = float(valid_m.mean())
        known_mask = valid & (margins >= 0.5)
        if known_mask.sum() > 0:
            preds = np.argmax(b_probs, axis=1)
            acc_known = float((preds[known_mask] == targets[known_mask]).mean())
        else:
            acc_known = 0.0
    else:
        unknown_rate = known_rate = avg_margin = acc_known = 0.0

    beliefInfoValue.set(
        f"belief: acc_known={acc_known:.2f} known_rate={known_rate:.2f} avg_margin={avg_margin:.2f}"
    )

    # show believed cards for opponents (as card images) among unknown cards
    valid_idx = np.flatnonzero(valid)
    def _top_cards(col):
        if valid_idx.size == 0:
            return []
        scores = b_probs[valid_idx, col]
        top_idx = valid_idx[np.argsort(-scores)][:beliefTopN]
        top_cards = [int(i) + 1 for i in top_idx]
        return sorted(top_cards)

    preds = np.argmax(b_probs, axis=1)
    top2 = np.partition(b_probs, -2, axis=1)
    margins = top2[:, -1] - top2[:, -2]
    known_mask = valid & (margins >= 0.8)
    p2_top = sorted([int(i) + 1 for i in np.flatnonzero(known_mask & (preds == 0))])
    p3_top = sorted([int(i) + 1 for i in np.flatnonzero(known_mask & (preds == 1))])
    p4_top = sorted([int(i) + 1 for i in np.flatnonzero(known_mask & (preds == 2))])

    if checkVar2.get() == 1:
        # P2 (left, vertical) - further left
        sX = -100 + X_OFFSET
        sY = 540
        for i in range(beliefTopN):
            if i < len(p2_top):
                beliefP2Cards[i].config(image=cardImages2[p2_top[i]])
            else:
                beliefP2Cards[i].config(image='')
            beliefP2Cards[i].place(x=sX, y=sY - i * (dX + 2))
            beliefP2Cards[i].lift()

        # P3 (top, horizontal) - further up
        sX = 450 + X_OFFSET
        sY = 140
        for i in range(beliefTopN):
            if i < len(p3_top):
                beliefP3Cards[i].config(image=cardImages3[p3_top[i]])
            else:
                beliefP3Cards[i].config(image='')
            beliefP3Cards[i].place(x=sX + i * (dX + 2), y=sY)
            beliefP3Cards[i].lift()

        # P4 (right, vertical) - further right
        sX = 1150 + X_OFFSET
        sY = 540
        for i in range(beliefTopN):
            if i < len(p4_top):
                beliefP4Cards[i].config(image=cardImages4[p4_top[i]])
            else:
                beliefP4Cards[i].config(image='')
            beliefP4Cards[i].place(x=sX, y=sY - i * (dX + 2))
            beliefP4Cards[i].lift()
    else:
        for i in range(beliefTopN):
            beliefP2Cards[i].config(image='')
            beliefP3Cards[i].config(image='')
            beliefP4Cards[i].config(image='')

    if POLICY_IS_POLICY_ONLY:
        policy_in = encode_belief_input(mainGame, player)
    else:
        policy_in = encode_full_info_with_belief(mainGame, player, b_probs)
    action_mask = np.isfinite(
        big2Game.convertAvailableActions(mainGame.returnAvailableActions().astype(np.float32))
    ).astype(np.float32)
    with torch.no_grad():
        p_in = torch.from_numpy(policy_in).float().unsqueeze(0).to(DEVICE)
        mask_t = torch.from_numpy(action_mask).float().unsqueeze(0).to(DEVICE)
        logits, value = policy_value_model(p_in, mask_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        value = float(value.cpu().numpy()[0])
    valid_actions = np.flatnonzero(action_mask == 1)
    if valid_actions.size == 0:
        policyInfoValue.set("policy: no valid actions")
        policyRecValue.set("")
    else:
        topk = valid_actions[np.argsort(-probs[valid_actions])][:5]
        top_lines = [f"{_action_to_string(int(a))} ({probs[int(a)]:.2f})" for a in topk]
        policyInfoValue.set("policy top5: " + " | ".join(top_lines))
        best = int(topk[0])
        policyRecValue.set(f"Policy recommend: {_action_to_string(best)} ({probs[best]:.2f})")
    modelInfoValue.set(f"value={value:.2f}")
    
def updateProbNegLog(index):
    probNegLogValue.set("N/A")
        
    
def updateOptions():
    n = listBox.size()
    listBox.delete(0,n)
    options = []
    availActions = mainGame.returnAvailableActions()
    for i in range(len(availActions)):
        if availActions[i] == 1:
            (cards, nC) = enumerateOptions.getOptionNC(i)
            if nC == 0:
                options.append(-1)
            else:
                options.append(np.array(cards))
    for i in range(len(options)):
        if isinstance(options[i],int):
            string = "pass"
            listBox.insert(i,string)
        else:
            string = ""
            for k in range(len(options[i])):
                string = string + _card_label(int(options[i][k])) + " "
            listBox.insert(i,string)
    return options

def sampleFromNetwork():
    global currSampledOption
    
    available_actions = mainGame.returnAvailableActions()
    valid_actions = np.flatnonzero(available_actions == 1)
    if valid_actions.size == 0:
        currSampledOption = -1
        sampledOptionValue.set("")
        return
    currSampledOption = int(np.random.choice(valid_actions))
    if currSampledOption == enumerateOptions.passInd:
        sampledOptionValue.set("pass")
    else:
        option, nC = enumerateOptions.getOptionNC(currSampledOption)
        optString = ""
        for i in range(len(option)):
            optString = optString + _card_label(int(option[i])) + " "
        sampledOptionValue.set(str(optString))

def onOptionSelect(evt):
    w = evt.widget
    index = int(w.curselection()[0])
    value = w.get(index)
    if value == "pass":
        updateCurrentOption(1,1)
        finalInd = enumerateOptions.passInd
    else:
        hand = availableOptions[index]
        updateCurrentOption(hand)
        finalInd = enumerateOptions.action_index_from_cards(hand)
    updateProbNegLog(finalInd)
    
        
def onPlayerSelect(evt):
    w = evt.widget
    index = int(w.curselection()[0])
    value = w.get(index)
    if value == "Player 1":
        updateNeuralNetwork(1)
    elif value == "Player 2":
        updateNeuralNetwork(2)
    elif value == "Player 3":
        updateNeuralNetwork(3)
    elif value == "Player 4":
        updateNeuralNetwork(4)
    
def movePlayerCircle(player):
    if control == 1:
        plyrsGoCircle.itemconfigure(circ,fill="green")
    else:
        plyrsGoCircle.itemconfigure(circ,fill="red")
    if player == 1:
        plyrsGoCircle.place(x=540 + X_OFFSET, y=615)
    elif player == 2:
        plyrsGoCircle.place(x=170 + X_OFFSET, y=510)
    elif player == 3:
        plyrsGoCircle.place(x=540 + X_OFFSET, y=375)
    else:
        plyrsGoCircle.place(x=955 + X_OFFSET, y=510)
        
def updateNeuralNetwork(player):
    #update player's cards.
    for i in range(1,23):
        for j in range(1,14):
            index = (j-1)*22 + (i-1)
            NNPlayerCardLabels[(i,j)].config(text=str(mainGame.neuralNetworkInputs[player][index]))
    #next player
    npInd = 22*13
    for i in range(1,14):
        index = npInd + (i-1)
        nextPlayerNCards[i].config(text=str(mainGame.neuralNetworkInputs[player][index]))
    npInd2 = npInd + 13
    for i in range(1,15):
        index = npInd2 + (i-1)
        nextPlayerHasPlayed[i].config(text=str(mainGame.neuralNetworkInputs[player][index]))
    #next next player
    nnpInd = npInd + 27
    for i in range(1,14):
        index = nnpInd + (i-1)
        nextNextPlayerNCards[i].config(text=str(mainGame.neuralNetworkInputs[player][index]))
    nnpInd2 = nnpInd + 13
    for i in range(1,15):
        index = nnpInd2 + (i-1)
        nextNextPlayerHasPlayed[i].config(text=str(mainGame.neuralNetworkInputs[player][index]))
    # next next next player
    nnnpInd = nnpInd + 27
    for i in range(1,14):
        index = nnnpInd + (i-1)
        nextNextNextPlayerNCards[i].config(text=str(mainGame.neuralNetworkInputs[player][index]))
    nnnpInd2 = nnnpInd + 13
    for i in range(1,15):
        index = nnnpInd2 + (i-1)
        nextNextNextPlayerHasPlayed[i].config(text=str(mainGame.neuralNetworkInputs[player][index]))
    sInd = nnnpInd + 27
    for i in range(1,5):
        for j in range(1,5):
            index = sInd + 4*(j-1) + (i-1)
            sharedCards[(i,j)].config(text=str(mainGame.neuralNetworkInputs[player][index]))
    #previous hand
    fInd = sInd + 16
    for i in range(1,18):
        prevValue[i].config(text=str(mainGame.neuralNetworkInputs[player][fInd+i-1]))
    fInd = fInd + 17
    for i in range(1,13):
        prevType[i].config(text=str(mainGame.neuralNetworkInputs[player][fInd+i-1]))
    
    
xv = 510 + X_OFFSET
yv = 60
xhd = 100
valueTitle = tkinter.StringVar()
valueTitle.set("State Value:")
valueTitleLabel = tkinter.Label(top, textvariable=valueTitle, font=("Helvetica",12))
valueTitleLabel.place(x=xv,y=yv)
valueValue = tkinter.StringVar()
valueValueLabel = tkinter.Label(top, textvariable=valueValue, font=("Helvetica",12))
valueValueLabel.place(x=xv+xhd,y=yv)

ndd = 25
probNeglogTitle = tkinter.StringVar()
probNeglogTitle.set("Option Info:")
probNeglogLabel = tkinter.Label(top, textvariable=probNeglogTitle, font=("Helvetica",12))
probNeglogLabel.place(x=xv, y=yv+ndd)
probNegLogValue = tkinter.StringVar()
probNegLogValueLabel = tkinter.Label(top, textvariable=probNegLogValue, font=("Helvetica",12))
probNegLogValueLabel.place(x=xv+200, y=yv+ndd)

modelInfoValue = tkinter.StringVar()
modelInfoValueLabel = tkinter.Label(top, textvariable=modelInfoValue, font=("Helvetica",12))
modelInfoValueLabel.place(x=160 + X_OFFSET, y=800)
beliefInfoValue = tkinter.StringVar()
beliefInfoValueLabel = tkinter.Label(top, textvariable=beliefInfoValue, font=("Helvetica",10))
beliefInfoValueLabel.place(x=160 + X_OFFSET, y=825)
policyInfoValue = tkinter.StringVar()
policyInfoValueLabel = tkinter.Label(top, textvariable=policyInfoValue, font=("Helvetica",10))
policyInfoValueLabel.place(x=160 + X_OFFSET, y=850)
policyRecValue = tkinter.StringVar()
policyRecLabel = tkinter.Label(top, textvariable=policyRecValue, font=("Helvetica",12))
policyRecLabel.place(x=160 + X_OFFSET, y=875)

plyrsGoCircle = tkinter.Canvas(top, height=40,width=40)
circ = plyrsGoCircle.create_oval(10,10,30,30,fill="red")

sampleButton = tkinter.Button(top, text="Sample Random Option", command=sampleFromNetwork, height=1, width=27)
sampleButton.place(x=xv + 50, y=yv+5*ndd-20)

sampledOptionTitle = tkinter.StringVar()
sampledOptionTitle.set("Sampled Option:")
sampledOptionLabel = tkinter.Label(top, textvariable=sampledOptionTitle, font=("Helvetica",12))
sampledOptionLabel.place(x=xv,y=yv+6*ndd-15)
sampledOptionValue = tkinter.StringVar()
sampledOptionValueLabel = tkinter.Label(top, textvariable=sampledOptionValue, font=("Helvetica",12))
sampledOptionValueLabel.place(x=xv+120, y=yv+6*ndd-15)

playSampledOptionButton = tkinter.Button(top, text="Play Sampled Option", command=playSampledOption, height=1, width=20)
playSampledOptionButton.place(x=xv+50, y=yv+7*ndd-15)

movePlayerCircle(mainGame.playersGo)

checkVar1 = tkinter.IntVar()
C1 = tkinter.Checkbutton(
    top, text="Show Other Hands", variable=checkVar1, command=changeShowHands
)
C1.place(x=160 + X_OFFSET, y=740)

checkVar2 = tkinter.IntVar(value=1)
def toggleShowPredictions():
    updateModelInsights()

C2 = tkinter.Checkbutton(
    top,
    text="Show Prediction of Other Hands",
    variable=checkVar2,
    command=toggleShowPredictions,
)
C2.place(x=160 + X_OFFSET, y=770)

playOptionButton = tkinter.Button(top, text="Play Selected Option", command=playSelectedOption, height=1, width=18)
playOptionButton.place(x=1050 + X_OFFSET, y=70)

p1text = tkinter.StringVar()
p2text = tkinter.StringVar()
p3text = tkinter.StringVar()
p4text = tkinter.StringVar()
p1Label = tkinter.Label(top, textvariable=p1text, font=("Helvetica",16))
p2Label = tkinter.Label(top, textvariable=p2text, font=("Helvetica",16))
p3Label = tkinter.Label(top, textvariable=p3text, font=("Helvetica",16))
p4Label = tkinter.Label(top, textvariable=p4text, font=("Helvetica",16))
p1text.set("Player 1")
p2text.set("Player 2")
p3text.set("Player 3")
p4text.set("Player 4")
p1Label.place(x=525 + X_OFFSET, y=650)
p2Label.place(x=80 + X_OFFSET, y=365)
p3Label.place(x=525 + X_OFFSET, y=250)
p4Label.place(x=1010 + X_OFFSET, y=365)

# belief card placeholders (top-N)
beliefTopN = 8
beliefP2Cards = {}
beliefP3Cards = {}
beliefP4Cards = {}
for i in range(beliefTopN):
    beliefP2Cards[i] = tkinter.Label(top)
    beliefP3Cards[i] = tkinter.Label(top)
    beliefP4Cards[i] = tkinter.Label(top)

cOptionsText = tkinter.StringVar()
cOptionsLabel = tkinter.Label(top, textvariable=cOptionsText, font=("Helvetica",16))
cOptionsText.set("Current Options")
cOptionsLabel.place(x=895 + X_OFFSET, y=31)

listFrame = tkinter.Frame(top, width=150, height=100)
listFrame.place(x=850 + X_OFFSET, y=75)

scrollbar = tkinter.Scrollbar(listFrame)
scrollbar.pack(side=tkinter.RIGHT,fill=tkinter.Y)

listBox = tkinter.Listbox(listFrame, height=10)
listBox.pack(side=tkinter.LEFT, fill=tkinter.Y)
scrollbar.config(command=listBox.yview)
listBox.config(yscrollcommand=scrollbar.set)
listBox.bind('<<ListboxSelect>>',onOptionSelect)

NNWindow = tkinter.Toplevel(top, width=1200,height=450)
NNWindow.title("Game State Input")
NNFrame1 = tkinter.Frame(NNWindow)
NNFrame1.place(x=10,y=40)
fontsize=6
topNNLabels={}
sideNNLabels={}
for i in range(1,14):
    topNNLabels[i] = tkinter.Label(NNFrame1,text=("C"+str(i)),anchor="center",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=0,column=i,sticky=tkinter.S)
for i in range(1,23):
    if i<=8:
        string = str(i+2)
    elif i==9:
        string = "J"
    elif i==10:
        string = "Q"
    elif i==11:
        string = "K"
    elif i==12:
        string = "A"
    elif i==13:
        string = "2"
    elif i==14:
        string = "D"
    elif i==15:
        string = "C"
    elif i==16:
        string = "H"
    elif i==17:
        string = "S"
    elif i==18:
        string = "inPair"
    elif i==19:
        string = "inThree"
    elif i==20:
        string = "inFour"
    elif i==21:
        string = "inStraight"
    elif i==22:
        string = "inFlush"
    sideNNLabels[i] = tkinter.Label(NNFrame1,text=string,font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=i,column=0,sticky=tkinter.E)
    
NNCardLabel = tkinter.Label(NNWindow, text="Current Player's Cards", font=("Helvetica",16))
NNCardLabel.place(x=50,y=5)
NNPlayerCardLabels = {}
for i in range(1,23):
    for j in range(1,14):
        NNPlayerCardLabels[(i,j)] = tkinter.Label(NNFrame1,text="0", font=("Helvetica",fontsize))
        NNPlayerCardLabels[(i,j)].grid(row=i, column=j)
#import pdb; pdb.set_trace()
NNNextPlayerCardLabel = tkinter.Label(NNWindow, text="Next Player's Cards", font=("Helvetica",16))
NNNextPlayerCardLabel.place(x=330,y=5)

NNFrame2 = tkinter.Frame(NNWindow)
NNFrame2.place(x=330,y=40)
NextPlayerLeftLabel = {}
for i in range(1,14):
     NextPlayerLeftLabel[i] = tkinter.Label(NNFrame2,text=str(i),font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=i,column=0,sticky=tkinter.E)
NextPlayerTopLabel = {}
NextPlayerTopLabel[1] =  tkinter.Label(NNFrame2,text="nCards",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=0,column=1,sticky=tkinter.S)
NextPlayerTopLabel[2] =  tkinter.Label(NNFrame2,text="hasPlayed",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=0,column=2,sticky=tkinter.S)
NextPlayerRightLabel = {}
for i in range(1,15):
    if i==1:
        string = "AD"
    elif i==2:
        string = "AC"
    elif i==3:
        string = "AH"
    elif i==4:
        string = "AS"
    elif i==5:
        string = "2D"
    elif i==6:
        string = "2C"
    elif i==7:
        string = "2H"
    elif i==8:
        string = "2S"
    elif i==9:
        string = "playedPair"
    elif i==10:
        string = "playedThree"
    elif i==11:
        string = "playedTwoPair"
    elif i==12:
        string = "playedStraight"
    elif i==13:
        string = "playedFlush"
    else:
        string = "playedFullHouse"
    NextPlayerRightLabel[i] =  tkinter.Label(NNFrame2,text=string,font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=i,column=3,sticky=tkinter.W)
nextPlayerNCards = {}
nextPlayerHasPlayed = {}
for i in range(1,14):
    nextPlayerNCards[i] = tkinter.Label(NNFrame2, text="0",font=("Helvetica",fontsize))
    nextPlayerNCards[i].grid(row=i,column=1)
for i in range(1,15):
    nextPlayerHasPlayed[i] = tkinter.Label(NNFrame2, text="0",font=("Helvetica",fontsize))
    nextPlayerHasPlayed[i].grid(row=i,column=2)
    
    
#NEXT NEXT Player
NNNextNextPlayerCardLabel = tkinter.Label(NNWindow, text="Next^2 Player's Cards", font=("Helvetica",16))
NNNextNextPlayerCardLabel.place(x=540,y=5)

NNFrame3 = tkinter.Frame(NNWindow)
NNFrame3.place(x=540,y=40)
NextNextPlayerLeftLabel = {}
for i in range(1,14):
     NextNextPlayerLeftLabel[i] = tkinter.Label(NNFrame3,text=str(i),font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=i,column=0,sticky=tkinter.E)
NextNextPlayerTopLabel = {}
NextNextPlayerTopLabel[1] =  tkinter.Label(NNFrame3,text="nCards",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=0,column=1,sticky=tkinter.S)
NextNextPlayerTopLabel[2] =  tkinter.Label(NNFrame3,text="hasPlayed",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=0,column=2,sticky=tkinter.S)
NextNextPlayerRightLabel = {}
for i in range(1,15):
    if i==1:
        string = "AD"
    elif i==2:
        string = "AC"
    elif i==3:
        string = "AH"
    elif i==4:
        string = "AS"
    elif i==5:
        string = "2D"
    elif i==6:
        string = "2C"
    elif i==7:
        string = "2H"
    elif i==8:
        string = "2S"
    elif i==9:
        string = "playedPair"
    elif i==10:
        string = "playedThree"
    elif i==11:
        string = "playedTwoPair"
    elif i==12:
        string = "playedStraight"
    elif i==13:
        string = "playedFlush"
    else:
        string = "playedFullHouse"
    NextNextPlayerRightLabel[i] =  tkinter.Label(NNFrame3,text=string,font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=i,column=3,sticky=tkinter.W)
nextNextPlayerNCards = {}
nextNextPlayerHasPlayed = {}
for i in range(1,14):
    nextNextPlayerNCards[i] = tkinter.Label(NNFrame3, text="0",font=("Helvetica",fontsize))
    nextNextPlayerNCards[i].grid(row=i,column=1)
for i in range(1,15):
    nextNextPlayerHasPlayed[i] = tkinter.Label(NNFrame3, text="0",font=("Helvetica",fontsize))
    nextNextPlayerHasPlayed[i].grid(row=i,column=2)

#NEXT NEXT NEXT Player
NNNextNextNextPlayerCardLabel = tkinter.Label(NNWindow, text="Next^3 Player's Cards", font=("Helvetica",16))
NNNextNextNextPlayerCardLabel.place(x=775,y=5)

NNFrame4 = tkinter.Frame(NNWindow)
NNFrame4.place(x=775,y=40)
NextNextNextPlayerLeftLabel = {}
for i in range(1,14):
     NextNextPlayerLeftLabel[i] = tkinter.Label(NNFrame4,text=str(i),font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=i,column=0,sticky=tkinter.E)
NextNextNextPlayerTopLabel = {}
NextNextNextPlayerTopLabel[1] =  tkinter.Label(NNFrame4,text="nCards",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=0,column=1,sticky=tkinter.S)
NextNextNextPlayerTopLabel[2] =  tkinter.Label(NNFrame4,text="hasPlayed",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=0,column=2,sticky=tkinter.S)
NextNextNextPlayerRightLabel = {}
for i in range(1,15):
    if i==1:
        string = "AD"
    elif i==2:
        string = "AC"
    elif i==3:
        string = "AH"
    elif i==4:
        string = "AS"
    elif i==5:
        string = "2D"
    elif i==6:
        string = "2C"
    elif i==7:
        string = "2H"
    elif i==8:
        string = "2S"
    elif i==9:
        string = "playedPair"
    elif i==10:
        string = "playedThree"
    elif i==11:
        string = "playedTwoPair"
    elif i==12:
        string = "playedStraight"
    elif i==13:
        string = "playedFlush"
    else:
        string = "playedFullHouse"
    NextNextNextPlayerRightLabel[i] =  tkinter.Label(NNFrame4,text=string,font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=i,column=3,sticky=tkinter.W)
nextNextNextPlayerNCards = {}
nextNextNextPlayerHasPlayed = {}
for i in range(1,14):
    nextNextNextPlayerNCards[i] = tkinter.Label(NNFrame4, text="0",font=("Helvetica",fontsize))
    nextNextNextPlayerNCards[i].grid(row=i,column=1)
for i in range(1,15):
    nextNextNextPlayerHasPlayed[i] = tkinter.Label(NNFrame4, text="0",font=("Helvetica",fontsize))
    nextNextNextPlayerHasPlayed[i].grid(row=i,column=2)

#SHARED MEMORY
sharedLabel = tkinter.Label(NNWindow, text="Cards Played", font=("Helvetica",16))
sharedLabel.place(x=400,y=300)
NNFrame5 = tkinter.Frame(NNWindow)
NNFrame5.place(x=450,y=335)
sharedTopLabel = {}
sharedTopLabel[1] =  tkinter.Label(NNFrame5,text="Q",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=0,column=1,sticky=tkinter.S)
sharedTopLabel[2] =  tkinter.Label(NNFrame5,text="K",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=0,column=2,sticky=tkinter.S)
sharedTopLabel[3] =  tkinter.Label(NNFrame5,text="A",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=0,column=3,sticky=tkinter.S)
sharedTopLabel[4] =  tkinter.Label(NNFrame5,text="2",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=0,column=4,sticky=tkinter.S)
sharedLeftLabel = {}
sharedLeftLabel[1] = tkinter.Label(NNFrame5,text="D",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=1,column=0,sticky=tkinter.E)
sharedLeftLabel[2] = tkinter.Label(NNFrame5,text="C",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=2,column=0,sticky=tkinter.E)
sharedLeftLabel[3] = tkinter.Label(NNFrame5,text="H",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=3,column=0,sticky=tkinter.E)
sharedLeftLabel[4] = tkinter.Label(NNFrame5,text="S",font=("Helvetica",fontsize),relief=tkinter.RIDGE).grid(row=4,column=0,sticky=tkinter.E)

sharedCards = {}
for i in range(1,5):
    for j in range(1,5):
        sharedCards[(i,j)] = tkinter.Label(NNFrame5,text="0",font=("Helvetica",fontsize))
        sharedCards[(i,j)].grid(row=i,column=j)
        
#previous hand in neural network.
prevHandLavel = tkinter.Label(NNWindow, text="Previous Hand", font=("Helvetica",16))
prevHandLavel.place(x=1010,y=10)
NNFrame6 = tkinter.Frame(NNWindow)
NNFrame6.place(x=1010,y=40)
prevTopLabel = {}
prevTopLabel[1] = tkinter.Label(NNFrame6,text="highVal", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=0, column=1,sticky=tkinter.S)
prevTopLabel[2] = tkinter.Label(NNFrame6,text="Type", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=0, column=2,sticky=tkinter.S)
prevLeftLabel = {}
prevLeftLabel[1] = tkinter.Label(NNFrame6, text="3", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=1,column=0,sticky=tkinter.E)
prevLeftLabel[2] = tkinter.Label(NNFrame6, text="4", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=2,column=0,sticky=tkinter.E)
prevLeftLabel[3] = tkinter.Label(NNFrame6, text="5", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=3,column=0,sticky=tkinter.E)
prevLeftLabel[4] = tkinter.Label(NNFrame6, text="6", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=4,column=0,sticky=tkinter.E)
prevLeftLabel[5] = tkinter.Label(NNFrame6, text="7", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=5,column=0,sticky=tkinter.E)
prevLeftLabel[6] = tkinter.Label(NNFrame6, text="8", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=6,column=0,sticky=tkinter.E)
prevLeftLabel[7] = tkinter.Label(NNFrame6, text="9", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=7,column=0,sticky=tkinter.E)
prevLeftLabel[8] = tkinter.Label(NNFrame6, text="10", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=8,column=0,sticky=tkinter.E)
prevLeftLabel[9] = tkinter.Label(NNFrame6, text="J", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=9,column=0,sticky=tkinter.E)
prevLeftLabel[10] = tkinter.Label(NNFrame6, text="Q", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=10,column=0,sticky=tkinter.E)
prevLeftLabel[11] = tkinter.Label(NNFrame6, text="K", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=11,column=0,sticky=tkinter.E)
prevLeftLabel[12] = tkinter.Label(NNFrame6, text="A", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=12,column=0,sticky=tkinter.E)
prevLeftLabel[13] = tkinter.Label(NNFrame6, text="2", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=13,column=0,sticky=tkinter.E)
prevLeftLabel[14] = tkinter.Label(NNFrame6, text="D", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=14,column=0,sticky=tkinter.E)
prevLeftLabel[15] = tkinter.Label(NNFrame6, text="C", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=15,column=0,sticky=tkinter.E)
prevLeftLabel[16] = tkinter.Label(NNFrame6, text="H", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=16,column=0,sticky=tkinter.E)
prevLeftLabel[17] = tkinter.Label(NNFrame6, text="S", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=17,column=0,sticky=tkinter.E)
prevRightLabel = {}
prevRightLabel[1] = tkinter.Label(NNFrame6, text="Control", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=1,column=3,sticky=tkinter.W)
prevRightLabel[2] = tkinter.Label(NNFrame6, text="Single Card", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=2,column=3,sticky=tkinter.W)
prevRightLabel[3] = tkinter.Label(NNFrame6, text="Pair", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=3,column=3,sticky=tkinter.W)
prevRightLabel[4] = tkinter.Label(NNFrame6, text="Three", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=4,column=3,sticky=tkinter.W)
prevRightLabel[5] = tkinter.Label(NNFrame6, text="Two Pair", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=5,column=3,sticky=tkinter.W)
prevRightLabel[6] = tkinter.Label(NNFrame6, text="Four Of A Kind", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=6,column=3,sticky=tkinter.W)
prevRightLabel[7] = tkinter.Label(NNFrame6, text="Straight", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=7,column=3,sticky=tkinter.W)
prevRightLabel[8] = tkinter.Label(NNFrame6, text="Flush", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=8,column=3,sticky=tkinter.W)
prevRightLabel[9] = tkinter.Label(NNFrame6, text="Full House", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=9,column=3,sticky=tkinter.W)
prevRightLabel[10] = tkinter.Label(NNFrame6, text="No passes", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=10,column=3,sticky=tkinter.W)
prevRightLabel[11] = tkinter.Label(NNFrame6, text="One Pass", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=11,column=3,sticky=tkinter.W)
prevRightLabel[12] = tkinter.Label(NNFrame6, text="Two Pass", font=("Helvetica",fontsize), relief=tkinter.RIDGE).grid(row=12,column=3,sticky=tkinter.W)


prevValue = {}
for i in range(1,18):
    prevValue[i] = tkinter.Label(NNFrame6, text="0", font=("Helvetica",fontsize))
    prevValue[i].grid(row=i,column=1)
prevType = {}
for i in range(1,13):
    prevType[i] = tkinter.Label(NNFrame6, text="0", font=("Helvetica",fontsize))
    prevType[i].grid(row=i,column=2)

#player selection in neural network input
nnPlayerSelect = tkinter.Listbox(NNWindow, height=4)
nnPlayerSelect.insert(0,"Player 1")
nnPlayerSelect.insert(1,"Player 2")
nnPlayerSelect.insert(2,"Player 3")
nnPlayerSelect.insert(3,"Player 4")
nnPlayerSelect.place(x=700,y=330)
nnPlayerSelect.bind('<<ListboxSelect>>',onPlayerSelect)


updatePlayerHand(mainGame.currentHands[1],1)
updatePlayerHand(mainGame.currentHands[2],2)
updatePlayerHand(mainGame.currentHands[3],3)
updatePlayerHand(mainGame.currentHands[4],4)
updateNeuralNetwork(mainGame.playersGo)
availableOptions = updateOptions()

updatePrevHands()
updateValue()

top.mainloop()
