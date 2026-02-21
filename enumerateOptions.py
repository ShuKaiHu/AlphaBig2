#enumerate options
import pickle
import warnings
import itertools
import numpy as np
import gameLogic


def _card_id(rank_value, suit_idx):
    # rank_value: 1-13 (3..2), suit_idx: 1-4 (C,D,H,S)
    return (rank_value - 1) * 4 + suit_idx


SINGLE_ACTIONS = list(range(1, 53))
SINGLE_INDEX = {cid: idx for idx, cid in enumerate(SINGLE_ACTIONS)}

PAIR_ACTIONS = []
for r in range(1, 14):
    cards = [_card_id(r, s) for s in range(1, 5)]
    for i in range(4):
        for j in range(i + 1, 4):
            PAIR_ACTIONS.append((cards[i], cards[j]))
PAIR_INDEX = {tuple(sorted(c)): idx for idx, c in enumerate(PAIR_ACTIONS)}

# Five-card legal actions: straight, full house, four of a kind, straight flush
FIVE_ACTIONS_SET = set()

# Straights (include straight flush; will be de-duplicated)
for seq in gameLogic._STRAIGHT_SEQUENCES:
    card_choices = [[_card_id(v, s) for s in range(1, 5)] for v in seq]
    for combo in itertools.product(*card_choices):
        hand = tuple(sorted(combo))
        if gameLogic.isStraight(np.array(hand)):
            FIVE_ACTIONS_SET.add(hand)

# Full house
for three_val in range(1, 14):
    for pair_val in range(1, 14):
        if pair_val == three_val:
            continue
        three_cards = [_card_id(three_val, s) for s in range(1, 5)]
        pair_cards = [_card_id(pair_val, s) for s in range(1, 5)]
        for three_combo in itertools.combinations(three_cards, 3):
            for pair_combo in itertools.combinations(pair_cards, 2):
                hand = tuple(sorted(three_combo + pair_combo))
                FIVE_ACTIONS_SET.add(hand)

# Four of a kind
for quad_val in range(1, 14):
    quad_cards = [_card_id(quad_val, s) for s in range(1, 5)]
    for kicker_val in range(1, 14):
        if kicker_val == quad_val:
            continue
        for s in range(1, 5):
            hand = tuple(sorted(quad_cards + [_card_id(kicker_val, s)]))
            FIVE_ACTIONS_SET.add(hand)

# Straight flush
for seq in gameLogic._STRAIGHT_SEQUENCES:
    for s in range(1, 5):
        hand = tuple(sorted(_card_id(v, s) for v in seq))
        FIVE_ACTIONS_SET.add(hand)

FIVE_ACTIONS = sorted(FIVE_ACTIONS_SET)
FIVE_INDEX = {tuple(c): idx for idx, c in enumerate(FIVE_ACTIONS)}

N_SINGLE = len(SINGLE_ACTIONS)
N_PAIR = len(PAIR_ACTIONS)
N_FIVE = len(FIVE_ACTIONS)

PAIR_OFFSET = N_SINGLE
FIVE_OFFSET = N_SINGLE + N_PAIR
passInd = N_SINGLE + N_PAIR + N_FIVE

nActions = np.array([N_SINGLE, N_PAIR, 0, 0, N_FIVE, passInd])
nAcSum = np.cumsum(nActions[:-1])

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"dtype\(\): align should be passed.*",
        category=Warning,
    )
    with open('actionIndices.pkl','rb') as f:  # Python 3: open(..., 'rb')
        twoCardIndices, threeCardIndices, fourCardIndices, fiveCardIndices, inverseTwoCardIndices, inverseThreeCardIndices, inverseFourCardIndices, inverseFiveCardIndices = pickle.load(f)

def action_index_from_cards(cards):
    cards = tuple(sorted(int(c) for c in cards))
    if len(cards) == 1:
        return SINGLE_INDEX[cards[0]]
    if len(cards) == 2:
        return PAIR_OFFSET + PAIR_INDEX[cards]
    if len(cards) == 5:
        return FIVE_OFFSET + FIVE_INDEX[cards]
    raise ValueError(f"unsupported cards length: {len(cards)}")


def getIndex(option, nCards):
    if nCards == 0:
        return passInd
    if isinstance(option, (list, tuple, np.ndarray)):
        return action_index_from_cards(option)
    if nCards == 1:
        return action_index_from_cards([int(option)])
    raise ValueError("getIndex expects cards list for nCards>1")

def getOptionNC(ind):
    if ind == passInd:
        return -1, 0
    if ind < PAIR_OFFSET:
        return [SINGLE_ACTIONS[ind]], 1
    if ind < FIVE_OFFSET:
        return list(PAIR_ACTIONS[ind - PAIR_OFFSET]), 2
    return list(FIVE_ACTIONS[ind - FIVE_OFFSET]), 5

def fiveCardOptions(handOptions, prevHand=[],prevType=0):
    #prevType = 0 - no hand, you have control and can play any 5 card
    #         = 1 - straight
    #         = 2 - full house
    #         = 3 - four of a kind
    #         = 4 - straight flush
        
    validInds = np.zeros((nActions[4],),dtype=int)
    c = 0
    cardInds = np.zeros((5,),dtype=int) #reuse

    # When following, only allow the same 5-card type here.
    # Overrides (four of a kind / straight flush) are handled elsewhere.
    restrict_to_type = prevType if prevType in (1, 2, 3, 4) else 0
    
    #first deal with straights
    if restrict_to_type and restrict_to_type != 1:
        pass
    else:
        if len(handOptions.straights) > 0:
            for straight in handOptions.straights:
                nC = straight.size
                for i1 in range(nC-4): #first index of hand
                    val1 = handOptions.cards[straight[i1]].value
                    cardInds[0] = handOptions.cards[straight[i1]].indexInHand
                    for i2 in range(i1+1,nC-3):
                        val2 = handOptions.cards[straight[i2]].value
                        if val1 == val2:
                            continue
                        if val2 > val1 + 1:
                            break
                        cardInds[1] = handOptions.cards[straight[i2]].indexInHand
                        for i3 in range(i2+1,nC-2):
                            val3 = handOptions.cards[straight[i3]].value
                            if val3 == val2:
                                continue
                            if val3 > val2 + 1:
                                break
                            cardInds[2] = handOptions.cards[straight[i3]].indexInHand
                            for i4 in range(i3+1,nC-1):
                                val4 = handOptions.cards[straight[i4]].value
                                if val4 == val3:
                                    continue
                                if val4 > val3 + 1:
                                    break
                                cardInds[3] = handOptions.cards[straight[i4]].indexInHand
                                for i5 in range(i4+1, nC):
                                    val5 = handOptions.cards[straight[i5]].value
                                    if val5 == val4:
                                        continue
                                    if val5 > val4 + 1:
                                        break
                                    #import pdb; pdb.set_trace()
                                    cardInds[4] = handOptions.cards[straight[i5]].indexInHand
                                    handBeingPlayed = handOptions.cHand[cardInds]
                                    # Guard against illegal straight sequences.
                                    if not gameLogic.isStraight(handBeingPlayed):
                                        continue
                                    if prevType == 1:
                                        if not gameLogic.compareStraights(handBeingPlayed, prevHand):
                                            continue
                                    validInds[c] = fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                                    c += 1
        #wrap straights: A-2-3-4-5 and 2-3-4-5-6
        def add_wrap_straights(values):
            nonlocal c
            value_cards = []
            for v in values:
                cards = [card_num for card_num in handOptions.cHand if handOptions.cards[card_num].value == v]
                if len(cards) == 0:
                    return
                value_cards.append(cards)
            for combo in itertools.product(*value_cards):
                card_indices = [handOptions.cards[cnum].indexInHand for cnum in combo]
                card_indices.sort()
                handBeingPlayed = handOptions.cHand[card_indices]
                if prevType == 1:
                    if not gameLogic.compareStraights(handBeingPlayed, prevHand):
                        continue
                validInds[c] = fiveCardIndices[card_indices[0]][card_indices[1]][card_indices[2]][card_indices[3]][card_indices[4]]
                c += 1

        add_wrap_straights([12, 13, 1, 2, 3])  # A 2 3 4 5
        add_wrap_straights([13, 1, 2, 3, 4])   # 2 3 4 5 6
    
    #now deal with straight flushes
    if not restrict_to_type or restrict_to_type == 4:
        if len(handOptions.flushes) > 0:
            for flush in handOptions.flushes:
                nC = flush.size
                for i1 in range(nC-4):
                    cardInds[0] = handOptions.cards[flush[i1]].indexInHand
                    for i2 in range(i1+1,nC-3):
                        cardInds[1] = handOptions.cards[flush[i2]].indexInHand
                        for i3 in range(i2+1,nC-2):
                            cardInds[2] = handOptions.cards[flush[i3]].indexInHand
                            for i4 in range(i3+1,nC-1):
                                cardInds[3] = handOptions.cards[flush[i4]].indexInHand
                                for i5 in range(i4+1,nC):
                                    cardInds[4] = handOptions.cards[flush[i5]].indexInHand
                                    handBeingPlayed = handOptions.cHand[cardInds]
                                    if not gameLogic.isStraightFlush(handBeingPlayed):
                                        continue
                                    if prevType == 4:
                                        if not gameLogic.compareStraights(handBeingPlayed, prevHand):
                                            continue
                                    validInds[c] = fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                                    c += 1
    
    #now deal with full houses
    if restrict_to_type and restrict_to_type != 2:
        pass
    else:
        if prevType == 2:
            threeVal = gameLogic.isFullHouse(prevHand)[1]
        nPairs = handOptions.nPairs
        nThree = handOptions.nThreeOfAKinds
        if nPairs > 0 and nThree > 0:
            for pair in handOptions.pairs:
                pVal = handOptions.cards[pair[0]].value
                for three in handOptions.threeOfAKinds:
                    tVal = handOptions.cards[three[0]].value
                    if tVal == pVal:
                        continue
                    if pVal > tVal:
                        cardInds[0] = handOptions.cards[three[0]].indexInHand
                        cardInds[1] = handOptions.cards[three[1]].indexInHand
                        cardInds[2] = handOptions.cards[three[2]].indexInHand
                        cardInds[3] = handOptions.cards[pair[0]].indexInHand
                        cardInds[4] = handOptions.cards[pair[1]].indexInHand
                    else:
                        cardInds[0] = handOptions.cards[pair[0]].indexInHand
                        cardInds[1] = handOptions.cards[pair[1]].indexInHand
                        cardInds[2] = handOptions.cards[three[0]].indexInHand
                        cardInds[3] = handOptions.cards[three[1]].indexInHand
                        cardInds[4] = handOptions.cards[three[2]].indexInHand
                    if prevType == 2:
                        if threeVal > tVal:
                            continue
                    validInds[c] = fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                    c += 1

    #now deal with four of a kinds (with kicker)
    if not restrict_to_type or restrict_to_type == 3:
        quad_value_prev = None
        if prevType == 3:
            quad_value_prev = gameLogic.fourOfAKindValue(prevHand)
        for four in handOptions.fourOfAKinds:
            quad_val = gameLogic.cardValue(four[0])
            if quad_value_prev is not None and quad_val <= quad_value_prev:
                continue
            quad_indices = [handOptions.cards[cnum].indexInHand for cnum in four]
            quad_indices.sort()
            quad_set = set(four)
            for kicker in handOptions.cHand:
                if kicker in quad_set:
                    continue
                cardInds = quad_indices + [handOptions.cards[kicker].indexInHand]
                cardInds.sort()
                validInds[c] = fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1

    

def fourCardOptions(handOptions, prevHand = [], prevType = 0):
    #prevType: 1 - pair, 2 - fourofakind    
    validInds = np.zeros((nActions[3],),dtype=int)
    c = 0
    cardInds = np.zeros((4,),dtype=int) #reuse
    
    #four of a kinds
    if len(handOptions.fourOfAKinds) > 0:
        for four in handOptions.fourOfAKinds:
            cardInds[0] = handOptions.cards[four[0]].indexInHand
            cardInds[1] = handOptions.cards[four[1]].indexInHand
            cardInds[2] = handOptions.cards[four[2]].indexInHand
            cardInds[3] = handOptions.cards[four[3]].indexInHand
            if prevType == 2:
                if handOptions.cHand[cardInds[0]] < prevHand[3]:
                    continue
            validInds[c] = fourCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]]
            c += 1
    #two pairs
    if prevType == 2:
        pass
    else:
        if handOptions.nDistinctPairs >= 2:
            nPairs = handOptions.nPairs
            for p1 in range(nPairs-1):
                p1Val = handOptions.cards[handOptions.pairs[p1][0]].value
                for p2 in range(p1+1,nPairs):
                    p2Val = handOptions.cards[handOptions.pairs[p2][0]].value
                    if p1Val == p2Val:
                        continue
                    cardInds[0] = handOptions.cards[handOptions.pairs[p1][0]].indexInHand
                    cardInds[1] = handOptions.cards[handOptions.pairs[p1][1]].indexInHand
                    cardInds[2] = handOptions.cards[handOptions.pairs[p2][0]].indexInHand
                    cardInds[3] = handOptions.cards[handOptions.pairs[p2][1]].indexInHand
                    if prevType == 1:
                        if handOptions.cHand[cardInds[3]] < prevHand[3]:
                            continue
                    validInds[c] = fourCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]]
                    c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1 #no four card hands

def threeCardOptions(handOptions, prevHand = [], prevType = 0):
    #prevType = 1 - played three of a kind
    validInds = np.zeros((nActions[2],), dtype=int)
    c = 0
    cardInds = np.zeros((3,), dtype=int) #reuse
    
    if handOptions.nThreeOfAKinds > 0:
        for three in handOptions.threeOfAKinds:
            cardInds[0] = handOptions.cards[three[0]].indexInHand
            cardInds[1] = handOptions.cards[three[1]].indexInHand
            cardInds[2] = handOptions.cards[three[2]].indexInHand
            if prevType == 1:
                if handOptions.cHand[cardInds[2]] < prevHand[2]:
                    continue
            validInds[c] = threeCardIndices[cardInds[0]][cardInds[1]][cardInds[2]]
            c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1


def fourOfAKindOnlyOptions(handOptions, prevHand=None):
    if len(handOptions.fourOfAKinds) == 0:
        return -1
    options = []
    prev_quad = None
    if prevHand is not None:
        prev_quad = gameLogic.fourOfAKindValue(prevHand)
    for four in handOptions.fourOfAKinds:
        quad_val = gameLogic.cardValue(four[0])
        if prev_quad is not None and quad_val <= prev_quad:
            continue
        quad_indices = [handOptions.cards[cnum].indexInHand for cnum in four]
        quad_indices.sort()
        quad_set = set(four)
        for kicker in handOptions.cHand:
            if kicker in quad_set:
                continue
            cardInds = quad_indices + [handOptions.cards[kicker].indexInHand]
            cardInds.sort()
            options.append(
                fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
            )
    if len(options) == 0:
        return -1
    return np.array(options, dtype=int)


def straightFlushOnlyOptions(handOptions, prevHand=None):
    if len(handOptions.flushes) == 0:
        return -1
    options = []
    cardInds = np.zeros((5,), dtype=int)
    for flush in handOptions.flushes:
        nC = flush.size
        for i1 in range(nC-4):
            cardInds[0] = handOptions.cards[flush[i1]].indexInHand
            for i2 in range(i1+1,nC-3):
                cardInds[1] = handOptions.cards[flush[i2]].indexInHand
                for i3 in range(i2+1,nC-2):
                    cardInds[2] = handOptions.cards[flush[i3]].indexInHand
                    for i4 in range(i3+1,nC-1):
                        cardInds[3] = handOptions.cards[flush[i4]].indexInHand
                        for i5 in range(i4+1,nC):
                            cardInds[4] = handOptions.cards[flush[i5]].indexInHand
                            handBeingPlayed = handOptions.cHand[cardInds]
                            if not gameLogic.isStraightFlush(handBeingPlayed):
                                continue
                            if prevHand is not None:
                                if not gameLogic.compareStraights(handBeingPlayed, prevHand):
                                    continue
                            options.append(
                                fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                            )
    if len(options) == 0:
        return -1
    return np.array(options, dtype=int)

def twoCardOptions(handOptions, prevHand = [], prevType = 0):
    #prevType = 1 - played a pair
    validInds = np.zeros((nActions[1],), dtype=int)
    c = 0
    cardInds = np.zeros((2,), dtype=int)
    
    if handOptions.nPairs > 0:
        for pair in handOptions.pairs:
            cardInds[0] = handOptions.cards[pair[0]].indexInHand
            cardInds[1] = handOptions.cards[pair[1]].indexInHand
            if prevType == 1:
                if handOptions.cHand[cardInds[1]] < prevHand[1]:
                    continue
            validInds[c] = twoCardIndices[cardInds[0]][cardInds[1]]
            c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1
    
def oneCardOptions(hand, prevHand = [], prevType = 0):
    nCards = len(hand)
    validInds = np.zeros((nCards,), dtype=int)
    c = 0
    for i in range(nCards):
        if prevType == 1:
            if prevHand > hand[i]:
                continue
        validInds[c] = i
        c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1
