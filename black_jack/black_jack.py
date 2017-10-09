import numpy as np
import pandas as pd

class Black_jack:
    def __init__(self):
        pass

    def valid_actions(self):
        return 'hit', 'stick'

    def is_valid_action(self, a):
        return a in self.valid_actions()

    def draw(self):
        p = min(np.random.randint(1, 13), 10)
        return p

    def reset(self):
        self.player_cards = []
        self.dealer_cards = []
        for _ in range(2):
            self.player_cards.append(self.draw())
            self.dealer_cards.append(self.draw())

        return (self.player_cards, self.dealer_cards[0])

    def best_score(self, cards):
        n_aces = cards.count(1)
        score = np.sum(cards)
        best_score = score
        usable_ace = False
        for n in range(1, n_aces + 1):
            s = score + n * 10
            if 21 >= s > best_score:
                best_score = s
                usable_ace = True

        return best_score, usable_ace

    def who_won(self):
        p_score = self.best_score(self.player_cards)[0]
        d_score = self.best_score(self.dealer_cards)[0]
        if 21 >= d_score > p_score:
            return -1
        elif d_score == p_score:
            return 0
        else:
            return 1

    def dealers_turn(self):
        # if player as gone bust
        if self.best_score(self.player_cards)[0] > 21:
            return -1

        while self.best_score(self.dealer_cards)[0] < 17:
            self.dealer_cards.append(self.draw())

        return self.who_won()

    def step(self, a):
        reward = 0
        if a == 'hit':
            self.player_cards.append(self.draw())
            done = False
        else:
            reward = self.dealers_turn()
            done = True

        return (self.player_cards, self.dealer_cards[0]), reward, done, {
               'd_cards': self.dealer_cards}



env = Black_jack()
# define an agent class and do things cleanly