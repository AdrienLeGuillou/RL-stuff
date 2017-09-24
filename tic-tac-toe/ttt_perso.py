import numpy as np

class TicTacToe:
    def __init__(self):
        self.reset_board()

    def check_win_draw(self):
        sums = list(np.sum(self.grid, 0))
        sums += list(np.sum(self.grid, 1))
        sums.append(np.trace(self.grid))
        sums.append(np.trace(np.flip(self.grid, 1)))

        # reward 1 for win, 0 for draw, no reward until game is over
        # return a logical that says if the game is over
        if 3 in np.abs(sums):
            self.done = True
            return 1
        elif not 0 in self.grid:
            self.done = True
            return 0


    def play(self, pos, player):
        """ play a turn of the game

        pos is a tupple of valid position on the grid,
        player is 1 for cross and -1 for circles
        """
        self.grid[pos[0], pos[1]] = player

    def get_state(self, grid):
        state = 0
        for i in range(3):
            for j in range(3):
                state += (grid[i,j] + 1) * 3**(i*3+j)

        return int(state)

    def get_current_state(self):
        return self.get_state(self.grid)

    def get_next_state(self, move, player):
        grid = self.grid.copy()
        grid[move[0], move[1]] = player

        return self.get_state(grid)

    def get_legal_moves(self):
        legal = np.isin(self.grid, 0)
        legal = np.where(legal)
        legal = np.transpose(legal)

        return legal

    def reset_board(self):
        self.grid = np.zeros((3, 3))
        self.done = False

    def draw_board(self):
        print('-------------')
        grid = self.grid.astype(str)
        grid[grid == '1.0'] = 'X'
        grid[grid == '-1.0'] = 'O'
        grid[grid == '0.0'] = ' '
        print(np.array2string(grid))

    def game_over(self):
        self.check_win_draw()
        return self.done


class Player:
    def __init__(self, sym, eps, lr):
        self.sym = sym
        self.eps = eps
        self.lr = lr
        self.reset()

    def play(self, env):
        legals = env.get_legal_moves()

        p = np.random.random()
        if p < self.eps:
            move = np.random.choice(len(legals))
        else:
            values = []
            for move in legals:
                next_state = env.get_next_state(move, self.sym)
                if next_state in self.states_value:
                    values.append(self.states_value[next_state])
                else:
                    values.append(0)
            move = np.argmax(values)

        move = legals[move]

        env.play(move, self.sym)

    def reset_history(self):
        self.states_history = []

    def reset(self):
        self.states_value = {}
        self.reset_history()

    def update_history(self, env):
        self.states_history.append(env.get_current_state())

    def update(self, reward):
        self.states_value[self.states_history[-1]] = reward
        previous = reward

        for state in reversed(self.states_history[:-1]):
            if not state in self.states_value:
                self.states_value[state] = 0
            self.states_value[state] += \
                self.lr * (previous - self.states_value[state])

            previous = self.states_value[state]


class Human(Player):
    def __init__(self, sym, eps=0, lr=0):
        super().__init__(sym, eps, lr)

    def play(self, env):
        legals = env.get_legal_moves()
        print("Here are the allowed moves :")
        for i in range(len(legals)):
            print(i, legals[i])

        j = input("what is your choice ?")

        env.play(legals[int(j)], self.sym)

def play_game(p1, p2, env, draw=False):
    env.reset_board()
    p1.reset_history()
    p2.reset_history()
    current_player = p2

    while not env.game_over():
        if current_player == p2:
            current_player = p1
        else:
            current_player = p2

        current_player.play(env)
        current_player.update_history(env)

        if draw:
            env.draw_board()

    # update the state values
    current_player.update(env.check_win_draw())
    if current_player == p2:
        current_player = p1
    else:
        current_player = p2
    current_player.update_history(env)
    current_player.update(env.check_win_draw() * (-1))



env = TicTacToe()
player1 = Player(1, 0.1, 0.1)
player2 = Player(-1, 0.1, 0.1)

# training
for _ in range(10000):
    play_game(player1, player2, env, False)

# visualize some games
for i in range(10):
    print('game: ', i)
    play_game(player1, player2, env, True)


# play against the agents at their best
player1.eps = 0
player2.eps = 0

human = Human(-1)
play_game(player1, human, env, True)

human = Human(1)
play_game(human,player2, env, True)