#!/usr/bin/env python

#Code based from TicTacToc at https://gist.github.com/fheisler/430e70fa249ba30e707f
import numpy as np
import random
import sys
import scipy.io

ROWS = 6
COLUMNS = 7

class ConnectFour:
    def __init__(self, playerX, playerO):
        global ROWS
        global COLUMNS
        self.rows = ROWS
        self.win = 0
        self.columns = COLUMNS
        self.board = np.zeros((self.rows, self.columns), dtype=np.int)
        self.playerX, self.playerO = playerX, playerO
        self.playerX_turn = random.choice([True, False])

    def play_game(self):
        self.playerX.start_game(1)
        self.playerO.start_game(2)
        while True:
            if self.playerX_turn:
                player, char, other_player = self.playerX, 1, self.playerO
            else:
                player, char, other_player = self.playerO, 2, self.playerX
            if player.breed == "human":
                self.display_board()
            r,c = player.move(self.board)
            if self.board[r-1][c-1] != 0: # illegal move
                player.reward(-99, self.board) # score of shame
                print "ouch";
                break
            self.board[r-1][c-1] = char
            if self.player_wins(r-1,c-1):
                self.win = char
                player.reward(1, self.board)
                other_player.reward(-1, self.board)
                break
            if self.board_full(): # tie game
                player.reward(0.5, self.board)
                other_player.reward(0.5, self.board)
                break
            other_player.reward(0, self.board)
            self.playerX_turn = not self.playerX_turn

    #Method idea from http://codereview.stackexchange.com/questions/112948/checking-for-a-win-in-connect-four
    def player_wins(self, row, col):
        item = self.board[row][col]
        if item == 0:
            return False
        for delta_row, delta_col in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            consecutive_items = 1
            for delta in (1, -1):
                delta_row *= delta
                delta_col *= delta
                next_row = row + delta_row
                next_col = col + delta_col
                while 0 <= next_row < self.rows and 0 <= next_col < self.columns:
                    if self.board[next_row][next_col] == item:
                        consecutive_items += 1
                    else:
                        break
                    if consecutive_items == 4:
                        return True
                    next_row += delta_row
                    next_col += delta_col
        return False

    def board_full(self):
        return np.all(self.board)

    def who_won(self):
        return self.win

    def display_board(self):
        row = "|" + " {} |" * self.columns
        hr = "\n-" + "----" * self.columns + "\n"
        entire_board = (row + hr) * self.rows
        printable = [ (' ' if x == 0 else 'X' if x == 1 else 'O') for x in self.board.flatten()]
        print entire_board.format(*printable)


class Player(object):
    def __init__(self):
        self.breed = "human"

    def start_game(self, char):
        print "\nNew game!"

    def move(self, board):
        r = int(raw_input("row? "))
        c = int(raw_input("column? "))
        return (r,c)

    def reward(self, value, board):
        print "{} rewarded: {}".format(self.breed, value)

    def available_moves(self, board):
        global ROWS
        global COLUMNS

        #Convert state to matrix in case it's a tuple
        corrected_board = board
        if not isinstance(corrected_board, np.ndarray):
            corrected_board = np.reshape(board,(ROWS,COLUMNS))

        #Check available moves
        return [(r+1,c+1) for r in range(0,ROWS) for c in range(0,COLUMNS) if corrected_board[r][c] == 0 and (r+1 == ROWS or corrected_board[r+1][c] != 0)]


class RandomPlayer(Player):
    def __init__(self):
        self.breed = "random"

    def reward(self, value, board):
        pass

    def start_game(self, char):
        pass

    def move(self, board):
        return random.choice(self.available_moves(board))


class QLearningPlayer(Player):
    def __init__(self, epsilon=0.2, alpha=0.3, gamma=0.9):
        self.breed = "Qlearner"
        self.harm_humans = False
        self.q = {} # (state, action) keys: Q values
        self.epsilon = epsilon # e-greedy chance of random exploration
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor for future rewards

    def start_game(self, char):
        global ROWS
        global COLUMNS
        self.last_board = np.zeros((ROWS, COLUMNS), dtype=np.int).flatten()
        self.last_move = None

    def getQ(self, state, action):
        # encourage exploration; "optimistic" 1.0 initial values
        if self.q.get((state, action)) is None:
            self.q[(state, action)] = 1.0
        return self.q.get((state, action))

    def move(self, board):
        self.last_board = tuple(board.flatten())
        actions = self.available_moves(board)

        if random.random() < self.epsilon: # explore!
            self.last_move = random.choice(actions)
            return self.last_move

        qs = [self.getQ(self.last_board, a) for a in actions]
        maxQ = max(qs)

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

        self.last_move = actions[i]
        return actions[i]

    def reward(self, value, board):
        if self.last_move:
            self.learn(self.last_board, self.last_move, value, tuple(board.flatten()))

    def learn(self, state, action, reward, result_state):
        prev = self.getQ(state, action)
        maxqnew = max([self.getQ(result_state, a) for a in self.available_moves(state)])
        self.q[(state, action)] = prev + self.alpha * ((reward + self.gamma*maxqnew) - prev)

class QLearningRandomInitPlayer(QLearningPlayer):
    def getQ(self, state, action):
        # encourage exploration; "optimistic" 1.0 initial values
        if self.q.get((state, action)) is None:
            self.q[(state, action)] = random.uniform(0,1)
        return self.q.get((state, action))

class QSoftmaxPlayer(QLearningPlayer):
    def softmax(self, qs):
        """Compute softmax values for each sets of scores in x."""
        #Use epsilon as temperature parameter
        e_x = np.exp(np.divide(qs,self.epsilon))
        return [x / e_x.sum(axis=0) for x in e_x]

    def move(self, board):
        self.last_board = tuple(board.flatten())
        actions = self.available_moves(board)

        qs = [self.getQ(self.last_board, a) for a in actions]
        probabilities = self.softmax(qs)

        #Select one element with softmax probabilities
        choice = np.random.multinomial(1,probabilities)

        self.last_move = actions[np.nonzero(choice)[0]]
        return self.last_move

px = QLearningPlayer()
po = QLearningPlayer()

iterations = 200000
evalsize = 1000

for i in xrange(0,iterations):
    t = ConnectFour(px, pr)
    t.play_game()    
    print("."),
    sys.stdout.flush()

px.epsilon = 0
po = Player()

while True:
    t = ConnectFour(px, po)
    t.play_game()
