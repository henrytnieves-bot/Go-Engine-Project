import random
import time
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from go_search_problem import GoProblem, GoState, Action

MAXIMIZER = 0
MIMIZER = 1

class GameAgent(ABC):
    @abstractmethod
    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        pass

class GoProblemLearnedHeuristic(GoProblem):
    def __init__(self, model=None, state=None):
        super().__init__(state=state)
        self.model = model

    def encoding(self, state):
        features = get_features(state)
        return features

    def heuristic(self, state, player_index):
        value = self.model.forward(torch.tensor(self.encoding(state), dtype=torch.float))
        return value

    def __str__(self) -> str:
        return "Learned Heuristic"

def get_features(game_state: GoState):
    board_size = game_state.size
    features = game_state.get_board()[0].flatten()
    features = np.append(features, game_state.get_board()[1].flatten())
    features = np.append(features, game_state.get_board()[2].flatten())
    
    sums = []
    for row in game_state.get_board()[0]:
        sums.append(int(sum(row)))
    for row in game_state.get_board()[1]:
        sums.append(int(sum(row)))
    for col in np.transpose(game_state.get_board()[0]):
        sums.append(int(sum(col)))
    for col in np.transpose(game_state.get_board()[1]):
        sums.append(int(sum(col)))
    
    features = np.append(features, game_state.get_board()[3][0][0])
    features = features.tolist() + sums

    # Count pieces
    features.append(int(sum(game_state.get_board()[0].flatten())))
    features.append(int(sum(game_state.get_board()[1].flatten())))
    
    return features

def load_model(path: str, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        output_size = 1
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))  
        x = self.relu(self.layer3(x))
        x = self.layer4(x)  
        return x

class IterativeLearnedAgent(GameAgent):
    def __init__(self, path, cutoff_time=1):
        super().__init__()
        self.cutoff_time = cutoff_time
        feature_size = 98
        self.model = load_model(path, ValueNetwork(feature_size))
        self.search_problem = GoProblemLearnedHeuristic(self.model)
        self.opening = True
        self.totaltime = 15

    def get_move(self, game_state, time_limit):
        player = game_state.player_to_move()
        actions = self.search_problem.get_available_actions(game_state)
        utility = {}
        done = False
        cutoff_depth = 1
        start = time.time()

        # Opening book
        if self.opening:
            self.opening = False
            self.totaltime += 1
            if player == 0:
                return 12
            else:
                return 12 if 12 in actions else 6

        def min_value(state, depth, alpha, beta):
            if self.search_problem.is_terminal_state(state):
                return self.search_problem.evaluate_terminal(state)
            elif depth == 0:
                return self.search_problem.heuristic(state, game_state.player_to_move())
            else:
                value = float('inf')
                for a in self.search_problem.get_available_actions(state):
                    nonlocal start, done
                    if (time.time() - start) > self.cutoff_time:
                        done = True
                        return float("inf")
                    value = min(value, max_value(self.search_problem.transition(state, a), depth-1, alpha, beta))
                    if value <= alpha:
                        return value
                    beta = min(beta, value)
                return value

        def max_value(state, depth, alpha, beta):
            if self.search_problem.is_terminal_state(state):
                return self.search_problem.evaluate_terminal(state)
            elif depth == 0:
                return self.search_problem.heuristic(state, game_state.player_to_move())
            else:
                value = -float('inf')
                for a in self.search_problem.get_available_actions(state):
                    nonlocal start, done
                    if (time.time() - start) > self.cutoff_time:
                        done = True
                        return -float("inf")
                    value = max(value, min_value(self.search_problem.transition(state, a), depth-1, alpha, beta))
                    if value >= beta:
                        return value
                    alpha = max(alpha, value)
                return value

        # Adjust cutoff time dynamically
        if self.totaltime > 10:
            self.cutoff_time = 2
        elif self.totaltime > 5:
            self.cutoff_time = 1.8
        elif self.totaltime > 3:
            self.cutoff_time = 1.2
        else:
            self.cutoff_time = 1

        while (time.time() - start) < self.cutoff_time:
            if player == 1:
                for action in actions:
                    if ((time.time() - start) < self.cutoff_time) and not done:
                        utility[action] = max_value(self.search_problem.transition(game_state, action), cutoff_depth-1, -float('inf'), float('inf'))
                    else:
                        done = True
                        break
                    if not done:
                        best_action = min(utility, key=utility.get)
            else:
                for action in actions:
                    if ((time.time() - start) < self.cutoff_time) and not done:
                        utility[action] = min_value(self.search_problem.transition(game_state, action), cutoff_depth-1, -float('inf'), float('inf'))
                    else:
                        done = True
                        break
                    if not done:
                        best_action = max(utility, key=utility.get)
            cutoff_depth += 1
            done = False

        self.totaltime -= (time.time() - start - 1)
        return best_action

    def __str__(self):
        return "Learned Deepening"

def get_final_agent_5x5():
    return IterativeLearnedAgent(path="new_value_model.pt")
