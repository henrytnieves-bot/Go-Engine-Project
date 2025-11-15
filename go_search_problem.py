from typing import Sequence, Type
# import go_utils
import numpy as np
from adversarial_search_problem import AdversarialSearchProblem, GameState
import copy
import go_utils
from pyspiel import Game

Action = int

DEFAULT_SIZE = 9

class GoState(GameState):
    """
    A state of the game of Go.
    Includes methods and properties for the state of the board, player to move, and other useful methods
    """
    def __init__(self, pyspiel_state: Game, player_to_move: int = 0):
        """
        Initialize GoState with pyspiel as backend Go engine.
        The initial state is created with a call to create_go_game() in go_utils.py
        Every other state will be generated from applying actions to the initial state.
        This essentially functions as a wrapper class to conver pyspiel game states to 
        The ASP interface used previously.

        :param pyspiel_state: pyspiel state of the game
        :param player_to_move: player to move
        """
        self.internal_state = pyspiel_state
        self.size = int(np.sqrt(len(pyspiel_state.observation_tensor()) / 4))


    def player_to_move(self) -> int:
        """
        Get the current player to move
        :return: player to move BLACK (0) or WHITE (1)
        """
        return self.internal_state.current_player()

    def get_board(self) -> np.ndarray:
        """
        Return the current board as a numpy array
        The board will have shape (4, size, size)
        The first channel (i.e., get_board()[0]) is the board for BLACK. There are 1's where the black pieces are and 0's elsewhere.
        The second channel (i.e., get_board()[1]) is the board for WHITE. There are 1's where the white pieces are and 0's elsewhere.
        The third channel (i.e., get_board()[2]) is the board for EMPTY. There are 1's where the empty spaces are and 0's elsewhere.
        The fourth channel (i.e., get_board()[3]) is the board for whose turn it is. There are 0's when it is BLACK's turn and 1's when it is white's.

        This is the default observation tensor used by pyspiel.
        """
        return np.array(self.internal_state.observation_tensor(0)).reshape(-1, self.size, self.size)

    def terminal_value(self) -> float:
        """
        Return the terminal value of the game.
        :return: 1 if BLACK wins, -1 if WHITE wins
        """
        return self.internal_state.returns()

    def clone(self) -> GameState:
        """
        Create a copy of the current game state.
        This is used for safety with the game runner.
        We don't want search algorithms to be able to directly modify the game state,
        so we only pass a copy of the state to the search algorithms.
        :return: a copy of the current game state
        """
        return GoState(self.internal_state.clone(), self.internal_state.current_player())

    def is_terminal_state(self) -> bool:
        """
        Checks if the game is in a terminal state.
        The state is if there are no legal actions left or the players have passed twice in a row.

        :return: True if the game is in a terminal state, False otherwise
        """
        return self.internal_state.is_terminal()

    def legal_actions(self) -> Sequence[Action]:
        """
        Return all possible legal actions for the given state.
        Note: Actions are represented as integers, by default.
        For a more human-readable representation, use action_index_to_coord()

        NOTE: It is preferrable to get the available actions from the search problem,
        not this state. 

        :return: list of legal actions
        """
        return self.internal_state.legal_actions()

    def apply_action(self, action: Action):
        """
        Apply action and update internal state.
        Action must be an int, not a coordinate.

        NOTE: It is preferrable to use the transition function from the search problem,
        not this method to apply actions. 
        """
        self.internal_state.apply_action(action)

    def get_pieces_coordinates(self, player_index: int):
        """
        Get the indices of the pieces of the given player.
        :param player_index: 0 for BLACK, 1 for WHITE
        :return: list of coordinates of the pieces of the given player
        """
        player_board = np.array(self.internal_state.observation_tensor(
            0)).reshape((-1, self.size, self.size))[player_index]
        return np.argwhere(player_board == 1)

    def get_pieces_array(self, player_index):
        """
        Get the 2D array of the pieces of the given player.
        The array will have shape (size, size) and will have 1's where the pieces are and 0's elsewhere.

        :param player_index: 0 for BLACK, 1 for WHITE
        :return: 2D np array of the pieces of the given player
        """
        player_board = np.array(self.internal_state.observation_tensor(
            0)).reshape((-1, self.size, self.size))[player_index]
        return player_board

    def get_empty_spaces(self):
        """
        return a 2D array of the empty spaces on the board
        The array will have shape (size, size) and will have 1's where the empty spaces are and 0's elsewhere.

        :return: 2D np array of the empty spaces on the board
        """
        return self.internal_state.observation_tensor(2)
    
    def action_index_to_coord(self, action: Action) -> tuple[int, int]:
        """
        Convert an action index to a coordinate.
        :param action: action index
        :return: coordinate (x, y)
        """
        return (action % self.size, action // self.size)

    def __repr__(self):
        return str(self.internal_state)


class GoProblem(AdversarialSearchProblem[GoState, Action]):
    def __init__(self, size=DEFAULT_SIZE, state=None, player_to_move=0):
        """
        Create a new Go search problem.
        If no state is provided, a new game is created with the given size.
        """
        if state is None:
            game_state = go_utils.create_go_game(size)
        else:
            game_state = state
        self.start_state = GoState(game_state, player_to_move)

    def get_available_actions(self, state: GoState) -> Sequence[Action]:
        """
        Get the available actions for the given state.
        Use this to get the list of available actions for a given state.
        Note: An action in this case is an integer in range [0, size^2].
        Each action index corresponds to a coordinate on the board (x, y) = (action % size, action // size).
        With action=size**2 reserved for the pass action.

        :param state: current state
        :return: list of available actions
        """
        return state.legal_actions()

    def transition(self, state: GoState, action: Action) -> GoState:
        """
        Return new_state resulting from applying action to state.

        :param state: current state
        :param action: action to apply
        :return: new state resulting from applying action to state
        """
        new_state = state.clone()
        new_state.apply_action(action)
        return new_state

    def is_terminal_state(self, state: GoState) -> bool:
        """
        Return if the given state is a terminal state.
        State is terminal if no legal actions are available or the players have passed twice in a row.

        :param state: current state
        :return: True if the state is terminal, False otherwise
        """
        return state.is_terminal_state()

    def evaluate_terminal(self, state: GoState) -> float:
        """
        Get the value of the terminal state.
        The value is 1 if BLACK wins and -1 if WHITE wins.

        :param state: current state
        :return: value of the terminal state
        """
        return state.terminal_value()[0]

    def action_index_to_string(self, action: Action) -> str:
        """
        Convert an Action (index) to a string.
        """
        return "(" + str(action % self.start_state.size) + ", " +  str(action // self.start_state.size) + ")"