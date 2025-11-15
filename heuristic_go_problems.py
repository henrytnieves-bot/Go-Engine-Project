from go_search_problem import GoProblem
import torch
BLACK = 0
WHITE = 1

class GoProblemSimpleHeuristic(GoProblem):
    def __init__(self, state=None):
        super().__init__(state=state)

    def heuristic(self, state, player_index):
        """
        Very simple heuristic that just compares the number of pieces for each player

        
        Having more pieces (>1) than the opponent means that some were captured, capturing is generally good.
        """
        return len(state.get_pieces_coordinates(BLACK)) - len(state.get_pieces_coordinates(WHITE))

    def __str__(self) -> str:
        return "Simple Heuristic"
