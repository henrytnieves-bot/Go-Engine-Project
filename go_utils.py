import numpy as np
import pyspiel
import pygame
import sys


def create_go_game(size):
    """
    load open-spiel game with provided size
    """
    if size == 5:
        komi = 0.5
    elif size == 9:
        komi = 5.5
    else:
        komi = 7.5
    game = pyspiel.load_game("go", {"board_size": size, "komi": komi})
    state = game.new_initial_state()
    return state
