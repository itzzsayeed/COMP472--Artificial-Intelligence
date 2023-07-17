#!/usr/bin/env python3

from P2 import bfs, dfs, b1s, move, successors
from P2 import N, S, E, W
from P2 import tiles, parse_state, h_func
from P2 import inversions, is_solvable
from P2 import (
    hamming,
    manhat_collision,
    permutation_inversion,
    manhattan,
    count_all_lc
)
import math
# CONF['print'] = False

no_h = lambda x: 1

l = [1, 2, 3,
     8, 9, 4,
     7, 6, 5]



GOAL = parse_state(l)
ONE_MOVE = parse_state(
    [1, 9, 3,
     8, 2, 4,
     7, 6, 5])


TWO_MOVE = parse_state(
    [9, 1, 3,
     8, 2, 4,
     7, 6, 5])

THREE_MOVE = parse_state(
    [8, 1, 3,
     9, 2, 4,
     7, 6, 5])

FIVE_MOVE = parse_state(
    [8, 1, 3,
     7, 2, 4,
     6, 9, 5])

NO_SOL = parse_state(
    [1, 2, 3,
     8, 9, 4,
     7, 5, 6 ])

def test_inversions():
    assert inversions(ONE_MOVE) == 2
    assert inversions(TWO_MOVE) == 2
    assert inversions(THREE_MOVE) == 4
    assert inversions(FIVE_MOVE) == 6
    assert inversions(NO_SOL) == 1

def test_move():
    initial = parse_state(GOAL[tiles])
    move(GOAL, N)
    move(GOAL, S)
    move(GOAL, E)
    move(GOAL, W)
    assert GOAL[tiles] == initial[tiles]

    s = [9, 2, 3,
         8, 1, 4,
         7, 6, 5]
    state = parse_state(s)


def test_successors():
    left = [
        1, 2, 3,
        9, 8, 4,
        7, 6, 5]

    right = [
        1, 2, 3,
        8, 4, 9,
        7, 6, 5]

    up = [
        1, 9, 3,
        8, 2, 4,
        7, 6, 5]
    down = [
        1, 2, 3,
        8, 6, 4,
        7, 9, 5]

    assert [x[tiles] for x in successors(GOAL)] == [up, left, right, down]

def test_solvable():
    assert is_solvable(ONE_MOVE) is True
    assert is_solvable(FIVE_MOVE) is True
    assert is_solvable(NO_SOL) is False

def test_bfs():
    assert bfs(NO_SOL)[2] == -1
    assert bfs(GOAL)[2] == 1
    assert bfs(ONE_MOVE)[2] == 2
    assert bfs(FIVE_MOVE)[2] == 6

def test_dfs():
    assert dfs(NO_SOL)[2] == -1
    assert dfs(GOAL)[2] == 1
    assert dfs(ONE_MOVE)[2] == 2
    # 218? depends on the move order
    # assert dfs(FIVE_MOVE)[2] == 6

def test_manhattan():
    # NOTE: it will compute, the validity test is done in heuristic()
    # assert manhattan(NO_SOL) == math.inf
    assert manhattan(GOAL) == 0
    assert manhattan(ONE_MOVE) == 1
    assert manhattan(TWO_MOVE) == 2
    assert manhattan(FIVE_MOVE) == 5
    randomized_test = [
        2, 4, 7,
        1, 5, 9,
        6, 3, 8]

    p = parse_state(randomized_test)
    assert manhattan(p) == 17

def test_b1s_manhat():
    NO_SOL[h_func] = manhattan
    GOAL[h_func] = manhattan
    ONE_MOVE[h_func] = manhattan
    TWO_MOVE[h_func] = manhattan
    assert b1s(NO_SOL)[2] == -1
    assert b1s(GOAL)[2] == 1
    assert b1s(ONE_MOVE)[2] == 2
    assert b1s(TWO_MOVE)[2] == 3
    assert b1s(FIVE_MOVE)[2] == 6


def test_lc():
    assert count_all_lc(GOAL) == 0
    assert count_all_lc(ONE_MOVE) == 0
