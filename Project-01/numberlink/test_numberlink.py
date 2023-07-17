#!/usr/bin/env python3

from numberlink import (
    admissible_h1, bfs, b1s, admissible_astar,
    connect, get_moves, get_successors, heuristic, initial,
    is_connected, is_goal, move, count_moves, parse_state,
    h_val, h_func, links, grid
)

SOL = parse_state(
    [3,2,2,2,
     3,2,1,1,
     3,3,3,1,
     4,4,4,4],
    4,4
)

Tby2 = parse_state([1,2,1,2], 2, 2)
Thby2 = parse_state([1,2,0,0,1,2], 3, 2)

def test_is_connected():
    assert is_connected(SOL, 1)
    assert not is_connected(Tby2, 1)
    connect(Tby2, 0)
    assert is_connected(Tby2, 0)

def test_is_goal():
    assert not is_goal(initial)
    assert is_goal(SOL)
    connect(Tby2, 0)
    connect(Tby2, 1)
    assert is_goal(Tby2)

def test_get_moves():
    assert get_moves(Tby2)['N'][0] == -2

def test_move():
    move_n = get_moves(Thby2)['E']
    assert move(Thby2, move_n, 2-1) == [
        [1, 2, 2, 0, 1, 2],
        [[0, None, 4], [1, 2, 5]],
        (3, 2)]


def test_get_successors():
    assert get_successors(SOL) == []
    assert get_successors(Thby2) == [
        [[1, 2, 0, 1, 1, 2],
         [[0, 3, 4], [1, None, 5]],
         (3, 2)],
        [[1, 2, 2, 0, 1, 2],
         [[0, None, 4],
          [1, 2, 5]],
         (3, 2)]]

def test_bfs():
    assert bfs(SOL)
    assert bfs(Tby2)
    assert bfs(Thby2)
    assert bfs(initial)

def test_admissible_h():
    SOL[h_func] = admissible_h1
    assert heuristic(SOL) == 0
    assert SOL[h_val] == 0

    Tby2[h_func] = admissible_h1
    assert heuristic(Tby2) == 0
    assert Tby2[h_val] == 0

    Thby2[h_func] = admissible_h1
    # 2 links and 2 empty stops
    assert heuristic(Thby2) == 4
    assert Thby2[h_val] == 4

    initial[h_func] = admissible_h1
    assert heuristic(initial) == 16
    assert initial[h_val] == 16

def test_b1s_admissible():
    SOL[h_func] = admissible_h1
    assert b1s(SOL)
    Tby2[h_func] = admissible_h1
    assert b1s(Tby2)
    Thby2[h_func] = admissible_h1
    assert b1s(Thby2)
    initial[h_func] = admissible_h1
    assert b1s(initial)

def test_astar_admissible():
    SOL[h_func] = admissible_astar
    assert b1s(SOL)
    Tby2[h_func] = admissible_astar
    assert b1s(Tby2)
    Thby2[h_func] = admissible_astar
    assert b1s(Thby2)
    initial[h_func] = admissible_astar
    assert b1s(initial)

def test_no_moves():
    """
    fake invalid move
    1 2 1
    0 1 2
    """
    Thby2[grid][2] = 1
    Thby2[links][0].insert(-2, 2)
    assert count_moves(Thby2, 0)
