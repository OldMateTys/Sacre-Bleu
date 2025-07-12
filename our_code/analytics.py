#### DISCLAIMER ####
# This was produced with the assistance of GPT 4o

from collections import deque
from lib.interact.structure import StructureType
from lib.interact.tile import Tile
from helper.game import Game
from helper.state_mutator import ClientSate
from lib.interface.events.moves.move_place_meeple import (
    MovePlaceMeeple,
    MovePlaceMeeplePass,
)
from lib.interface.events.moves.move_place_tile import MovePlaceTile
from lib.interface.queries.typing import QueryType
from lib.interface.queries.query_place_tile import QueryPlaceTile
from lib.interface.queries.query_place_meeple import QueryPlaceMeeple
from lib.interface.events.moves.typing import MoveType
from lib.models.tile_model import TileModel



def find_valid_edges (grid, x: int ,y: int) -> list[str]:
    
    directions = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
            )

    valid_edges = []
    for i in range(len(directions)):
        edge = Tile.get_edges()[i]

        dx = x + directions[i][0]
        dy = y + directions[i][1]

        if grid[dy][dx] is not None:
            valid_edges.append(edge)
        
    return valid_edges


def analyse_board (game: Game, placement_list):
    
    score = [0 for _ in placement_list]
    grid = game.state.map._grid

    for option in placement_list:
        x, y, r, tile = option
        structures = game.state.get_tile_structures(tile)
        structs = {}
        for edge, structure in structures:

            if structure not in structs.keys():
                structs[structure] = [edge]
            else:
                structs[structure].append(edge)


        for edge in structures.keys():
            print(edge)




    return score



def bfs (structure: StructureType, tile: Tile, x: int, y: int):
    
    que = deque()
    que.append((x, y, structure))
    visited = set()
    
    count_structures = 0
    open_ends = 0
    meeples_active = 0
    my_meeples_active = 0

    while que:
        

