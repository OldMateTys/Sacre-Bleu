#### DISCLAIMER ####
# This was produced with the assistance of GPT 4o

from collections import deque
from lib.interact.structure import StructureType
from lib.interact.tile import Tile, TileModifier
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


def analyse_board (game: Game, placement_list: list[tuple[int,int,int,Tile, int,dict[StructureType,list[Tile]]]]):
    
    score = [0 for _ in placement_list]
    grid = game.state.map._grid

    data = []
    for option in placement_list:
        x, y, r, tile, tile_idx, connected_comps = option
        

        placeable_structures = game.state.get_placeable_structures(tile._to_model())
        tile_data = []

        for structure in placeable_structures.values():
            tile_data.append(bfs(game, structure, x, y))


        



    return score

def find_target_struct_edges(target_structure, tile):
    if target_structure in (StructureType.ROAD, StructureType.ROAD_START):
        return (
            edge
            for edge, structure in tile.internal_edges
            if structure in (StructureType.ROAD, StructureType.ROAD_START)
        )    
    else:
        return (
            edge
            for edge, structure in tile.internal_edges
            if structure == target_structure
        )    

def is_meeple_on_tile_struct(tile, edges):
    for edge, meeple in tile.internal_claims.items():
        if edge not in edges:
            continue
        if meeple is not None:
            return meeple.player_id
    return None

def is_my_meeple_on_tile_struct(game: Game, tile, edges):
    me = game.state.me.player_id
    return sum(
            1
            for edge, meeple in tile.internal_claims.items()
            if edge in edges 
                and meeple is not None
                and meeple.player_id == me
            ) > 0


# Obtains as much information as possible about the current game state for a particular structure
# on a specific tile. E.g. find the point capcaity of a city tile on some tile x. 
def bfs (game: Game, targStruct: StructureType, x: int, y: int):
    
    grid = game.state.map._grid
    tile = grid[y][x]

    if tile is None:
        return
    
    que = deque()
    que.append((x, y))

    visited = set()
    
    count_structures = 0
    open_ends = set()
    meeples_active = {}
    my_meeples_active = 0
    count_emblems = 0

    while que:
        x, y = que.popleft()

        if (x, y) in visited:
            continue

        tile = grid[y][x]

        if tile is None:
            continue

        edges = find_target_struct_edges(targStruct, tile)
        visited.add((x,y))
        count_structures += 1

        meeple =  is_meeple_on_tile_struct(tile, edges)

        if meeple is not None:
            if meeple not in meeples_active:
                meeples_active[meeple] = 1
            else:
                meeples_active[meeple] += 1

        my_meeples_active += 1 if is_my_meeple_on_tile_struct(game, tile, edges) else 0

        if targStruct in (StructureType.ROAD, StructureType.ROAD_START) and TileModifier.BROKEN_ROAD_CENTER in tile.modifiers:
            continue

        if targStruct == StructureType.CITY:
            if tile.tile_type in ("H", "I", "R4"):
                continue
            if TileModifier.EMBLEM in tile.modifiers:
                count_emblems += 1

        if targStruct == StructureType.GRASS and tile.tile_type in ("F","G","U"):
            continue

        for edge in edges:
            match edge:
                case "top_edge":
                    next = grid[y-1][x]
                    if next is None:
                        open_ends.add((x, y-1))
                        continue
                    que.append((x, y-1))

                case "left_edge":
                    next = grid[y][x-1]
                    if next is None:
                        open_ends.add((x-1, y))
                        continue
                    que.append((x-1, y))

                case "right_edge":
                    next = grid[y][x+1]
                    if next is None:
                        open_ends.add((x+1, y))
                        continue
                    que.append((x+1, y))

                case "bottom_edge":
                    next = grid[y+1][x]
                    if next is None:
                        open_ends.add((x, y+1))
                        continue
                    que.append((x, y+1))


    return count_structures, len(open_ends), max(meeples_active.values()), my_meeples_active, count_emblems
