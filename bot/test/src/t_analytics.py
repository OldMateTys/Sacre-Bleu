from collections import deque
import unittest

from helper.game import Game
from lib.config.map_config import MONASTARY_IDENTIFIER
from lib.interact.structure import StructureType
from lib.interact.tile import Tile, TileModifier
from lib.interact.meeple import Meeple
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

from copy import deepcopy

from helper.game import Game, Tile

def print_map(grid: list[list["Tile | None"]], print_range: range) -> None:
    assert grid
    assert len(grid) >= max(print_range) + 1

    # Print top header
    print("\t.", end="")
    for i in print_range:
        print(f" {i:>2} ", end=",")
    print()

    for i in print_range:
        row = grid[i]
        print(f"{i:>2}", end="\t")
        print(
            [col.tile_type.ljust(1, " ") + str(col.rotation) if col else "__" for col in [row[j] for j in print_range]],
            flush=True,
        )

    print("### END ###\n", end="")


def find_target_struct_edges(target_structure, tile):
    if target_structure in (StructureType.ROAD, StructureType.ROAD_START):
        return (
            edge
            for edge, structure in tile.internal_edges.items()
            if structure in (StructureType.ROAD, StructureType.ROAD_START)
        )    
    else:
        return (
            edge
            for edge, structure in tile.internal_edges.items()
            if structure == target_structure
        )    
def get_edge_from_delta(x, y, x_prev, y_prev):
    dx, dy = x_prev - x, y_prev - y
    if dx == -1: 
        return "left_edge"
    if dx ==  1: 
        return "right_edge"
    if dy == -1: 
        return "top_edge"
    if dy ==  1: 
        return "bottom_edge"
    return None  # Shouldn't happen

def is_meeple_on_tile_struct(tile, edges, x_prev, y_prev):

    x, y = tile.placed_pos
    for edge, meeple in tile.internal_claims.items():

        # Special handling for CITY on disconnected city tiles
        if tile.tile_type in ("H", "I", "R4") and tile.internal_edges[edge] == StructureType.CITY:
            # Only check the edge facing the previous tile
            if edge != get_edge_from_delta(x, y, x_prev, y_prev):
                continue

        # Skip irrelevant edges
        if edge not in edges:
            continue

        if meeple is not None:
            return meeple.player_id

    return None

def is_my_meeple_on_tile_struct(tile, edges, x_prev, y_prev):
    player_id = 1


    x, y = tile.placed_pos
    relevant_edge = get_edge_from_delta(x, y, x_prev, y_prev)

    return sum(
        1
        for edge, meeple in tile.internal_claims.items()
        if edge in edges
            and meeple is not None
            and meeple.player_id == player_id
            and not (
                tile.tile_type in ("H", "I", "R4")
                and tile.internal_edges[edge] == StructureType.CITY
                and edge != relevant_edge
            )
    ) > 0

def get_info_tester(grid: list[list[Tile | None]], tile: Tile, x: int, y: int, targStruct: StructureType, starting_edge: str = "") -> tuple:

    original_tile = grid[y][x]  # Should be None, but just in case

    # Place the tile temporarily
    tile.placed_pos = (x, y)
    grid[y][x] = tile
    
    try:
        return bfs_tester(grid, targStruct, x, y, starting_edge)
    finally:
        print_map(grid, range(10))
        tile.placed_pos = (0,0)
        grid[y][x] = original_tile


def bfs_tester (grid: list[list[Tile | None]], targStruct: StructureType, x: int, y: int, starting_edge: str) -> tuple:
    
    tile = grid[y][x]

    if tile is None:
        return ()
    
    que = deque()
    que.append((x, y, x, y))

    visited = set()
    
    count_structures = 0
    open_ends = set()
    meeples_active = 0
    my_meeples_active = 0
    count_emblems = 0

    while que:
        x, y, x_prev, y_prev = que.popleft()
        print("DEBUG 1")

        if (x, y) in visited:
            continue

        print("DEBUG 2")
        tile = grid[y][x]

        if tile is None:
            continue

        print("DEBUG 3")
        if (x, y) == (x_prev, y_prev) and targStruct == StructureType.CITY and tile.tile_type in {"H", "I", "R4"}:
            edges = [starting_edge] if starting_edge in tile.internal_edges else []
        else:
            edges = list(find_target_struct_edges(targStruct, tile))

        print("DEBUG 4")
        visited.add((x,y))
        count_structures += 1
        print(f"Added: {(x, y)}")

        meeples_active += 1 if is_meeple_on_tile_struct(tile, edges, x_prev, y_prev) else 0

        my_meeples_active += 1 if is_my_meeple_on_tile_struct(tile, edges, x_prev, y_prev) else 0

        print("DEBUG 5")
        if targStruct in (StructureType.ROAD, StructureType.ROAD_START) and TileModifier.BROKEN_ROAD_CENTER in tile.modifiers:
            continue

        print("DEBUG 6")
        if targStruct == StructureType.CITY:
            if tile.tile_type in ("H", "I", "R4"):
                continue
            if TileModifier.EMBLEM in tile.modifiers:
                count_emblems += 1

        print("DEBUG 7")
        if targStruct == StructureType.GRASS and tile.tile_type in ("F","G","U"):
            continue

        print("DEBUG 8")
        print(edges)
        for edge in edges:
            match edge:
                case "top_edge":
                    next = grid[y-1][x]
                    if next is None:
                        open_ends.add((x, y-1))
                        continue
                    que.append((x, y-1, x, y))

                case "left_edge":
                    next = grid[y][x-1]
                    if next is None:
                        open_ends.add((x-1, y))
                        continue
                    que.append((x-1, y, x, y))

                case "right_edge":
                    next = grid[y][x+1]
                    if next is None:
                        open_ends.add((x+1, y))
                        continue
                    que.append((x+1, y, x, y))

                case "bottom_edge":
                    next = grid[y+1][x]
                    if next is None:
                        open_ends.add((x, y+1))
                        continue
                    que.append((x, y+1, x, y))


    return count_structures, len(open_ends), meeples_active, my_meeples_active, count_emblems

class TestBFS(unittest.TestCase):

    def setUp(self):
        
        self.grid: list[list[Tile | None]]
        self.grid = []

        for _ in range(10):
            row = []
            for _ in range(10):
                row.append(None)
            self.grid.append(row)
        # Setup: Place a basic connected CITY across (1,1) and (1,2)

        self.tileB = Tile(
            tile_id="B",
            left_edge=StructureType.GRASS,
            right_edge=StructureType.GRASS,
            top_edge=StructureType.GRASS,
            bottom_edge=StructureType.GRASS,
            modifiers=[TileModifier.MONASTARY],
        )
        self.tileK = Tile(
            tile_id="K",
            left_edge=StructureType.ROAD,
            right_edge=StructureType.CITY,
            top_edge=StructureType.ROAD,
            bottom_edge=StructureType.GRASS,
        )
        self.tileK.rotate_clockwise(1)
        self.tileF = Tile(
            tile_id="F",
            left_edge=StructureType.CITY,
            right_edge=StructureType.CITY,
            top_edge=StructureType.GRASS,
            bottom_edge=StructureType.GRASS,
            modifiers=[TileModifier.EMBLEM, TileModifier.OPP_CITY_BRIDGE],
        )

        self.tileO = Tile(
            tile_id="O",
            left_edge=StructureType.CITY,
            right_edge=StructureType.ROAD,
            top_edge=StructureType.CITY,
            bottom_edge=StructureType.ROAD,
            modifiers=[TileModifier.EMBLEM],
        )
        

        self.grid[2][2] = self.tileB
        self.grid[3][2] = self.tileF
        self.grid[2][3] = self.tileK
        self.grid[3][3] = self.tileO

        self.tileB.placed_pos = (2, 2)
        self.tileF.placed_pos = (2, 3)
        self.tileK.placed_pos = (3, 2)
        self.tileO.placed_pos = (3, 3)

        self.meeple0 = Meeple(0)
        self.meeple1 = Meeple(1)
        self.meeple2 = Meeple(2)
        self.meeple3 = Meeple(3)

        self.tileB.internal_claims[MONASTARY_IDENTIFIER] = self.meeple0
        self.tileO.internal_claims["top_edge"] = self.meeple1


    def test_bfs_city_connection(self):
        new_tile =Tile(
                tile_id="L",
                left_edge=StructureType.ROAD_START,
                right_edge=StructureType.CITY,
                top_edge=StructureType.ROAD_START,
                bottom_edge=StructureType.ROAD_START,
        )

        count_structures, open_ends, meeples, my_meeples, emblems = get_info_tester(
            self.grid, new_tile, 1, 3, StructureType.CITY, ""
        )

        print(f"[DEBUG] CITY INFO: structures={count_structures}, open_ends={open_ends}, "
              f"meeples={meeples}, my_meeples={my_meeples}, emblems={emblems}")
        
        print_map(self.grid, range(10))
        
        assert(count_structures == 4)
        assert(open_ends == 0)
        assert(meeples == 1)
        assert(my_meeples == 1)
        assert(emblems == 2)

    def test_bfs_road_connection_simple(self):

        new_tile =Tile(
            tile_id="D",
            left_edge=StructureType.GRASS,
            right_edge=StructureType.CITY,
            top_edge=StructureType.ROAD,
            bottom_edge=StructureType.ROAD,
        )
        new_tile.rotate_clockwise(1)

        count_structures, open_ends, meeples, my_meeples, emblems = get_info_tester(
            self.grid, new_tile, 4, 2, StructureType.ROAD, ""
        )

        print(f"[DEBUG] ROAD INFO: ROAD={count_structures}, open_ends={open_ends}, "
              f"meeples={meeples}, my_meeples={my_meeples}, emblems={emblems}")
        
        print_map(self.grid, range(10))

        assert(count_structures == 2)
        assert(open_ends == 2)
        assert(meeples == 0)


    def test_bfs_isolated_tile(self):
        return
        # Isolated ROAD
        tile = Tile("R1", left_edge=StructureType.ROAD, right_edge=StructureType.ROAD,
                    top_edge=StructureType.GRASS, bottom_edge=StructureType.GRASS)
        tile.tile_type = "L"
        tile.x, tile.y = 5, 5
        self.grid[5][5] = tile

        count_structures, open_ends, meeples, my_meeples, emblems = bfs_tester(
            self.grid, StructureType.ROAD, 5, 5, ""
        )

        self.assertEqual(count_structures, 1)
        self.assertEqual(open_ends, 2)  # both left and right are open
        self.assertEqual(meeples, {})
        self.assertEqual(my_meeples, 0)
        self.assertEqual(emblems, 0)


if __name__ == '__main__':
    unittest.main()
