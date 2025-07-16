#### DISCLAIMER ####
# This was produced with the assistance of GPT 4o

from collections import deque
import random
from helper.game import Game
from lib.config.map_config import MONASTARY_IDENTIFIER
from lib.interact.structure import StructureType
from lib.interact.tile import Tile, TileModifier
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

from helper.utils import print_map


MEEPLE_FREE_SCORE_MODIFIER = 2


class TileStructData:

    def __init__(self, tile, struct_type, num_structs, num_open_ends, meeples_active, my_meeples_active, num_emblems, pos, edges: list[str]):

        # Convert ROAD_START to ROAD for code simplification
        self.tile = tile
        self.struct_type = struct_type if struct_type != StructureType.ROAD_START else StructureType.ROAD
        self.num_structs = num_structs
        self.num_open_ends = num_open_ends
        self.meeples_active = meeples_active
        self.my_meeples_active = my_meeples_active
        self.num_emblems = num_emblems
        self.x, self.y = pos
        self.edges = edges

        self.is_completed = num_open_ends == 0 or struct_type == StructureType.MONASTARY
        # print(f"TILE: {tile}, STRUCT_TYPE: {struct_type}")
        # print(f"TOTAL MEEPLES PLACED: {self.meeples_active}")

    def compute_score_gain_with_meeple(self, game) -> float:
        # Penalize contested or incomplete structures
        if self.struct_type == StructureType.MONASTARY:
            return self.get_monastary_score(game)
        if self.meeples_active:
            return -99    # If there are meeples active, we cannot place meeple. -99 makes this not a valid option, ever.
        if self.is_completed:
            return self.compute_completed_score(game) + self.my_meeples_active * MEEPLE_FREE_SCORE_MODIFIER
        
        return 0  # Boost if not contested

    def compute_score_gain_without_meeple(self, game) -> float:

        # Without placing a meeple, nearby monasteries count towards points
        point_total = self.get_monastary_score(game)

        # Not placing a meeple on monastary -> no points
        if self.struct_type == StructureType.MONASTARY:
            return point_total

        # If we are the equal or dominant party on this structure, points.
        if self.my_meeples_active * 2 >= self.meeples_active:

            # If completed, we get more points for 
            if self.is_completed:
                return self.compute_completed_score(game) + self.my_meeples_active * MEEPLE_FREE_SCORE_MODIFIER
            else:
                return self.compute_incomplete_score(game)

        # else we are not dominant, and receive no points for this structure.
        return point_total

    def get_monastary_score(self, game):
        grid = game.state.map._grid
        x, y = self.x, self.y
        return sum(1 for dx in (x-1,x,x+1) for dy in (y-1,y,y+1) if grid[dy][dx] is not None and (dx != x or dy != y)) + 1

    def get_nearby_monastary_score(self, game):
        grid = game.state.map._grid
        x, y = self.x, self.y
        return sum(1 
                   for dx in (x-1,x,x+1) 
                   for dy in (y-1,y,y+1) 
                   if (dx != x or dy != y) 
                   and grid[dy][dx] is not None 
                   and grid[dy][dx].internal_claims[MONASTARY_IDENTIFIER] is not None 
                   and grid[dy][dx].internal_claims[MONASTARY_IDENTIFIER].player_id == game.state.me.player_id
                   )

    def compute_completed_score(self, _):
        
        match self.struct_type:
            case StructureType.CITY:
                prev_points = self.num_structs - 1 + self.num_emblems
                prev_points -= 1 if TileModifier.EMBLEM in self.tile.modifiers else 0
                new_points = 2 * (self.num_structs + self.num_emblems)
                return new_points - prev_points
            case StructureType.ROAD:
                return 1
        return 0


    def compute_incomplete_score(self, _):

        match self.struct_type:
            case StructureType.CITY:
                return 2 if TileModifier.EMBLEM in self.tile.modifiers else 1
            case StructureType.ROAD:
                return 1
        return 0



class TileData:
    def __init__(self, features: list[TileStructData]):
        self.features_info = features
        self.best_edge = 0


    def compute_highest_score_with_meeple(self, game):
        best_edge: str | None
        best_edge = None
        highest_score = float('-inf')

        meeple_scores = []
        non_meeple_scores = []

        # print(f"[DEBUG] Total structures: {len(self.features_info)}")

        for idx, struct in enumerate(self.features_info):
            score_with = struct.compute_score_gain_with_meeple(game)
            score_without = struct.compute_score_gain_without_meeple(game)
            meeple_scores.append(score_with)
            non_meeple_scores.append(score_without)

            # print(f"[DEBUG] Struct {idx}: type={struct.struct_type}, score_with={score_with}, "
                  # f"score_without={score_without}, edges={struct.edges}")

        total_non_meeple_scores = sum(non_meeple_scores)

        for i in range(len(self.features_info)):
            score = total_non_meeple_scores - non_meeple_scores[i] + meeple_scores[i]
            # print(f"[DEBUG] Evaluating struct {i}: total score if meeple placed = {score}")

            if score > highest_score:
                highest_score = score

                if not self.features_info[i].edges:
                    # print(f"[WARNING] Struct {i} has empty edge list! Struct type: {self.features_info[i].struct_type}, tile: {getattr(self.features_info[i], 'tile', 'unknown')}")
                    best_edge = None
                else:
                    # print(f"Edges list: {self.features_info[i].edges}")
                    # for edge in self.features_info[i].edges:
                        # print(f"-- {edge}: {self.features_info[i].tile.internal_edges[edge]}")
                    best_edge = self.features_info[i].edges[0]

        # print(f"[DEBUG] Final decision: score={highest_score}, best_edge={best_edge}", flush=True)
        return highest_score, best_edge

    def compute_effective_score_without_meeple(self, game):
        score = 0
        for struct in self.features_info:
            score += struct.compute_score_gain_without_meeple(game)

        return score

    def compute_impact_of_meeple(self, game):
        meeple_score, edge = self.compute_highest_score_with_meeple(game)
        non_meeple_score = self.compute_effective_score_without_meeple(game)

        return meeple_score - non_meeple_score, edge


def get_placeable_structures_for_unplaced_tile(tile: Tile) -> dict[str, StructureType]:
    """
    Returns a mapping of edge name -> StructureType for unplaced tiles.
    This works independently of map position.
    """
    placeable = {}

    for edge in Tile.get_edges():
        structure = tile.internal_edges[edge]
        
        # Ignore empty or GRASS edges
        if structure in (StructureType.GRASS, StructureType.RIVER, None):
            continue
        
        placeable[edge] = structure

    # Optionally include center structure (e.g., for monasteries)
    if tile.tile_type in ("R8", "B"):
        placeable[MONASTARY_IDENTIFIER] = StructureType.MONASTARY

    return placeable
def get_all_edges() -> tuple:
    return "top_edge", "bottom_edge", "left_edge", "right_edge", MONASTARY_IDENTIFIER

def find_struct_on_edge(tile: Tile, edge: str):
    
    if edge == MONASTARY_IDENTIFIER:
        if tile.tile_type not in ("R8", "B"):
            return None
        # print("Entering.")
        # print(f"Tile type: {tile.tile_type}")
        # print(f"Tile modifiers: {tile.modifiers}")
        return StructureType.MONASTARY if TileModifier.MONASTARY in tile.modifiers else None
    return tile.internal_edges[edge]


def analyse_board (game: Game, placement_list: list[tuple[int,int,int,Tile, int,dict[StructureType,list[Tile]]]]):
    
    best_score_with_meeple = float('-inf')
    best_score_without_meeple = float('-inf')
    best_meeple_option = 0
    best_meeple_less_option = 0
    best_place_meeple = False
    best_meeple_edge = None

    data = []


    for i, option in enumerate(placement_list):
        x, y, r, tile, tile_idx, connected_comps = option
        
        # print(f"TILE MODEL: {tile._to_model()}", flush=True)
        placeable_structures = get_placeable_structures_for_unplaced_tile(tile)
        structs_data = []


        DISCONNECTED_CITIES = {"H", "I", "R4"}
        DISCONNECTED_ROADS = {"L", "W", "X"}



        for struct_type in set(placeable_structures.values()):

            if tile.tile_type in DISCONNECTED_CITIES and struct_type == StructureType.CITY:

                valid_edges = [edge for edge in Tile.get_edges() if tile.internal_edges[edge] == StructureType.CITY]
                for edge in valid_edges:
                    x1, x2, x3, x4, x5 = get_info(game, tile, x, y, struct_type, edge)
                    structs_data.append(TileStructData(tile, struct_type, x1, x2, x3, x4, x5, (x, y), [edge]))

            elif tile.tile_type in DISCONNECTED_ROADS and struct_type == StructureType.ROAD_START:

                valid_edges = [edge for edge in Tile.get_edges() if tile.internal_edges[edge] == StructureType.ROAD_START]
                for edge in valid_edges:
                    x1, x2, x3, x4, x5 = get_info(game, tile, x, y, struct_type, edge)
                    structs_data.append(TileStructData(tile, struct_type, x1, x2, x3, x4, x5, (x, y), [edge]))

            else:
                valid_edges = [edge for edge in get_all_edges() if find_struct_on_edge(tile, edge) == struct_type]
                x1, x2, x3, x4, x5 = get_info(game, tile, x, y, struct_type, "")
                # if struct_type == StructureType.MONASTARY:
                    # print()
                    # print(f"ALL EDGES: {get_all_edges()}")
                    # for edge in get_all_edges():
                        # print(f"-- {edge}: {find_struct_on_edge(tile, edge)}")
                    # print(f"VALID EDGES: {valid_edges}")
                structs_data.append(TileStructData(tile, struct_type, x1, x2, x3, x4, x5, (x, y), valid_edges))
            
        
        tile_data = TileData(structs_data)
        data.append(tile_data)

        score_with_meeple, meeple_edge = tile_data.compute_highest_score_with_meeple(game)
        score_without_meeple = tile_data.compute_effective_score_without_meeple(game)


        if score_with_meeple > best_score_with_meeple:
            best_score_with_meeple = score_with_meeple
            best_meeple_option = i
            best_meeple_edge = meeple_edge
        if score_without_meeple > best_score_without_meeple:
            best_score_without_meeple = score_without_meeple
            best_meeple_less_option = i



    # print("Exiting this function!")
    # # # # print(f"Num meeples: {game.state.me.num_meeples}")
    # print(f"BEST MEEPLE OPTION: {best_meeple_option}")
    # print(f"BEST NON MEEPLE OPTION: {best_meeple_less_option}")
    
    if best_score_with_meeple >= 7 - game.state.me.num_meeples:
        # print("MEEPLE PLACED.")
        # print(f"MEEPLE ON STRUCT: {best_meeple_edge}")
        best_place_meeple = True

        return best_meeple_option, best_place_meeple, best_meeple_edge, data[best_meeple_option]
    else:
        # print("NO MEEPLE PLACED.")
        return best_meeple_less_option, False, None, data[best_meeple_option]



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
    relevant_edge = get_edge_from_delta(x, y, x_prev, y_prev)

    return sum(
        1
        for edge, meeple in tile.internal_claims.items()
        if edge in edges
            and meeple is not None
            and not (
                tile.tile_type in ("H", "I", "R4")
                and tile.internal_edges[edge] == StructureType.CITY
                and edge != relevant_edge
            )
    ) > 0

def is_my_meeple_on_tile_struct(game: Game, tile, edges, x_prev, y_prev):
    me = game.state.me.player_id


    x, y = tile.placed_pos
    relevant_edge = get_edge_from_delta(x, y, x_prev, y_prev)

    return sum(
        1
        for edge, meeple in tile.internal_claims.items()
        if edge in edges
            and meeple is not None
            and meeple.player_id == me
            and not (
                tile.tile_type in ("H", "I", "R4")
                and tile.internal_edges[edge] == StructureType.CITY
                and edge != relevant_edge
            )
    ) > 0

def get_info(game: Game, tile: Tile, x: int, y: int, targStruct: StructureType, starting_edge: str = "") -> tuple:

    grid = game.state.map._grid
    original_tile = grid[y][x]  # Should be None, but just in case

    # Place the tile temporarily
    tile.placed_pos = (x, y)
    grid[y][x] = tile
    
    try:
        return bfs(game, targStruct, x, y, starting_edge)
    finally:
        tile.placed_pos = (0,0)
        grid[y][x] = original_tile

# Obtains as much information as possible about the current game state for a particular structure
# on a specific tile. E.g. find the point capcaity of a city tile on some tile x. 
def bfs (game: Game, targStruct: StructureType, x: int, y: int, starting_edge: str) -> tuple:
    
    grid = game.state.map._grid
    tile = grid[y][x]



    if tile is None:
        return ()

    # print("\n\n\n")
    # print(f"============ START NEW BFS, struct: {targStruct}, xy = ({x}, {y}), edge: {starting_edge}, Tile: {tile}, rotation: {tile.rotation}")
    
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

        if (x, y) in visited:
            continue

        tile = grid[y][x]

        if tile is None:
            continue

        if (x, y) == (x_prev, y_prev) and (targStruct == StructureType.CITY and tile.tile_type in {"H", "I", "R4"}
             or targStruct == StructureType.ROAD_START and tile.tile_type in ("W","X","L")):
            edges = [starting_edge] if starting_edge in tile.internal_edges else []
        else:
            edges = list(find_target_struct_edges(targStruct, tile))

        visited.add((x,y))
        count_structures += 1

        # Debugging meeple detection
        # print()
        # print(f"\n[DEBUG] Visiting tile at ({x},{y})")
        # print(f"[DEBUG] Tile type: {tile.tile_type}, Edges: {edges}")
        # print(f"[DEBUG] Internal claims: {tile.internal_claims}")

        has_any_meeple = is_meeple_on_tile_struct(tile, edges, x_prev, y_prev)
        is_mine = is_my_meeple_on_tile_struct(game, tile, edges, x_prev, y_prev)

        if has_any_meeple:
            # print(f"[DEBUG] Meeple found on structure at ({x},{y})\n")
            meeples_active += 1
        if is_mine:
            # print(f"[DEBUG] It's my meeple at ({x},{y})\n")
            my_meeples_active += 1

        if (x, y) != (x_prev, y_prev) and targStruct in (StructureType.ROAD, StructureType.ROAD_START) and TileModifier.BROKEN_ROAD_CENTER in tile.modifiers:
            continue

        if targStruct == StructureType.CITY:
            if TileModifier.EMBLEM in tile.modifiers:
                count_emblems += 1
            if (x, y) != (x_prev, y_prev) and tile.tile_type in ("H", "I", "R4"):
                continue


        if targStruct == StructureType.GRASS and tile.tile_type in ("F","G","U"):
            continue

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

    
    # print(f"============ END BFS, struct: {targStruct}, xy = ({x}, {y}), edge: {starting_edge}")

    return count_structures, len(open_ends), meeples_active, my_meeples_active, count_emblems





class BotState:
    def __init__(self) -> None:
        self.last_tile: TileModel | None = None
        self.place_meeple = False
        self.place_meeple_on_edge: str | None
        self.place_meeple_on_edge = ""


def main() -> None:
    game = Game()
    bot_state = BotState()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    return handle_place_tile(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    return handle_place_meeple(game, bot_state, q)

        game.send_move(choose_move(query))

def generate_possible_placement_coordinates(grid: list[list[Tile | None]]) -> set[tuple[int,int]]:
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    possible_tile_coords = set()
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if grid[y][x] is not None:
                for i in (x-1, x+1):
                    if grid[y][i] is None:
                        possible_tile_coords.add((i,y))
                for j in (y-1, y+1):
                    if grid[j][x] is None:
                        possible_tile_coords.add((x,j))

    return possible_tile_coords

def generate_placement_configurations(coords: set[tuple[int, int]], game: Game):
    grid = game.state.map._grid

    directions = {
        (0, -1): "top_edge",
        (1, 0): "right_edge",
        (0, 1): "bottom_edge",
        (-1, 0): "left_edge",
    }

    ls = []

    # Precompute all rotated versions of each tile
    rotated_tiles = []
    for tile_index, base_tile in enumerate(game.state.my_tiles):
        for r in range(4):
            tile = deepcopy(base_tile)  # You can optimize this further if tile rotation is pure
            tile.rotate_clockwise(r)
            rotated_tiles.append((tile_index, r, tile))

    for x, y in coords:
        for tile_index, r, tile in rotated_tiles:
            valid = True
            valid_edges = {}

            for (dx, dy), edge in directions.items():
                nx, ny = x + dx, y + dy

                if not (0 <= ny < len(grid) and 0 <= nx < len(grid[0])):
                    continue

                neighbour = grid[ny][nx]
                if neighbour is None:
                    continue

                opp_edge = Tile.get_opposite(edge)

                if neighbour.internal_edges[opp_edge] != tile.internal_edges[edge]:
                    valid = False
                    break

                struct_type = tile.internal_edges[edge]
                valid_edges.setdefault(struct_type, []).append(neighbour)

            if valid:
                ls.append((x, y, r, tile, tile_index, valid_edges))

    return ls

def debug_list_tiles_in_hand (game, bot_state):
    pass

def get_edge(tile, string):
    return tile.internal_edges[string]

def tile_edge_structures_str(tile) -> str:
    string = f"left:{get_edge(tile, "left_edge")},right:{get_edge(tile,"right_edge")},top:{get_edge(tile,"top_edge")},bottom:{get_edge(tile,"bottom_edge")}"

    return string

def debug_list_tile (tile):

    if tile is None:
        print()
        return
    print(tile, f" | edges: {tile_edge_structures_str(tile)})")

def debug_list_adjacent_tiles (game, bot_state, x, y):
    grid = game.state.map._grid

    print("-- Top:", end="")
    debug_list_tile(grid[y-1][x])
    print("-- Left:", end="")
    debug_list_tile(grid[y][x-1])
    print("-- Right:", end="")
    debug_list_tile(grid[y][x+1])
    print("-- Down:", end="")
    debug_list_tile(grid[y+1][x])

def log_u_turn_check(t1: Tile, t2: Tile, axis: str) -> None:
    if axis == "vertical":
        print("    Compare vertical U-turn:")
        print(f"      t1.top    = {t1.internal_edges['top_edge']}")
        print(f"      t2.top    = {t2.internal_edges['top_edge']}")
        print(f"      t1.bottom = {t1.internal_edges['bottom_edge']}")
        print(f"      t2.bottom = {t2.internal_edges['bottom_edge']}")
    else:
        print("    Compare horizontal U-turn:")
        print(f"      t1.left  = {t1.internal_edges['left_edge']}")
        print(f"      t2.left  = {t2.internal_edges['left_edge']}")
        print(f"      t1.right = {t1.internal_edges['right_edge']}")
        print(f"      t2.right = {t2.internal_edges['right_edge']}")

def check_u_turn(t1: Tile, t2: Tile, dir: int) -> bool:

    # 1 is check bottom/top, 2 is check left/right
    if dir not in (1, 2):
        raise ValueError("Invalid dir parameter in u_turn check.")

    match dir:
        case 1:
            if (t1.internal_edges["top_edge"] == StructureType.RIVER and    \
                t1.internal_edges["top_edge"] == t2.internal_edges["top_edge"]) or \
               (t1.internal_edges["bottom_edge"] == StructureType.RIVER and 
                t1.internal_edges["bottom_edge"] == t2.internal_edges["bottom_edge"]):
                return True
            else:
                return False
        case 2:
            if  (t1.internal_edges["left_edge"] == StructureType.RIVER and    \
                t1.internal_edges["left_edge"] == t2.internal_edges["left_edge"]) or \
                (t1.internal_edges["right_edge"] == StructureType.RIVER and \
                t1.internal_edges["right_edge"] == t2.internal_edges["right_edge"]):
                return True
            else:
                return False

def is_river_phase(game: Game) -> bool:
    """
    Returns True if any tile in the current player's hand has a river edge,
    indicating the river phase is still active.
    """
    for tile in game.state.my_tiles:
        edges = tile.internal_edges
        if (
            edges.left_edge == StructureType.RIVER
            or edges.right_edge == StructureType.RIVER
            or edges.top_edge == StructureType.RIVER
            or edges.bottom_edge == StructureType.RIVER
        ):
            return True
    return False               

def is_curved_river_tile(tile) -> bool:
    
    left_edge = tile.internal_edges["left_edge"]
    right_edge = tile.internal_edges["right_edge"]
    top_edge = tile.internal_edges["top_edge"]
    bottom_edge = tile.internal_edges["bottom_edge"]

    edges = (left_edge, right_edge, top_edge, bottom_edge)

    if sum([1 for x in edges if x == StructureType.RIVER]) < 2:
        return False

    if left_edge == StructureType.RIVER and right_edge == StructureType.RIVER:
        return False

    if top_edge == StructureType.RIVER and bottom_edge == StructureType.RIVER:
        return False

    return True
     
def debug_decision(game: Game, placement_list):
    print("\n=== MY CURRENT TILE HAND ===")
    for idx, tile in enumerate(game.state.my_tiles):
        print(f"[{idx}] Tile Type: {tile.tile_type}")
        print(f"  Rotation: {tile.rotation}")
        print("  Edge Structures:")
        print(f"    Left   -> {tile.internal_edges.left_edge}")
        print(f"    Right  -> {tile.internal_edges.right_edge}")
        print(f"    Top    -> {tile.internal_edges.top_edge}")
        print(f"    Bottom -> {tile.internal_edges.bottom_edge}")
        print("")

        print("\n=== ALL VALID TILE PLACEMENTS ===")
    if not placement_list:
        print("No valid placements found.")
    else:
        for i, (x, y, r, tile, tile_idx, connection_types) in enumerate(placement_list):
            print(f"[{i}] Tile {tile.tile_type} (index {tile_idx})")
            print(f"  Position: ({x}, {y})")
            print(f"  Rotation: {r}")
            print("  Edge Structures:")
            print(f"    Left   -> {tile.internal_edges.left_edge}")
            print(f"    Right  -> {tile.internal_edges.right_edge}")
            print(f"    Top    -> {tile.internal_edges.top_edge}")
            print(f"    Bottom -> {tile.internal_edges.bottom_edge}")
            print(f"  Connected structures: {[s.name for s in connection_types]}")
            print("")


def debug_meeples(game):
    # === DEBUG MESSAGES === 
    print(f"MY POINTS: {game.state.points}")
    print("Meeples remaining per player:")
    for player_id, meeples in game.state.players_meeples.items():
        print(f"  Player {player_id}: {meeples}")
    print("Meeples placed per player:")


    for player_id in game.state.players:
        meeples_placed = game.state.get_meeples_placed_by(player_id)
        print(f"  Player {player_id}: {len(meeples_placed)}: {meeples_placed}")
    
    print(flush=True)

    #  === END DEBUG ===

def debug_selection(game: Game, bot_state: BotState, tile: Tile, x, y):
    grid = game.state.map._grid

    print("\n=== SELECTED TILE PLACEMENT DEBUG ===")
    print(f"Chosen Tile Type: {tile.tile_type}")
    print(f"Chosen Tile Rotation: {tile.rotation}")
    print("Chosen Tile Edge Structures:")
    print(f"  Left   -> {tile.internal_edges.left_edge}")
    print(f"  Right  -> {tile.internal_edges.right_edge}")
    print(f"  Top    -> {tile.internal_edges.top_edge}")
    print(f"  Bottom -> {tile.internal_edges.bottom_edge}")
    print(f"Placing at position: ({x}, {y})\n")

    directions = {
        (0, -1): "top_edge",
        (0, 1): "bottom_edge",
        (-1, 0): "left_edge",
        (1, 0): "right_edge",
    }
    opposite_edges = {
        "top_edge": "bottom_edge",
        "bottom_edge": "top_edge",
        "left_edge": "right_edge",
        "right_edge": "left_edge",
    }

# Check all 4 neighboring tiles
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        neighbor_tile = game.state.map._grid[ny][nx]
        direction = directions[(dx, dy)]
        opp_direction = opposite_edges[direction]

        print(f"-- Neighbor at ({nx}, {ny}) on {direction}:")
        if neighbor_tile is None:
            print("  [EMPTY]")
        else:
            print(f"  Neighbor Tile Type: {neighbor_tile.tile_type}")
            print(f"  Neighbor Rotation: {neighbor_tile.rotation}")
            print("  Neighbor Edge Structures:")
            print(f"    Left   -> {neighbor_tile.internal_edges['left_edge']}")
            print(f"    Right  -> {neighbor_tile.internal_edges['right_edge']}")
            print(f"    Top    -> {neighbor_tile.internal_edges['top_edge']}")
            print(f"    Bottom -> {neighbor_tile.internal_edges['bottom_edge']}")
            print("  ↔ Comparing:")
            print(f"    Your {direction} = {tile.internal_edges[direction]}")
            print(f"    vs Neighbor {opp_direction} = {neighbor_tile.internal_edges[opp_direction]}")
            match = tile.internal_edges[direction] == neighbor_tile.internal_edges[opp_direction]
            print(f"    → Match: {'YES ✅' if match else 'NO ❌'}")
        print()

    assert(bot_state.last_tile is not None)
    print(f"🚀 Sending tile: {bot_state.last_tile.tile_type}, pos={bot_state.last_tile.pos}, rotation={bot_state.last_tile.rotation}")    
    print(flush=True)

    print_map(grid, range(72, 89))

def filter_non_river_connections(placement_list):

    i = 0
    while i < len(placement_list):

        # If no rivers connect neighbouring tilesd, break
        # print(f"KEYS: {list(placement_list[i][5].keys())}")
        if StructureType.RIVER not in placement_list[i][5].keys():
            # print("Filtered")
            
            # print("  No RIVER connection — discarding")
            placement_list.pop(i)
            continue
        
        # Check for U-turns. We check if neighbouring river card also has
        # a river on the adjacent side to the connection (e.g. connect on top/bot,
        # but both tiles have a river edge on the left)
        
        neighbours: list[list[Tile]]
        neighbours = list(placement_list[i][5].values())

        assert(len(neighbours) == 1)
        assert(len(neighbours[0]) == 1)

        neighbour: Tile
        neighbour = deepcopy(neighbours[0][0])
        
        nx, ny = (0, 0)
        
        if neighbour.placed_pos is not None:
            nx, ny = neighbour.placed_pos 
        
        x, y, r, currTile, tile_idx = placement_list[i][:5]

        if x != nx:
            # debug_turn_check(currTile, neighbour, "vertical")
            if check_u_turn(currTile, neighbour, 1):
               # print("  U-turn detected (vertical) — discarding")
               placement_list.pop(i)
               continue
        elif y != ny:
            # debug_u_turn_check(currTile, neighbour, "horizontal")
            if check_u_turn(currTile, neighbour, 2):
                # print("  U-turn detected (horizontal) — discarding")  
                placement_list.pop(i)
                continue

        i += 1

    chosen = 0
    max_points = -99

    for i, t in enumerate(placement_list):
        points = 0
        
        x, y, r, currTile = t[:4]
        if y < 85:
            if currTile.internal_edges["top_edge"] == StructureType.RIVER:
                points += 1
            if currTile.internal_edges["bottom_edge"] == StructureType.RIVER:
                points -= 1
        if y > 85:
            if currTile.internal_edges["bottom_edge"] == StructureType.RIVER:
                points += 1
            if currTile.internal_edges["top_edge"] == StructureType.RIVER:
                points -= 1
        if x < 85:
            if currTile.internal_edges["left_edge"] == StructureType.RIVER:
                points += 1
            if currTile.internal_edges["right_edge"] == StructureType.RIVER:
                points -= 1
        if x > 85:
            if currTile.internal_edges["right_edge"] == StructureType.RIVER:
                points += 1
            if currTile.internal_edges["left_edge"] == StructureType.RIVER:
                points -= 1

        if points > max_points:
            chosen = i
            max_points = points

    selected = placement_list[chosen]
    placement_list.clear()
    placement_list.append(selected)


def debug_chosen_option(tile, data, meeple, edge):

    print(f"BEST OPTION IDENTIFIED: {tile}")
    print("NUMBER OF MEEPLES ON STRUCTS?")
    print(data)
    for feature in data.features_info:
        print(f"STRUCT: {feature.struct_type} | NUM_MEEPLES: {feature.meeples_active}")
    print(f"MEEPLE?: {meeple}")
    if meeple and edge is not None:
        print(f"WHERE?:   {edge}")
        print(f"WHICH IS? {find_struct_on_edge(tile, edge)}")

def handle_place_tile(
    game: Game, bot_state: BotState, query: QueryPlaceTile
) -> MovePlaceTile:
    """Place a tile in the first valid location found on the board - brute force"""
    grid = game.state.map._grid

    ## Find availabile insertion positions.
    
    river_phase = is_river_phase(game)

    # e.g. [(x1, y1), (x2, y2) ... (xn, yn)]
    placement_coords: set[tuple[int,int]]
    placement_coords = generate_possible_placement_coordinates(grid)

    # e.g. [(tile_x, tile_y, tile_rotation, tile_object, tile_index_in_hand, tile_valid_edges)]
    # valid edges is for the river phase filtering
    placement_list: list[tuple[int,int,int,Tile, int,dict[StructureType,list[Tile]]]]
    placement_list = generate_placement_configurations(placement_coords, game)


    # If river phase, only select river cards. We remove any cards not connecting to rivers here from the list.
    if river_phase:
        filter_non_river_connections(placement_list)
            
    idx = random.randrange(len(placement_list))


    best_option, meeple, edge, data = analyse_board (game, placement_list)


    bot_state.place_meeple = meeple
    bot_state.place_meeple_on_edge = edge


    x, y, r, tile, tile_idx, connection_types = placement_list[best_option]

    # debug_chosen_option(tile, data, meeple, edge)
    # print(connection_types)


    tile = deepcopy(game.state.my_tiles[tile_idx])  # Fresh, unrotated tile

    tile.rotation = 0
    tile.rotate_clockwise(r)
    tile.placed_pos = (x, y)
    bot_state.last_tile = tile._to_model()
    
    debug_selection(game, bot_state, tile, x, y)

    game.state.my_tiles[tile_idx].rotate_clockwise(r)
    return game.move_place_tile(query, tile._to_model(), tile_idx)

def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeple | MovePlaceMeeplePass:
    """Try to place a meeple on the tile just placed, or pass."""
    assert bot_state.last_tile is not None

    x, y = bot_state.last_tile.pos
    tile = game.state.map._grid[y][x]

    assert tile is not None

    tile_model = bot_state.last_tile
    bot_state.last_tile = None

    # debug_meeples(game)

    if bot_state.place_meeple is True:
        assert(bot_state.place_meeple_on_edge is not None)
        return game.move_place_meeple(query, tile_model, placed_on=bot_state.place_meeple_on_edge)
    
    # print(f"CURRENT SCORE: {game.state.me.points}", flush=True)
    bot_state.place_meeple = False
    return game.move_place_meeple_pass(query)


if __name__ == "__main__":
    main()
