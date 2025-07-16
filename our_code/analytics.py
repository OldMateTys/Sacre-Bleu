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

class TileStructData:

    def __init__(self, struct_type, num_structs, num_open_ends, meeples_active, my_meeples_active, num_emblems, pos, edges):

        # Convert ROAD_START to ROAD for code simplification
        self.struct_type = struct_type if struct_type != StructureType.ROAD_START else StructureType.ROAD
        self.num_structs = num_structs
        self.num_open_ends = num_open_ends
        self.meeples_active = meeples_active
        self.my_meeples_active = my_meeples_active
        self.num_emblems = num_emblems
        self.x, self.y = pos
        self.is_completed = num_open_ends == 0
        self.edges = edges
        self.free_point = self.free_point_check()


    def compute_score_with_meeple(self) -> float:
        # Penalize contested or incomplete structures
        if not self.free_point:
            return 0.5 * self.compute_incomplete_score()
        
        if self.is_completed:
            return self.compute_completed_score()
        
        return self.compute_incomplete_score() * 0.8  # Boost if not contested

    def compute_score_without_meeple(self) -> float:
        # Only get points if it's completed and I already own it
        if self.my_meeples_active > 0 and self.is_completed:
            return self.compute_completed_score()
        return 0.0

    def compute_completed_score(self):
        
        match self.struct_type:
            case StructureType.CITY:
                return 2 * self.num_structs + 2 * self.num_emblems
            case StructureType.MONASTARY:
                # TO BE COMPLETED
                pass
            case StructureType.ROAD:
                return self.num_structs

        return 0


    def compute_incomplete_score(self):
        match self.struct_type:
            case StructureType.CITY:
                return 1 * self.num_structs + 1 * self.num_emblems
            case StructureType.MONASTARY:
                # TO BE COMPLETED
                pass
            case StructureType.ROAD:
                return self.num_structs
            case StructureType.GRASS:
                pass
        
        return 0

    def free_point_check(self):
        if not self.is_completed:
            return False
        if sum(self.meeples_active.values()) <= 2 * self.my_meeples_active:
            return True
        return False


class TileData:
    def __init__(self, features: list[TileStructData]):
        self.features_info = features


    def compute_highest_score_with_meeple(self):
        best_edge = None
        highest_score = float('-inf')

        meeple_scores = []
        non_meeple_scores = []

        for struct in self.features_info:
            meeple_scores.append(struct.compute_score_with_meeple())
            non_meeple_scores.append(struct.compute_score_without_meeple())

        total_non_meeple_scores = sum(non_meeple_scores)

        for i in range(len(self.features_info)):
            if highest_score < total_non_meeple_scores - non_meeple_scores[i] + meeple_scores[i]:
                highest_score = total_non_meeple_scores - non_meeple_scores[i] + meeple_scores[i]
                best_edge = self.features_info[i].edges[0]

        return highest_score, best_edge

    def compute_effective_score_without_meeple(self):

        scores = []
        for struct in self.features_info:
            scores.append(struct.compute_score_without_meeple())

        return sum(scores)

    def compute_impact_of_meeple(self):
        meeple_score, edge = self.compute_highest_score_with_meeple()
        non_meeple_score = self.compute_effective_score_without_meeple()

        return meeple_score - non_meeple_score, edge

            
def analyse_board (game: Game, placement_list: list[tuple[int,int,int,Tile, int,dict[StructureType,list[Tile]]]]):
    
    best_score = float('-inf')
    best_option = 0
    best_place_meeple = False
    best_meeple_edge = None

    data = []


    for i, option in enumerate(placement_list):
        x, y, r, tile, tile_idx, connected_comps = option
        
        placeable_structures = game.state.get_placeable_structures(tile._to_model())
        structs_data = []


        DISCONNECTED_CITIES = {"H", "I", "R4"}
        DISCONNECTED_ROADS = {"L", "W", "X"}

        for struct_type in set(placeable_structures.values()):

            if tile.tile_type in DISCONNECTED_CITIES and struct_type == StructureType.CITY:

                valid_edges = [edge for edge in Tile.get_edges() if tile.internal_edges[edge] == StructureType.CITY]
                for edge in valid_edges:
                    x1, x2, x3, x4, x5 = bfs(game, struct_type, x, y, edge)
                    structs_data.append(TileStructData(struct_type, x1, x2, x3, x4, x5, (x, y), valid_edges))

            elif tile.tile_type in DISCONNECTED_ROADS and struct_type == StructureType.ROAD_START:

                valid_edges = [edge for edge in Tile.get_edges() if tile.internal_edges[edge] == StructureType.ROAD_START]
                for edge in valid_edges:
                    x1, x2, x3, x4, x5 = bfs(game, struct_type, x, y, edge)
                    structs_data.append(TileStructData(struct_type, x1, x2, x3, x4, x5, (x, y), valid_edges))

            else:
                valid_edges = [edge for edge in Tile.get_edges() if tile.internal_edges[edge] == struct_type]
                x1, x2, x3, x4, x5 = bfs(game, struct_type, x, y, "")
                structs_data.append(TileStructData(struct_type, x1, x2, x3, x4, x5, (x, y), valid_edges))
            
        
        tile_data = TileData(structs_data)
        score_with_meeple, meeple_edge = tile_data.compute_highest_score_with_meeple()
        score_without_meeple = tile_data.compute_effective_score_without_meeple()

        score_delta = score_with_meeple - score_without_meeple
        score = score_with_meeple  # Or choose a smarter mix

        if score > best_score:
            best_score = score
            best_option = i
            best_place_meeple = score_delta > 1.5  # You can tune this threshold
            best_meeple_edge = meeple_edge if best_place_meeple else None


        data.append((option, score_with_meeple, score_without_meeple, meeple_edge))


    return best_option, best_place_meeple, best_meeple_edge

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
    for edge, meeple in tile.internal_claims.items():

        # Special handling for CITY on disconnected city tiles
        if tile.tile_type in ("H", "I", "R4") and tile.internal_edges[edge] == StructureType.CITY:
            # Only check the edge facing the previous tile
            if edge != get_edge_from_delta(tile.x, tile.y, x_prev, y_prev):
                continue

        # Skip irrelevant edges
        if edge not in edges:
            continue

        if meeple is not None:
            return meeple.player_id

    return None

def is_my_meeple_on_tile_struct(game: Game, tile, edges, x_prev, y_prev):
    me = game.state.me.player_id


    relevant_edge = get_edge_from_delta(tile.x, tile.y, x_prev, y_prev)

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
    
    que = deque()
    que.append((x, y, x, y))

    visited = set()
    
    count_structures = 0
    open_ends = set()
    meeples_active = {}
    my_meeples_active = 0
    count_emblems = 0

    while que:
        x, y, x_prev, y_prev = que.popleft()

        if (x, y) in visited:
            continue

        tile = grid[y][x]

        if tile is None:
            continue

        if (x, y) == (x_prev, y_prev) and targStruct == StructureType.CITY and tile.tile_type in {"H", "I", "R4"}:
            edges = [starting_edge] if starting_edge in tile.internal_edges else []
        else:
            edges = find_target_struct_edges(targStruct, tile)

        visited.add((x,y))
        count_structures += 1

        meeple = is_meeple_on_tile_struct(tile, edges, x_prev, y_prev)

        if meeple is not None:
            if meeple not in meeples_active:
                meeples_active[meeple] = 1
            else:
                meeples_active[meeple] += 1

        my_meeples_active += 1 if is_my_meeple_on_tile_struct(game, tile, edges, x_prev, y_prev) else 0

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
