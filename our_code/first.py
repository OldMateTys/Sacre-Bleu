#### DISCLAIMER ####
# This was produced with the assistance of GPT 4o

import random
from helper.game import Game
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

place_meeple = False
place_on : str
place_on = ""

class BotState:
    def __init__(self) -> None:
        self.last_tile: TileModel | None = None


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

def generate_placement_configurations(coords: set[tuple[int,int]], game: Game):

    grid: list[list[Tile | None]]
    grid = game.state.map._grid

    directions = {
        (0, -1): "top_edge",
        (1, 0): "right_edge",
        (0, 1): "bottom_edge",
        (-1, 0): "left_edge",
    }

    ls: list[tuple[int,int,int,Tile,int,dict[StructureType,list[Tile]]]]
    ls = []

    for x, y in coords:
        for tile_index, base_tile in enumerate(game.state.my_tiles):
            for r in range(4):
                tile = deepcopy(base_tile)
                tile.rotate_clockwise(r)
                valid = True
                valid_edges: dict[StructureType,list[Tile]]
                valid_edges = {}

                for dx, dy in directions.keys():
                    
                    #check tile below
                    neighbour = deepcopy(grid[y+dy][x+dx])

                    if neighbour is None:
                        continue

                    edge = directions[(dx, dy)]
                    opp_edge = Tile.get_opposite(edge)
                    
                    
                    if neighbour.internal_edges[opp_edge] != tile.internal_edges[edge]:
                        valid = False
                        break

                    if tile.internal_edges[edge] not in valid_edges:
                        valid_edges[tile.internal_edges[edge]] = [neighbour]
                    else:
                        valid_edges[tile.internal_edges[edge]].append(neighbour)
                if valid:
                    ls.append((x, y,r, tile, tile_index, valid_edges))


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
        print(f"KEYS: {list(placement_list[i][5].keys())}")
        if StructureType.RIVER not in placement_list[i][5].keys():
            print("Filtered")
            
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
            log_u_turn_check(currTile, neighbour, "vertical")
            if check_u_turn(currTile, neighbour, 1):
               print("  U-turn detected (vertical) — discarding")
               placement_list.pop(i)
               continue
        elif y != ny:
            log_u_turn_check(currTile, neighbour, "horizontal")
            if check_u_turn(currTile, neighbour, 2):
                print("  U-turn detected (horizontal) — discarding")  
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

    debug_decision(game, placement_list)

    # If river phase, only select river cards. We remove any cards not connecting to rivers here from the list.
    if river_phase:
        filter_non_river_connections(placement_list)
            
    idx = random.randrange(len(placement_list))
    idx, meeple, edge = analyse_board (game, placement_list)

    x, y, r, tile, tile_idx, connection_types = placement_list[idx]

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
    structures = game.state.get_placeable_structures(bot_state.last_tile)

    print(structures)

    x, y = bot_state.last_tile.pos
    tile = game.state.map._grid[y][x]

    assert tile is not None

    tile_model = bot_state.last_tile
    bot_state.last_tile = None

    
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

    if place_meeple is True:
        return game.move_place_meeple(query, tile_model, placed_on=place_on)

    return game.move_place_meeple_pass(query)


if __name__ == "__main__":
    main()
