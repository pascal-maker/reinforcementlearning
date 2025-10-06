"""
A* Pathfinding Visualization with Multiple Heuristic Functions
---------------------------------------------------------------
This demo extends a basic A* visualization to support:
    1. Euclidean Distance
    2. Manhattan Distance
    3. Chebyshev Distance
    4. Weighted Euclidean Distance (user-adjustable weight)

User controls:
    - Space: start/pause algorithm
    - S + Click: set Start node
    - G + Click: set Goal node
    - Left/Right mouse: draw/erase walls
    - D: toggle diagonal movement
    - R: reset algorithm
    - C: clear grid
    - 1/2/3/4: switch heuristic functions
    - [ and ]: decrease/increase weight (for weighted Euclidean)

Visual feedback:
    - Path glow effect
    - Walls, open/closed nodes, start & goal markers
"""

import pygame
import math
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Set, Callable

# --- Config: Window, Grid, Colors ----------------------------------------

W, H = 900, 700                  # Window size
COLS, ROWS = 45, 35              # Grid size in cells
CELL = min((W-40)//COLS, (H-40)//ROWS)  # Cell pixel size
OX, OY = 20, 20                  # Grid offset (margin)
GRID_W, GRID_H = COLS*CELL, ROWS*CELL

# Soft dark UI colors
BG = (18, 18, 22)
GRID_BG = (28, 28, 34)
GRID_LINE = (50, 54, 63)
WALL = (70, 76, 90)
START = (72, 207, 173)
GOAL = (245, 130, 130)
OPEN = (120, 170, 255)
CLOSED = (160, 140, 255)
PATH = (250, 210, 85)
TEXT = (225, 228, 235)
MUTED = (165, 170, 180)
ROUND = 6                        # Rounded corner radius

# --- Heuristic Functions -------------------------------------------------
# Each returns an estimate (float) of the distance between two points

def euclidean_heuristic(pos1, pos2) -> float:
    """Classic straight-line distance."""
    x1, y1 = (pos1.x, pos1.y) if hasattr(pos1, "x") else pos1
    x2, y2 = (pos2.x, pos2.y) if hasattr(pos2, "x") else pos2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def manhattan_heuristic(pos1, pos2) -> float:
    """Sum of horizontal + vertical differences."""
    x1, y1 = (pos1.x, pos1.y) if hasattr(pos1, "x") else pos1
    x2, y2 = (pos2.x, pos2.y) if hasattr(pos2, "x") else pos2
    return abs(x1 - x2) + abs(y1 - y2)

def chebyshev_heuristic(pos1, pos2) -> float:
    """Maximum of horizontal or vertical distance."""
    x1, y1 = (pos1.x, pos1.y) if hasattr(pos1, "x") else pos1
    x2, y2 = (pos2.x, pos2.y) if hasattr(pos2, "x") else pos2
    return max(abs(x1 - x2), abs(y1 - y2))

def weighted_euclidean_heuristic(pos1, pos2, weight: float = 1.0) -> float:
    """Euclidean distance scaled by user-provided weight."""
    return weight * euclidean_heuristic(pos1, pos2)

def octile_heuristic(a, b) -> float:
    """Admissible heuristic for 8-directional movement."""
    x1, y1 = (a.x, a.y) if hasattr(a, "x") else a
    x2, y2 = (b.x, b.y) if hasattr(b, "x") else b
    dx, dy = abs(x1 - x2), abs(y1 - y2)
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

# --- Node Dataclass ------------------------------------------------------
@dataclass(frozen=True)
class Node:
    """Immutable grid cell identified by (x, y) coordinates."""
    x: int
    y: int
    def __iter__(self):  # Allows unpacking like: x, y = node
        yield self.x; yield self.y

# --- Grid Helper Functions ----------------------------------------------
def in_bounds(x:int,y:int)->bool:
    """Return True if (x,y) lies within grid dimensions."""
    return 0<=x<COLS and 0<=y<ROWS

def to_cell(mx:int,my:int)->Optional[Node]:
    """Convert mouse pixel position → grid cell Node."""
    if not (OX <= mx < OX+GRID_W and OY <= my < OY+GRID_H):
        return None
    return Node((mx-OX)//CELL, (my-OY)//CELL)

def rect_of(n:Node)->pygame.Rect:
    """Get drawing rectangle for node position."""
    return pygame.Rect(OX + n.x*CELL+1, OY + n.y*CELL+1, CELL-2, CELL-2)

# --- Core A* Components --------------------------------------------------
def neighbors(n:Node, diag:bool)->Iterable[Node]:
    """Generate valid 4-way or 8-way neighboring cells."""
    dirs4 = [(1,0),(-1,0),(0,1),(0,-1)]
    dirs8 = dirs4 + [(1,1),(1,-1),(-1,1),(-1,-1)]
    for dx,dy in (dirs8 if diag else dirs4):
        nx,ny = n.x+dx, n.y+dy
        if in_bounds(nx,ny):
            yield Node(nx,ny)

def step_cost(a:Node,b:Node)->float:
    """Return cost between two adjacent cells (1 or √2)."""
    return math.sqrt(2) if (a.x!=b.x and a.y!=b.y) else 1.0

def corner_cut(grid, a:Node, b:Node)->bool:
    """Prevent diagonal move through wall corners."""
    if a.x==b.x or a.y==b.y: return False
    return grid[a.x][b.y] or grid[b.x][a.y]

def astar(grid, start:Node, goal:Node, diag:bool,
          heuristic_fn: Callable[[Node, Node], float]):
    """
    A* pathfinding algorithm implemented as a generator.

    Parameters:
        grid: 2D array (0=free, 1=wall)
        start, goal: Node positions
        diag: allow diagonal movement
        heuristic_fn: distance estimate between two nodes

    Yields:
        ("state", current, open_set, closed, came_from) each step
        ("done", current, open_set, closed, came_from) on completion
    """
    # --- Initialize containers ---
    openh:List[Tuple[float,int,Node]]=[]  # (f, tie_breaker, node)
    g:Dict[Node,float]={start:0.0}        # g = cost from start
    came:Dict[Node,Optional[Node]]={start:None}  # path reconstruction
    seen:Set[Node]={start}
    t=0                                   # tie-breaker for heap
    heapq.heappush(openh,(heuristic_fn(start,goal),t,start)); t+=1
    closed:Set[Node]=set()

    # --- Main loop ---
    while openh:
        _,_,cur = heapq.heappop(openh)    # node with lowest f
        if cur in closed: continue
        closed.add(cur)
        yield "state", cur, set(seen), set(closed), dict(came)
        if cur==goal:                     # goal reached
            yield "done", cur, set(seen), set(closed), dict(came)
            return

        # Expand neighbors
        for nb in neighbors(cur,diag):
            if grid[nb.x][nb.y]:          # wall check
                continue
            if diag and corner_cut(grid,cur,nb):
                continue
            ng = g[cur] + step_cost(cur,nb)
            if nb not in g or ng < g[nb]-1e-9:
                g[nb]=ng
                came[nb]=cur
                f = ng + heuristic_fn(nb,goal)
                heapq.heappush(openh,(f,t,nb)); t+=1
                seen.add(nb)

    # --- No path found ---
    yield "done", None, set(), set(closed), dict(came)

def reconstruct(came:Dict[Node,Optional[Node]], end:Node)->List[Node]:
    """Reconstruct final path from end → start using came_from map."""
    if end not in came: return []
    p=[end]; cur=end
    while came[cur] is not None:
        cur=came[cur]; p.append(cur)
    p.reverse()
    return p

# --- Main Visualization App ---------------------------------------------
def main():
    """Pygame-based visualization of A* with selectable heuristics."""
    pygame.init()
    pygame.display.set_caption("Simple A* Pathfinding (Multi-Heuristic)")
    screen = pygame.display.set_mode((W,H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,DejaVuSansMono,monospace", 18)
    small = pygame.font.SysFont("Menlo,Consolas,DejaVuSansMono,monospace", 14)

    # --- Initial setup ---
    grid = [[0 for _ in range(ROWS)] for _ in range(COLS)]
    start, goal = Node(2,2), Node(COLS-3, ROWS-3)
    diag = True

    # Heuristic configuration
    WEIGHTS = [0.5, 1.0, 1.5, 2.0]
    w_idx = 1
    mode = "euclidean"  # default

    # Helper to switch heuristics dynamically
    def make_heuristic():
        if mode == "euclidean":
            return euclidean_heuristic
        elif mode == "manhattan":
            return manhattan_heuristic
        elif mode == "chebyshev":
            return chebyshev_heuristic
        elif mode == "weighted":
            return lambda a,b: weighted_euclidean_heuristic(a,b, WEIGHTS[w_idx])
        else:
            return octile_heuristic

    # --- Runtime state ---
    running=False
    gen=None
    path:List[Node]=[]

    def reset():
        """Reinitialize the algorithm with current settings."""
        nonlocal gen, running, path
        gen = astar(grid,start,goal,diag, make_heuristic())
        running=False
        path=[]

    reset()

    # --- Event + Draw Loop ----------------------------------------------
    while True:
        # Handle keyboard & mouse
        for e in pygame.event.get():
            if e.type==pygame.QUIT: return
            if e.type==pygame.KEYDOWN:
                if e.key==pygame.K_ESCAPE: return
                if e.key==pygame.K_SPACE: running=not running
                if e.key==pygame.K_r: reset()
                if e.key==pygame.K_c:
                    for x in range(COLS):
                        for y in range(ROWS):
                            grid[x][y]=0
                    reset()
                if e.key==pygame.K_d:
                    diag = not diag; reset()

                # Heuristic hotkeys
                if e.key==pygame.K_1: mode="euclidean"; reset()
                if e.key==pygame.K_2: mode="manhattan"; reset()
                if e.key==pygame.K_3: mode="chebyshev"; reset()
                if e.key==pygame.K_4: mode="weighted"; reset()
                if e.key==pygame.K_LEFTBRACKET:   # '[' decrease weight
                    if mode=="weighted":
                        w_idx = max(0, w_idx-1); reset()
                if e.key==pygame.K_RIGHTBRACKET:  # ']' increase weight
                    if mode=="weighted":
                        w_idx = min(len(WEIGHTS)-1, w_idx+1); reset()

            # Mouse input: draw walls or set start/goal
            if e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
                buttons = pygame.mouse.get_pressed(3)
                pos = pygame.mouse.get_pos()
                cell = to_cell(*pos)
                keys = pygame.key.get_pressed()
                if cell:
                    if keys[pygame.K_s] and not grid[cell.x][cell.y] and cell!=goal:
                        start = cell; reset()
                    elif keys[pygame.K_g] and not grid[cell.x][cell.y] and cell!=start:
                        goal = cell; reset()
                    elif buttons[0] and cell not in (start,goal):
                        grid[cell.x][cell.y]=1
                    elif buttons[2] and cell not in (start,goal):
                        grid[cell.x][cell.y]=0

        # --- Step algorithm if running ---
        if running and gen:
            try:
                tag, cur, open_set, closed, came = next(gen)
                if tag=="done":  # finished
                    running=False
                    if goal in came:
                        path = reconstruct(came, goal)
                else:
                    # show "live" shortest path to nearest open node
                    heuristic_fn = make_heuristic()
                    if open_set:
                        best = min(open_set, key=lambda n: heuristic_fn(n,goal))
                        path = reconstruct(came, best)
            except StopIteration:
                running=False

        # --- Drawing section ---------------------------------------------
        screen.fill(BG)
        pygame.draw.rect(screen, GRID_BG, (OX-6, OY-6, GRID_W+12, GRID_H+12), border_radius=12)

        # Grid lines
        for x in range(COLS+1):
            X = OX + x*CELL
            pygame.draw.line(screen, GRID_LINE, (X, OY), (X, OY+GRID_H))
        for y in range(ROWS+1):
            Y = OY + y*CELL
            pygame.draw.line(screen, GRID_LINE, (OX, Y), (OX+GRID_W, Y))

        # Walls
        for x in range(COLS):
            for y in range(ROWS):
                if grid[x][y]:
                    pygame.draw.rect(screen, WALL, (OX+x*CELL+1, OY+y*CELL+1, CELL-2, CELL-2), border_radius=ROUND)

        # Path overlay (glow effect)
        overlay = pygame.Surface((GRID_W, GRID_H), pygame.SRCALPHA)
        for i,n in enumerate(path):
            alpha = 120 if i==len(path)-1 else 90
            r = rect_of(n)
            pygame.draw.rect(overlay, (*PATH, alpha), (r.x-OX, r.y-OY, r.w, r.h), border_radius=ROUND)
        screen.blit(overlay,(OX,OY))

        # Start & goal nodes
        pygame.draw.rect(screen, START, rect_of(start), border_radius=ROUND)
        pygame.draw.rect(screen, GOAL, rect_of(goal), border_radius=ROUND)

        # HUD text (controls + heuristic mode)
        hudy = OY+GRID_H+14
        mode_label = {
            "euclidean": "Euclidean",
            "manhattan": "Manhattan",
            "chebyshev": "Chebyshev",
            "weighted": f"Weighted Euclidean (w={WEIGHTS[w_idx]:.1f})",
        }[mode]

        txt1 = f"Space: run/pause  |  S/G: set start/goal  |  D: diagonals: {'ON' if diag else 'OFF'}  |  R: reset  C: clear"
        txt2 = f"Heuristic [1:Euclid  2:Manhattan  3:Chebyshev  4:Weighted, '['/']' weight]:  {mode_label}"
        screen.blit(small.render(txt1, True, MUTED), (OX, hudy))
        screen.blit(small.render(txt2, True, MUTED), (OX, hudy+18))
        screen.blit(font.render("A* Pathfinding", True, TEXT), (OX, 6))

        pygame.display.flip()
        clock.tick(60)

# --- Run program ---------------------------------------------------------
if __name__ == "__main__":
    main()
