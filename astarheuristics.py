"""
A* Pathfinding Visualization with Multiple Heuristic Functions
---------------------------------------------------------------
Extends a basic A* visualization to support:
    1. Euclidean Distance
    2. Manhattan Distance
    3. Chebyshev Distance
    4. Weighted Euclidean Distance (user-adjustable weight)

Also includes benchmark mode (Task 2) and obstacle-pattern generator (Task 3).

Controls:
    Space – start/pause
    S/G + click – set Start/Goal
    Left/Right mouse – draw/erase walls
    D – toggle diagonals
    R – reset algorithm
    C – clear grid
    1/2/3/4 – switch heuristic
    [ and ] – change weight (for weighted Euclidean)
    P – cycle obstacle pattern families (off → linear → geometric → maze → off)
    , and . – cycle pattern variants within the family
    B – run benchmark table in terminal
"""

import pygame, math, heapq, random, time, argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Set, Callable

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
W, H = 900, 700
COLS, ROWS = 45, 35
CELL = min((W-40)//COLS, (H-40)//ROWS)
OX, OY = 20, 20
GRID_W, GRID_H = COLS*CELL, ROWS*CELL

# Colors
BG = (18, 18, 22)
GRID_BG = (28, 28, 34)
GRID_LINE = (50, 54, 63)
WALL = (70, 76, 90)
START = (72, 207, 173)
GOAL = (245, 130, 130)
PATH = (250, 210, 85)
TEXT = (225, 228, 235)
MUTED = (165, 170, 180)
ROUND = 6

# --------------------------------------------------------------------------- #
# NODE + HELPERS
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Node:
    x: int
    y: int
    def __iter__(self):
        yield self.x; yield self.y

def in_bounds(x:int,y:int)->bool:
    """Return True when the grid index is inside the drawable board."""
    return 0<=x<COLS and 0<=y<ROWS

def to_cell(mx:int,my:int)->Optional[Node]:
    """Translate mouse coordinates into a grid node or None if outside."""
    if not (OX <= mx < OX+GRID_W and OY <= my < OY+GRID_H):
        return None
    return Node((mx-OX)//CELL, (my-OY)//CELL)

def rect_of(n:Node)->pygame.Rect:
    """Compute the pixel rect used to draw a cell for node n."""
    return pygame.Rect(OX + n.x*CELL+1, OY + n.y*CELL+1, CELL-2, CELL-2)

# --------------------------------------------------------------------------- #
# HEURISTICS (Task 1)
# --------------------------------------------------------------------------- #
def euclidean_heuristic(p1,p2)->float:
    x1,y1=(p1.x,p1.y) if hasattr(p1,"x") else p1
    x2,y2=(p2.x,p2.y) if hasattr(p2,"x") else p2
    return math.hypot(x1-x2,y1-y2)

def manhattan_heuristic(p1,p2)->float:
    x1,y1=(p1.x,p1.y) if hasattr(p1,"x") else p1
    x2,y2=(p2.x,p2.y) if hasattr(p2,"x") else p2
    return abs(x1-x2)+abs(y1-y2)

def chebyshev_heuristic(p1,p2)->float:
    x1,y1=(p1.x,p1.y) if hasattr(p1,"x") else p1
    x2,y2=(p2.x,p2.y) if hasattr(p2,"x") else p2
    return max(abs(x1-x2),abs(y1-y2))

def weighted_euclidean_heuristic(p1,p2,weight:float=1.0)->float:
    return weight * euclidean_heuristic(p1,p2)

# --------------------------------------------------------------------------- #
# CORE A*
# --------------------------------------------------------------------------- #
def neighbors(n:Node, diag:bool)->Iterable[Node]:
    """Yield orthogonal neighbours; include diagonals when diag is True."""
    dirs4=[(1,0),(-1,0),(0,1),(0,-1)]
    dirs8=dirs4+[(1,1),(1,-1),(-1,1),(-1,-1)]
    for dx,dy in (dirs8 if diag else dirs4):
        nx,ny=n.x+dx,n.y+dy
        if in_bounds(nx,ny):
            yield Node(nx,ny)

def step_cost(a:Node,b:Node)->float:
    """Chebyshev metric cost: diagonal moves cost sqrt(2)."""
    return math.sqrt(2) if (a.x!=b.x and a.y!=b.y) else 1.0

def corner_cut(grid, a:Node, b:Node)->bool:
    """Block diagonal moves if they would sneak through the corner of a wall."""
    if a.x==b.x or a.y==b.y: return False
    return grid[a.x][b.y] or grid[b.x][a.y]

def astar(grid,start:Node,goal:Node,diag:bool,heuristic_fn:Callable[[Node,Node],float]):
    openh:List[Tuple[float,int,Node]]=[]
    g:Dict[Node,float]={start:0.0}
    came:Dict[Node,Optional[Node]]={start:None}
    seen:Set[Node]={start}
    closed:Set[Node]=set()
    t=0
    heapq.heappush(openh,(heuristic_fn(start,goal),t,start)); t+=1

    while openh:
        _,_,cur=heapq.heappop(openh)
        if cur in closed: continue
        closed.add(cur)
        # Expose current search frontier/closed sets for the visualiser.
        yield "state",cur,set(seen),set(closed),dict(came)
        if cur==goal:
            yield "done",cur,set(seen),set(closed),dict(came)
            return
        for nb in neighbors(cur,diag):
            if grid[nb.x][nb.y]: continue
            if diag and corner_cut(grid,cur,nb): continue
            ng=g[cur]+step_cost(cur,nb)
            if nb not in g or ng<g[nb]-1e-9:
                g[nb]=ng
                came[nb]=cur
                heapq.heappush(openh,(ng+heuristic_fn(nb,goal),t,nb)); t+=1
                seen.add(nb)
    yield "done",None,set(),set(closed),dict(came)

def reconstruct(came:Dict[Node,Optional[Node]], end:Node)->List[Node]:
    """Walk backwards from the goal to produce the displayed path."""
    if end not in came: return []
    p=[end]; cur=end
    while came[cur] is not None:
        cur=came[cur]; p.append(cur)
    p.reverse(); return p

# --------------------------------------------------------------------------- #
# SCENARIO BUILDERS (reuse + for Task 2 benchmarks)
# --------------------------------------------------------------------------- #
def empty_grid()->list:
    return [[0 for _ in range(ROWS)] for _ in range(COLS)]

def simple_barrier()->list:
    """Single vertical wall in the middle with a gap so a path exists."""
    grid=empty_grid()
    midx=COLS//2; gap_y=ROWS//2
    for y in range(ROWS):
        if y==gap_y: continue
        grid[midx][y]=1
    return grid

def scattered_obstacles(density=0.20, start:Node=None, goal:Node=None)->list:
    """Fill the grid with random blockers and reroll until path exists."""
    rnd=random.Random(42)
    while True:
        grid=empty_grid()
        for x in range(COLS):
            for y in range(ROWS):
                if rnd.random() < density:
                    grid[x][y]=1
        if start: grid[start.x][start.y]=0
        if goal:  grid[goal.x][goal.y]=0
        ok,_=dijkstra_length(grid,start,goal,diag=True)
        if ok: return grid

def maze_dfs(start:Node, goal:Node)->list:
    """Carve a DFS maze on odd cells and make sure start/goal are open."""
    grid=[[1 for _ in range(ROWS)] for _ in range(COLS)]
    def cells_neighbors(cx,cy):
        for dx,dy in ((2,0),(-2,0),(0,2),(0,-2)):
            nx,ny=cx+dx,cy+dy
            if 1<=nx<COLS-1 and 1<=ny<ROWS-1:
                yield nx,ny
    sx,sy=1 if COLS>2 else 0, 1 if ROWS>2 else 0
    stack=[(sx,sy)]
    grid[sx][sy]=0
    visited={(sx,sy)}
    rnd=random.Random(42)
    while stack:
        cx,cy=stack[-1]
        nbrs=[(nx,ny) for nx,ny in cells_neighbors(cx,cy) if (nx,ny) not in visited]
        if nbrs:
            nx,ny=rnd.choice(nbrs)
            wx,wy=cx+(nx-cx)//2,cy+(ny-cy)//2
            grid[wx][wy]=0; grid[nx][ny]=0
            visited.add((nx,ny)); stack.append((nx,ny))
        else:
            stack.pop()
    grid[start.x][start.y]=0; grid[goal.x][goal.y]=0
    ok,_=dijkstra_length(grid,start,goal,diag=True)
    if not ok:
        x,y=start.x,start.y
        while x!=goal.x: x+=1 if goal.x>x else -1; grid[x][y]=0
        while y!=goal.y: y+=1 if goal.y>y else -1; grid[x][y]=0
    return grid

# --------------------------------------------------------------------------- #
# OBSTACLE PATTERNS (Task 3)
# --------------------------------------------------------------------------- #
def linear_barriers(kind:str)->list:
    """Deterministic line-based obstacles to illustrate detours."""
    g=empty_grid()
    if kind=="horizontal":
        y=ROWS//2
        for x in range(COLS):
            if x not in (2,COLS-3): g[x][y]=1
    elif kind=="vertical":
        x=COLS//2
        for y in range(ROWS):
            if y not in (2,ROWS-3): g[x][y]=1
    elif kind=="diagonal":
        for i in range(min(COLS,ROWS)): g[i][i]=1
    elif kind=="L":
        for x in range(5,COLS//2): g[x][5]=1
        for y in range(5,ROWS//2): g[5][y]=1
    return g

def geometric_shape(kind:str)->list:
    """Render geometric blocker patterns for heuristic stress tests."""
    g=empty_grid(); cx,cy=COLS//2,ROWS//2
    if kind=="square":
        for x in range(max(0,cx-8),min(COLS,cx+8)):
            if 0<=cy-6<ROWS: g[x][cy-6]=1
            if 0<=cy+6<ROWS: g[x][cy+6]=1
        for y in range(max(0,cy-6),min(ROWS,cy+6)):
            if 0<=cx-8<COLS: g[cx-8][y]=1
            if 0<=cx+8<COLS: g[cx+8][y]=1
    elif kind=="circle":
        r=8
        for x in range(COLS):
            for y in range(ROWS):
                if (x-cx)**2+(y-cy)**2<=r*r: g[x][y]=1
    elif kind=="star":
        r=8
        for i in range(-r,r+1):
            if in_bounds(cx+i,cy): g[cx+i][cy]=1
            if in_bounds(cx,cy+i): g[cx][cy+i]=1
            if in_bounds(cx+i,cy+i): g[cx+i][cy+i]=1
            if in_bounds(cx+i,cy-i): g[cx+i][cy-i]=1
    elif kind=="spiral":
        r=10
        left, right, top, bottom = cx-r, cx+r, cy-r, cy+r
        while left<=right and top<=bottom:
            for x in range(max(0,left),min(COLS,right+1)):
                if 0<=top<ROWS: g[x][top]=1
            for y in range(max(0,top),min(ROWS,bottom+1)):
                if 0<=right<COLS: g[right][y]=1
            left+=2; top+=2; right-=2; bottom-=2
    return g

def maze_types(kind:str)->list:
    """Family of deterministic mazes with varying corridor structures."""
    g=empty_grid()
    if kind=="simple":
        y=ROWS//2
        for x in range(2,COLS-2):
            if x%4!=0: g[x][y]=1
    elif kind=="rooms":
        w,h=COLS//4,ROWS//4
        for i in range(1,4):
            for j in range(1,3):
                x0,y0=i*w - w//2, j*h
                for xx in range(max(1,x0-w//3), min(COLS-1,x0+w//3)):
                    for yy in range(max(1,y0-h//3), min(ROWS-1,y0+h//3)):
                        g[xx][yy]=1
        for x in range(0,COLS,3): g[x][ROWS//2]=0
    elif kind=="open":
        for x in range(1,COLS-1,2):
            for y in range(1,ROWS-1,3):
                g[x][y]=1
    return g

PATTERNS = {
    "off": [None],
    "linear": ["horizontal","vertical","diagonal","L"],
    "geometric": ["square","circle","star","spiral"],
    "maze": ["simple","rooms","open"],
}

# --------------------------------------------------------------------------- #
# MAIN VISUALIZATION
# --------------------------------------------------------------------------- #
def main():
    pygame.init()
    pygame.display.set_caption("A* Multi-Heuristic + Patterns + Benchmarks")
    screen=pygame.display.set_mode((W,H))
    clock=pygame.time.Clock()
    font=pygame.font.SysFont("Menlo,Consolas,DejaVuSansMono,monospace",18)
    small=pygame.font.SysFont("Menlo,Consolas,DejaVuSansMono,monospace",14)

    random.seed(42)  # deterministic patterns when random noise is used

    # Initial state
    start, goal = Node(2,2), Node(COLS-3, ROWS-3)
    diag = True

    WEIGHTS = [0.5, 1.0, 1.5, 2.0]
    w_idx = 1
    mode = "euclidean"

    pattern_families = list(PATTERNS.keys())
    p_idx, v_idx = 0, 0  # start with 'off'
    def apply_pattern(family:str, variant:Optional[str])->list:
        if family=="off" or variant is None: return empty_grid()
        if family=="linear": return linear_barriers(variant)
        if family=="geometric": return geometric_shape(variant)
        if family=="maze": return maze_types(variant)
        return empty_grid()

    family = pattern_families[p_idx]
    variant = PATTERNS[family][v_idx]
    grid = apply_pattern(family, variant)

    def make_heur():
        """Return the currently selected heuristic callable."""
        if mode=="euclidean": return euclidean_heuristic
        if mode=="manhattan": return manhattan_heuristic
        if mode=="chebyshev": return chebyshev_heuristic
        if mode=="weighted": return lambda a,b: weighted_euclidean_heuristic(a,b,WEIGHTS[w_idx])
        return euclidean_heuristic

    running=False; gen=None; path:List[Node]=[]

    def reset():
        """Rebuild the A* generator after any settings or grid change."""
        nonlocal gen, running, path
        gen = astar(grid, start, goal, diag, make_heur())
        running=False; path=[]

    reset()

    # Event loop
    while True:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: return
            if e.type==pygame.KEYDOWN:
                if e.key==pygame.K_ESCAPE: return
                if e.key==pygame.K_SPACE: running=not running
                if e.key==pygame.K_r: reset()
                if e.key==pygame.K_c:
                    grid = empty_grid(); reset()
                if e.key==pygame.K_d:
                    diag = not diag; reset()

                # Heuristic hotkeys
                if e.key==pygame.K_1: mode="euclidean"; reset()
                if e.key==pygame.K_2: mode="manhattan"; reset()
                if e.key==pygame.K_3: mode="chebyshev"; reset()
                if e.key==pygame.K_4: mode="weighted"; reset()
                if e.key==pygame.K_LEFTBRACKET and mode=="weighted":
                    w_idx=max(0,w_idx-1); reset()
                if e.key==pygame.K_RIGHTBRACKET and mode=="weighted":
                    w_idx=min(len(WEIGHTS)-1,w_idx+1); reset()

                # Benchmarks
                if e.key==pygame.K_b:
                    # Print benchmark table to the console without leaving the UI.
                    run_benchmarks()

                # Pattern cycling (Task 3)
                if e.key==pygame.K_p:
                    p_idx=(p_idx+1)%len(pattern_families)
                    v_idx=0
                    family=pattern_families[p_idx]
                    variant=PATTERNS[family][v_idx]
                    grid=apply_pattern(family,variant)
                    reset()
                if e.key==pygame.K_COMMA:   # previous variant
                    v_idx=(v_idx-1)%len(PATTERNS[family])
                    variant=PATTERNS[family][v_idx]
                    grid=apply_pattern(family,variant)
                    reset()
                if e.key==pygame.K_PERIOD:  # next variant
                    v_idx=(v_idx+1)%len(PATTERNS[family])
                    variant=PATTERNS[family][v_idx]
                    grid=apply_pattern(family,variant)
                    reset()

            # Mouse drawing / start-goal placement
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

        # Step A* if running
        if running and gen:
            try:
                tag, cur, open_set, closed, came = next(gen)
                if tag=="done":
                    running=False
                    if goal in came:
                        path = reconstruct(came, goal)
                else:
                    heur = make_heur()
                    if open_set:
                        best = min(open_set, key=lambda n: heur(n,goal))
                        path = reconstruct(came, best)
            except StopIteration:
                running=False

        # DRAW
        screen.fill(BG)
        pygame.draw.rect(screen, GRID_BG, (OX-6, OY-6, GRID_W+12, GRID_H+12), border_radius=12)
        for x in range(COLS+1):
            X=OX+x*CELL
            pygame.draw.line(screen, GRID_LINE, (X, OY), (X, OY+GRID_H))
        for y in range(ROWS+1):
            Y=OY+y*CELL
            pygame.draw.line(screen, GRID_LINE, (OX, Y), (OX+GRID_W, Y))

        # walls
        for x in range(COLS):
            for y in range(ROWS):
                if grid[x][y]:
                    pygame.draw.rect(screen, WALL, (OX+x*CELL+1, OY+y*CELL+1, CELL-2, CELL-2), border_radius=ROUND)

        # path overlay
        overlay=pygame.Surface((GRID_W,GRID_H), pygame.SRCALPHA)
        for i,n in enumerate(path):
            alpha = 120 if i==len(path)-1 else 90
            r = rect_of(n)
            pygame.draw.rect(overlay, (*PATH, alpha), (r.x-OX, r.y-OY, r.w, r.h), border_radius=ROUND)
        screen.blit(overlay, (OX,OY))

        # start/goal
        pygame.draw.rect(screen, START, rect_of(start), border_radius=ROUND)
        pygame.draw.rect(screen, GOAL, rect_of(goal), border_radius=ROUND)

        # HUD
        hudy = OY+GRID_H+14
        mode_label = {
            "euclidean": "Euclidean",
            "manhattan": "Manhattan",
            "chebyshev": "Chebyshev",
            "weighted": f"Weighted (w={WEIGHTS[w_idx]:.1f})",
        }[mode]
        screen.blit(small.render(f"Heuristic: {mode_label}", True, MUTED), (OX, hudy))
        screen.blit(small.render(f"Pattern: {family} / {variant}", True, MUTED), (OX, hudy+18))
        screen.blit(small.render("P: cycle families, ,/. variants  |  B: bench  |  D: diagonals  |  R/C: reset/clear", True, MUTED), (OX, hudy+36))
        screen.blit(font.render("A* Pathfinding", True, TEXT), (OX, 6))

        pygame.display.flip()
        clock.tick(60)

# --------------------------------------------------------------------------- #
# BENCHMARKS (Task 2)
# --------------------------------------------------------------------------- #
def dijkstra_length(grid, start:Node, goal:Node, diag:bool)->Tuple[bool,float]:
    """Run Dijkstra to obtain optimal path length for benchmark comparisons."""
    openh:List[Tuple[float,int,Node]]=[]; g:Dict[Node,float]={start:0.0}; t=0
    heapq.heappush(openh,(0.0,t,start)); closed:Set[Node]=set()
    while openh:
        gc,_,cur=heapq.heappop(openh)
        if cur in closed: continue
        closed.add(cur)
        if cur==goal: return True,gc
        for nb in neighbors(cur,diag):
            if grid[nb.x][nb.y]: continue
            if diag and corner_cut(grid,cur,nb): continue
            ng=gc+step_cost(cur,nb)
            if nb not in g or ng<g[nb]-1e-9:
                g[nb]=ng; t+=1; heapq.heappush(openh,(ng,t,nb))
    return False,float("inf")

def astar_run(grid,start,goal,diag,heur_fn):
    """Execute A* once and return aggregate metrics for the benchmark table."""
    openh=[]; g={start:0.0}; came={start:None}; t=0
    heapq.heappush(openh,(heur_fn(start,goal),t,start)); closed=set()
    t0=time.perf_counter()
    while openh:
        _,_,cur=heapq.heappop(openh)
        if cur in closed: continue
        closed.add(cur)
        if cur==goal:
            path=reconstruct(came,goal)
            length=sum(step_cost(a,b) for a,b in zip(path,path[1:]))
            return {"found":True,"path_len":length,"time_ms":(time.perf_counter()-t0)*1000.0,"explored":len(closed)}
        for nb in neighbors(cur,diag):
            if grid[nb.x][nb.y]: continue
            if diag and corner_cut(grid,cur,nb): continue
            ng=g[cur]+step_cost(cur,nb)
            if nb not in g or ng<g[nb]-1e-9:
                g[nb]=ng; came[nb]=cur; t+=1; heapq.heappush(openh,(ng+heur_fn(nb,goal),t,nb))
    return {"found":False,"path_len":float("inf"),"time_ms":(time.perf_counter()-t0)*1000.0,"explored":len(closed)}

def run_benchmarks():
    """Compare heuristics across canned scenarios and print a CSV-like table."""
    random.seed(42)
    start = Node(5,5)
    goal  = Node(min(45, COLS-1), min(25, ROWS-1))
    goal  = Node(max(0,goal.x), max(0,goal.y))

    scenarios = [
        ("Empty Grid",              empty_grid()),
        ("Simple Barrier",          simple_barrier()),
        ("Maze DFS",                maze_dfs(start, goal)),
        ("Scattered Obstacles",     scattered_obstacles(0.20, start, goal)),

        # Task 3 — Linear
        ("Linear Horizontal",       linear_barriers("horizontal")),
        ("Linear Vertical",         linear_barriers("vertical")),
        ("Linear Diagonal",         linear_barriers("diagonal")),
        ("Linear L-Shape",          linear_barriers("L")),

        # Task 3 — Geometric
        ("Geo Hollow Square",       geometric_shape("square")),
        ("Geo Filled Circle",       geometric_shape("circle")),
        ("Geo Star",                geometric_shape("star")),
        ("Geo Spiral",              geometric_shape("spiral")),

        # Task 3 — Maze types
        ("Maze Simple",             maze_types("simple")),
        ("Maze Rooms",              maze_types("rooms")),
        ("Maze Open",               maze_types("open")),
    ]

    WEIGHTS=[0.5,1.0,1.5,2.0]
    heuristics = [
        ("Euclidean",       lambda a,b: euclidean_heuristic(a,b)),
        ("Manhattan",       lambda a,b: manhattan_heuristic(a,b)),
        ("Chebyshev",       lambda a,b: chebyshev_heuristic(a,b)),
    ] + [(f"Weighted w={w}", (lambda ww: (lambda a,b: weighted_euclidean_heuristic(a,b,ww)))(w)) for w in WEIGHTS]

    print("\n=== A* Benchmark (COLS={}, ROWS={}, diag=True) ===".format(COLS, ROWS))
    print("Start={}, Goal={}\n".format((start.x,start.y),(goal.x,goal.y)))
    print("{:<22} | {:<18} | {:>9} | {:>9} | {:>8} | {:>8}".format("Scenario","Heuristic","Length","Time(ms)","Optimal","Explrd"))
    print("-"*90)

    for sname,sgrid in scenarios:
        ok_opt, opt_len = dijkstra_length(sgrid, start, goal, diag=True)
        if not ok_opt:
            print(f"[WARN] No path in scenario '{sname}', skipping.")
            continue
        for hname,hfn in heuristics:
            res = astar_run(sgrid, start, goal, diag=True, heur_fn=hfn)
            is_opt = (abs(res["path_len"] - opt_len) < 1e-6) if res["found"] else False
            length_str = f"{res['path_len']:.3f}" if res["found"] else "—"
            print("{:<22} | {:<18} | {:>9} | {:>9.2f} | {:>8} | {:>8}".format(
                sname, hname, length_str, res["time_ms"], "YES" if is_opt else "NO", res["explored"]))
    print("\nDone.\n")

# --------------------------------------------------------------------------- #
# CLI SWITCH
# --------------------------------------------------------------------------- #
def _maybe_run_bench_from_argv()->bool:
    """Catch the --bench flag so the script can run headless from the CLI."""
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bench", action="store_true", default=False)
    try:
        args,_=parser.parse_known_args()
        if args.bench:
            run_benchmarks()
            return True
    except SystemExit:
        return False
    return False

if __name__=="__main__":
    did=_maybe_run_bench_from_argv()
    if not did:
        main()
