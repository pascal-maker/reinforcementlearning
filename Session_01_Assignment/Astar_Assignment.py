import pygame
import math
import heapq
from typing import List, Tuple, Optional, Set
import sys

# Initialize Pygame
pygame.init()

# Constants
GRID_WIDTH = 50
GRID_HEIGHT = 30
CELL_SIZE = 20
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + 100  # Extra space for controls

# Colors
WHITE = (255, 255, 255)
BLACK = (50, 50, 50)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)

class Node:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.g_cost = 0  # Distance from start
        self.h_cost = 0  # Heuristic distance to end
        self.f_cost = 0  # Total cost (g + h)
        self.parent: Optional['Node'] = None
        
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

class AStarDemo:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("A* Pathfinding Demo - Click to draw obstacles, R to reset, Space to find path")
        self.clock = pygame.time.Clock()
        
        # Grid state
        self.obstacles: Set[Tuple[int, int]] = set()
        self.start_pos: Optional[Tuple[int, int]] = (2, 2)
        self.end_pos: Optional[Tuple[int, int]] = (GRID_WIDTH-3, GRID_HEIGHT-3)
        self.path: List[Tuple[int, int]] = []
        self.open_set: Set[Tuple[int, int]] = set()
        self.closed_set: Set[Tuple[int, int]] = set()
        
        # UI state
        self.mode = "obstacle"  # obstacle, start, end
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Animation
        self.animate_search = False
        self.search_step = 0
        self.animation_speed = 10  # Lower = faster
        
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighbors for a given position"""
        neighbors = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < GRID_WIDTH and 
                0 <= ny < GRID_HEIGHT and 
                (nx, ny) not in self.obstacles):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance (Euclidean)"""
        return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))
        #return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from end node to start"""
        path = []
        current = node
        while current:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]
    
    def find_path(self) -> bool:
        """Find path using A* algorithm"""
        if not self.start_pos or not self.end_pos:
            return False
        
        self.path = []
        self.open_set = set()
        self.closed_set = set()
        
        start_node = Node(self.start_pos[0], self.start_pos[1])
        end_node = Node(self.end_pos[0], self.end_pos[1])
        
        open_heap = [start_node]
        open_dict = {(start_node.x, start_node.y): start_node}
        
        while open_heap:
            current_node = heapq.heappop(open_heap)
            current_pos = (current_node.x, current_node.y)
            
            self.open_set.discard(current_pos)
            self.closed_set.add(current_pos)
            
            if current_node == end_node:
                self.path = self.reconstruct_path(current_node)
                return True
            
            for neighbor_pos in self.get_neighbors(current_node.x, current_node.y):
                if neighbor_pos in self.closed_set:
                    continue
                
                neighbor_x, neighbor_y = neighbor_pos
                
                # Calculate movement cost (diagonal = ~1.414, straight = 1)
                dx = abs(neighbor_x - current_node.x)
                dy = abs(neighbor_y - current_node.y)
                movement_cost = 1.414 if dx == 1 and dy == 1 else 1
                
                tentative_g = current_node.g_cost + movement_cost
                
                if neighbor_pos in open_dict:
                    neighbor_node = open_dict[neighbor_pos]
                    if tentative_g < neighbor_node.g_cost:
                        neighbor_node.parent = current_node
                        neighbor_node.g_cost = tentative_g
                        neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                        heapq.heapify(open_heap)  # Re-heapify after cost change
                else:
                    neighbor_node = Node(neighbor_x, neighbor_y)
                    neighbor_node.parent = current_node
                    neighbor_node.g_cost = tentative_g
                    neighbor_node.h_cost = self.heuristic(neighbor_pos, self.end_pos)
                    neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                    
                    heapq.heappush(open_heap, neighbor_node)
                    open_dict[neighbor_pos] = neighbor_node
                    self.open_set.add(neighbor_pos)
        
        return False
    
    def grid_to_screen(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        """Convert grid coordinates to screen coordinates"""
        return grid_x * CELL_SIZE, grid_y * CELL_SIZE
    
    def screen_to_grid(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """Convert screen coordinates to grid coordinates"""
        return screen_x // CELL_SIZE, screen_y // CELL_SIZE
    
    def draw_grid(self):
        """Draw the grid background"""
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, LIGHT_GRAY, (x, 0), (x, GRID_HEIGHT * CELL_SIZE))
        for y in range(0, GRID_HEIGHT * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, LIGHT_GRAY, (0, y), (WINDOW_WIDTH, y))
    
    def draw_cells(self):
        """Draw all cells with their appropriate colors"""
        # Draw closed set (explored nodes)
        for x, y in self.closed_set:
            screen_x, screen_y = self.grid_to_screen(x, y)
            pygame.draw.rect(self.screen, CYAN, 
                           (screen_x, screen_y, CELL_SIZE, CELL_SIZE))
        
        # Draw open set (frontier nodes)
        for x, y in self.open_set:
            screen_x, screen_y = self.grid_to_screen(x, y)
            pygame.draw.rect(self.screen, YELLOW, 
                           (screen_x, screen_y, CELL_SIZE, CELL_SIZE))
        
        # Draw obstacles
        for x, y in self.obstacles:
            screen_x, screen_y = self.grid_to_screen(x, y)
            pygame.draw.rect(self.screen, BLACK, 
                           (screen_x, screen_y, CELL_SIZE, CELL_SIZE))
        
        # Draw path
        for i, (x, y) in enumerate(self.path):
            screen_x, screen_y = self.grid_to_screen(x, y)
            if i == 0:  # Start
                continue
            elif i == len(self.path) - 1:  # End
                continue
            else:  # Path
                pygame.draw.rect(self.screen, PURPLE, 
                               (screen_x + 2, screen_y + 2, 
                                CELL_SIZE - 4, CELL_SIZE - 4))
        
        # Draw start position
        if self.start_pos:
            screen_x, screen_y = self.grid_to_screen(self.start_pos[0], self.start_pos[1])
            pygame.draw.rect(self.screen, GREEN, 
                           (screen_x, screen_y, CELL_SIZE, CELL_SIZE))
            
        # Draw end position
        if self.end_pos:
            screen_x, screen_y = self.grid_to_screen(self.end_pos[0], self.end_pos[1])
            pygame.draw.rect(self.screen, RED, 
                           (screen_x, screen_y, CELL_SIZE, CELL_SIZE))
    
    def draw_ui(self):
        """Draw user interface elements"""
        y_offset = GRID_HEIGHT * CELL_SIZE + 10
        
        # Mode indicator
        mode_text = f"Mode: {self.mode.title()}"
        mode_color = GREEN if self.mode == "start" else RED if self.mode == "end" else BLACK
        text_surface = self.font.render(mode_text, True, mode_color)
        self.screen.blit(text_surface, (10, y_offset))
        
        # Instructions
        instructions = [
            "Left Click: Place/Remove | Right Click: Change Mode | SPACE: Find Path | R: Reset | ESC: Quit",
            "Green: Start | Red: End | Black: Obstacles | Purple: Path | Yellow: Open | Cyan: Explored"
        ]
        
        for i, instruction in enumerate(instructions):
            text_surface = self.small_font.render(instruction, True, GRAY)
            self.screen.blit(text_surface, (10, y_offset + 30 + i * 20))
        
        # Path length
        if self.path:
            path_text = f"Path Length: {len(self.path) - 1} steps"
            text_surface = self.font.render(path_text, True, PURPLE)
            self.screen.blit(text_surface, (WINDOW_WIDTH - 200, y_offset))
    
    def handle_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse clicks"""
        grid_x, grid_y = self.screen_to_grid(pos[0], pos[1])
        
        # Check if click is within grid
        if not (0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT):
            return
        
        grid_pos = (grid_x, grid_y)
        
        if button == 1:  # Left click
            if self.mode == "obstacle":
                if grid_pos in self.obstacles:
                    self.obstacles.remove(grid_pos)
                else:
                    # Don't place obstacle on start or end
                    if grid_pos != self.start_pos and grid_pos != self.end_pos:
                        self.obstacles.add(grid_pos)
                        
            elif self.mode == "start":
                if grid_pos not in self.obstacles:
                    self.start_pos = grid_pos
                    
            elif self.mode == "end":
                if grid_pos not in self.obstacles:
                    self.end_pos = grid_pos
            
            # Clear previous search results
            self.path = []
            self.open_set = set()
            self.closed_set = set()
            
        elif button == 3:  # Right click - cycle mode
            modes = ["obstacle", "start", "end"]
            current_index = modes.index(self.mode)
            self.mode = modes[(current_index + 1) % len(modes)]
    
    def reset(self):
        """Reset the grid"""
        self.obstacles.clear()
        self.path = []
        self.open_set = set()
        self.closed_set = set()
        self.start_pos = (2, 2)
        self.end_pos = (GRID_WIDTH-3, GRID_HEIGHT-3)
    
    def run(self):
        """Main game loop"""
        running = True
        mouse_pressed = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        success = self.find_path()
                        if not success:
                            print("No path found!")
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_1:
                        self.mode = "obstacle"
                    elif event.key == pygame.K_2:
                        self.mode = "start"
                    elif event.key == pygame.K_3:
                        self.mode = "end"
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.pos[1] < GRID_HEIGHT * CELL_SIZE:  # Only within grid area
                        mouse_pressed = True
                        self.handle_click(event.pos, event.button)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_pressed = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if mouse_pressed and pygame.mouse.get_pressed()[0]:  # Left button held
                        if event.pos[1] < GRID_HEIGHT * CELL_SIZE:  # Only within grid area
                            self.handle_click(event.pos, 1)
            
            # Clear screen
            self.screen.fill(WHITE)
            
            # Draw everything
            self.draw_grid()
            self.draw_cells()
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    demo = AStarDemo()
    demo.run()