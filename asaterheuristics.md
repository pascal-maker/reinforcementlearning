🎮 How to Use the A Pathfinding Visualization*
Getting Started
The window should now be open with:

Green square = Start position (2,2)
Red square = Goal position (42,32)
Empty grid ready for you to draw obstacles
Basic Controls
Running the Algorithm:

Space - Start/pause the A* algorithm
R - Reset the algorithm (keep current grid)
C - Clear the entire grid (remove all walls)
Drawing Obstacles:

Left mouse drag - Draw walls (black squares)
Right mouse drag - Erase walls
S + click - Set new Start position (green)
G + click - Set new Goal position (red)
Movement Options:

D - Toggle diagonal movement on/off
Testing Different Heuristics
Switch Heuristics:

1 - Euclidean (straight-line distance) ✨ Best for diagonal movement
2 - Manhattan (grid distance) ✨ Best for 4-way movement
3 - Chebyshev (max coordinate difference)
4 - Weighted Euclidean
Adjust Weight (when using 4):

[ - Decrease weight (more optimal, slower)
] - Increase weight (faster, less optimal)
Testing Obstacle Patterns
Cycle Pattern Families:

P - Cycle through: off → linear → geometric → maze → off
Change Pattern Variants:

, (comma) - Previous variant
. (period) - Next variant
Pattern Types:

Linear: horizontal, vertical, diagonal, L-shape
Geometric: square, circle, star, spiral
Maze: simple corridors, rooms, open maze
Benchmarking:
B - Run full benchmark comparison (prints to terminal)
🧪 Suggested Testing Workflow
Test Empty Grid:
Press C to clear
Press Space to watch A* find the direct path
Try different heuristics (1, 2, 3, 4)
Test with Obstacles:
Draw some walls with left mouse
Press R then Space to see pathfinding
Notice how different heuristics explore differently
Test Preset Patterns:
Press P to cycle patterns
Press . to see variants
Watch how each heuristic handles different obstacle types
Compare Performance:
Press B to see detailed benchmark table in terminal
To exit: Press ESC or close the window

Have fun exploring! The visualization really helps understand how different heuristics work. 🚀