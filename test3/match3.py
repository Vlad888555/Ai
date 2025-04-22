import pygame
import random
import sys


# Constants
GRID_SIZE = 10
CELL_SIZE = 40
PADDING = 5
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE + 50  # additional space for score
FPS = 60
NUM_COLORS = 6

# Colors
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
]
BACKGROUND_COLOR = (30, 30, 30)
GRID_BG_COLOR = (50, 50, 50)
TEXT_COLOR = (255, 255, 255)

class Match3Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Match-3 10x10")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.score = 0
        self.grid = [[random.randrange(NUM_COLORS) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.selected = None  # (row, col)
        self.animating = False
        self.fill_initial_matches()

    def fill_initial_matches(self):
        # Remove any initial matches
        changed = True
        while changed:
            changed = False
            matches = self.find_matches()
            if matches:
                changed = True
                for (r, c) in matches:
                    self.grid[r][c] = random.randrange(NUM_COLORS)

    def find_matches(self):
        matches = set()
        # horizontal
        for r in range(GRID_SIZE):
            count = 1
            for c in range(1, GRID_SIZE):
                if self.grid[r][c] == self.grid[r][c-1]:
                    count += 1
                else:
                    if count >= 3:
                        for k in range(c-count, c):
                            matches.add((r, k))
                    count = 1
            if count >= 3:
                for k in range(GRID_SIZE-count, GRID_SIZE):
                    matches.add((r, k))
        # vertical
        for c in range(GRID_SIZE):
            count = 1
            for r in range(1, GRID_SIZE):
                if self.grid[r][c] == self.grid[r-1][c]:
                    count += 1
                else:
                    if count >= 3:
                        for k in range(r-count, r):
                            matches.add((k, c))
                    count = 1
            if count >= 3:
                for k in range(GRID_SIZE-count, GRID_SIZE):
                    matches.add((k, c))
        return matches

    def remove_matches(self, matches):
        for (r, c) in matches:
            self.grid[r][c] = None
        self.score += len(matches) * 10

    def collapse_columns(self):
        for c in range(GRID_SIZE):
            col = [self.grid[r][c] for r in range(GRID_SIZE) if self.grid[r][c] is not None]
            missing = GRID_SIZE - len(col)
            new_col = [random.randrange(NUM_COLORS) for _ in range(missing)] + col
            for r in range(GRID_SIZE):
                self.grid[r][c] = new_col[r]

    def swap(self, a, b):
        (r1, c1), (r2, c2) = a, b
        self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]

    def valid_swap(self, a, b):
        self.swap(a, b)
        matches = self.find_matches()
        self.swap(a, b)
        return bool(matches)

    def handle_click(self, pos):
        x, y = pos
        if y > GRID_SIZE * CELL_SIZE:
            return
        col = x // CELL_SIZE
        row = y // CELL_SIZE
        if self.selected is None:
            self.selected = (row, col)
        else:
            if (abs(self.selected[0] - row) == 1 and self.selected[1] == col) or \
               (abs(self.selected[1] - col) == 1 and self.selected[0] == row):
                if self.valid_swap(self.selected, (row, col)):
                    self.swap(self.selected, (row, col))
                    self.resolve()
                else:
                    # invalid swap, just deselect
                    pass
            self.selected = None

    def resolve(self):
        # remove and collapse until no matches
        while True:
            matches = self.find_matches()
            if not matches:
                break
            self.remove_matches(matches)
            self.collapse_columns()

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        # draw grid background
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = pygame.Rect(c*CELL_SIZE+PADDING, r*CELL_SIZE+PADDING,
                                   CELL_SIZE-2*PADDING, CELL_SIZE-2*PADDING)
                pygame.draw.rect(self.screen, GRID_BG_COLOR, rect)
                color_idx = self.grid[r][c]
                if color_idx is not None:
                    inner = rect.inflate(-4, -4)
                    pygame.draw.rect(self.screen, COLORS[color_idx], inner)
        # highlight selected
        if self.selected:
            r, c = self.selected
            highlight = pygame.Rect(c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, (255,255,255), highlight, 3)
        # draw score
        score_surf = self.font.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.screen.blit(score_surf, (10, GRID_SIZE*CELL_SIZE + 10))
        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            self.draw()
            self.clock.tick(FPS)

if __name__ == "__main__":
    game = Match3Game()
    game.run()
    
