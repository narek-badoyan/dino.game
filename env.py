import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random



# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
DINO_WIDTH, DINO_HEIGHT = 40, 40
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 20, 40
GROUND_HEIGHT = 300
FONT_SIZE = 24
WHITE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (0    , 0, 0)
FPS = 60

class Obstacle:
    def __init__(self, x, y, width, height, speed):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.speed = speed

    def move(self):
        self.x -= self.speed
        if self.x + self.width < 0:  # If the obstacle goes off-screen
            self.x = SCREEN_WIDTH + random.randint(50, 300)  # Reset position randomly
            self.speed = random.randint(5, 10)  # Randomize speed

    def render(self, screen):
        pygame.draw.rect(screen, RED, (self.x, self.y, self.width, self.height))


class FlyingObstacle(Obstacle):
    def __init__(self, x, y, width, height, speed):
        super().__init__(x, y, width, height, speed)
        self.vertical_speed = random.randint(-5, 5)  # Random vertical speed for flying obstacles

    def move(self):
        self.x -= self.speed
        self.y += self.vertical_speed

        # Keep flying obstacles within the screen boundaries
        if self.y <= 0:
            self.y = 0
            self.vertical_speed = abs(self.vertical_speed)  # Prevent going above the screen
        elif self.y + self.height >= GROUND_HEIGHT:
            self.y = GROUND_HEIGHT - self.height
            self.vertical_speed = -abs(self.vertical_speed)  # Prevent going below the ground

        if self.x + self.width < 0:  # If the obstacle goes off-screen
            self.x = SCREEN_WIDTH + random.randint(50, 300)  # Reset position randomly
            self.speed = random.randint(5, 10)  # Randomize speed
            self.vertical_speed = random.randint(-5, 5)  # Reset vertical speed


class DinoGame(gym.Env):
    def __init__(self):
        super(DinoGame, self).__init__()
        self.state = None
        self.reward = 0
        self.score = 0  # Continuous score
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Google Dino Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, FONT_SIZE)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.dino_y = GROUND_HEIGHT - DINO_HEIGHT
        self.dino_velocity = 0
        self.is_jumping = False
        self.score = 0  # Reset the score

        # Create both ground and flying obstacles
        self.obstacles = [
            Obstacle(SCREEN_WIDTH, GROUND_HEIGHT - OBSTACLE_HEIGHT, OBSTACLE_WIDTH, OBSTACLE_HEIGHT, random.randint(5, 10)),
            FlyingObstacle(SCREEN_WIDTH + 300, random.randint(50, 200), OBSTACLE_WIDTH, OBSTACLE_HEIGHT, random.randint(6, 12)),
            Obstacle(SCREEN_WIDTH + 600, GROUND_HEIGHT - OBSTACLE_HEIGHT, OBSTACLE_WIDTH, OBSTACLE_HEIGHT, random.randint(7, 14))
        ]

        self.state = np.array([self.dino_y, self.obstacles[0].x, self.obstacles[0].speed, self.score], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        if action == 1 and not self.is_jumping:
            self.is_jumping = True
            self.dino_velocity = -15

        if self.is_jumping:
            self.dino_y += self.dino_velocity
            self.dino_velocity += 1
            if self.dino_y >= GROUND_HEIGHT - DINO_HEIGHT:
                self.dino_y = GROUND_HEIGHT - DINO_HEIGHT
                self.is_jumping = False

        for obstacle in self.obstacles:
            obstacle.move()

        self.score += 0.1

        done = False
        for obstacle in self.obstacles:
            if (
                obstacle.x < 50 + DINO_WIDTH and
                obstacle.x + obstacle.width > 50 and
                self.dino_y + DINO_HEIGHT > obstacle.y and
                self.dino_y < obstacle.y + obstacle.height
            ):
                done = True

        self.state = np.array([self.dino_y, self.obstacles[0].x, self.obstacles[0].speed, self.score], dtype=np.float32)
        reward = 1 if not done else -100
        return self.state, reward, done, False, {}

    def render(self, mode="human"):
        self.screen.fill(WHITE)
        pygame.draw.line(self.screen, BLACK, (0, GROUND_HEIGHT), (SCREEN_WIDTH, GROUND_HEIGHT), 2)
        pygame.draw.rect(self.screen, BLACK, (50, self.dino_y, DINO_WIDTH, DINO_HEIGHT))

        # Render all obstacles
        for obstacle in self.obstacles:
            obstacle.render(self.screen)

        # Render score
        score_text = self.font.render(f"Score: {int(self.score)}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(FPS)
        return self.screen

    def close(self):
        pygame.quit()





