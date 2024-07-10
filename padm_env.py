
import sys
import pygame
import numpy as np
import gymnasium as gym

# Paths to your images
AGENT_IMG_PATH = r'C:\Users\admin\Desktop\test1_a3\Boy.png'
GOAL_IMG_PATH = r'C:\Users\admin\Desktop\test1_a3\Treasure.png'
OBSTACLE_IMG_PATH = r'C:\Users\admin\Desktop\test1_a3\Snake.png'
BLACK_HOLE_IMG_PATH = r'C:\Users\admin\Desktop\test1_a3\Cave.png'
BACKGROUND_IMG_PATH = r'C:\Users\admin\Desktop\test1_a3\Grass.jpg'

class ChildEnv(gym.Env):
    """
    Custom environment that follows the gym interface.
    This is a simple grid environment where the agent needs to reach a goal while avoiding obstacles.
    """
    def __init__(self, grid_size=6):
        super(ChildEnv, self).__init__()
        
        # Grid and cell size
        self.grid_size = grid_size
        self.cell_size = 100
        
        # Initial and goal states
        self.state = np.array([0, 0])
        self.goal = np.array([grid_size - 1, grid_size - 1])
        
        # Reward and other state variables
        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.info = {}
        self.hell_states = []
        self.black_hole_states = [np.array([0, 3]), np.array([2, 0]), np.array([5, 2])]
        
        # Load images
        self.agent_image = pygame.transform.scale(pygame.image.load(AGENT_IMG_PATH), (self.cell_size, self.cell_size))
        self.goal_image = pygame.transform.scale(pygame.image.load(GOAL_IMG_PATH), (self.cell_size, self.cell_size))
        self.hell_image = pygame.transform.scale(pygame.image.load(OBSTACLE_IMG_PATH), (self.cell_size, self.cell_size))
        self.black_hole_image = pygame.transform.scale(pygame.image.load(BLACK_HOLE_IMG_PATH), (self.cell_size, self.cell_size))
        self.background_image = pygame.transform.scale(pygame.image.load(BACKGROUND_IMG_PATH), (self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        
        # Action and observation space
        self.action_space = gym.spaces.Discrete(4)  # Four possible actions: up, down, right, left
        self.observation_space = gym.spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        pygame.display.set_caption('ChildEnv')
    
    def add_hell_states(self, hell_states):
        self.hell_states = hell_states

    def reset(self):
        self.state = np.array([0, 0])
        self.total_reward = 0
        self.done = False
        self.info = {}
        # Add hell states (obstacles)
        self.add_hell_states([np.array([2, 3]), np.array([3, 1]), np.array([4, 4])])
        return self.state, self.info
    
    def step(self, action):
        if action == 0 and self.state[0] > 0:  # Up
            self.state[0] -= 1
        elif action == 1 and self.state[0] < self.grid_size - 1:  # Down
            self.state[0] += 1
        elif action == 2 and self.state[1] < self.grid_size - 1:  # Right
            self.state[1] += 1
        elif action == 3 and self.state[1] > 0:  # Left
            self.state[1] -= 1
        
        self.reward = -1
        if np.array_equal(self.state, self.goal):
            self.reward = 10
            self.done = True
        
        for hell in self.hell_states:
            if np.array_equal(self.state, hell):
                self.reward = -10
                self.done = True
        
        for black_hole in self.black_hole_states:
            if np.array_equal(self.state, black_hole):
                self.reward = -5
        
        self.total_reward += self.reward
        return self.state, self.reward, self.done, self.info
    
    def render(self):
        self.screen.blit(self.background_image, (0, 0))
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
        
        for hell in self.hell_states:
            self.screen.blit(self.hell_image, (hell[1] * self.cell_size, hell[0] * self.cell_size))
        self.screen.blit(self.goal_image, (self.goal[1] * self.cell_size, self.goal[0] * self.cell_size))
        self.screen.blit(self.agent_image, (self.state[1] * self.cell_size, self.state[0] * self.cell_size))
        for black_hole in self.black_hole_states:
            self.screen.blit(self.black_hole_image, (black_hole[1] * self.cell_size, black_hole[0] * self.cell_size))
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()
