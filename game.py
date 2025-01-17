from env import DinoGame
import pygame

if __name__ == "__main__":
    env = DinoGame()
    done = False
    obs, _ = env.reset()

    print("Press SPACE to jump. Close the game window to exit.")

    while not done:
        env.render()
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                action = 1

        obs, reward, done, truncated, info = env.step(action)

    env.close()