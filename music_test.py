
import pygame
import time

pygame.mixer.init()

pygame.mixer.music.load("music/song1.mp3")
pygame.mixer.music.play()

print("Music is playing!")

while pygame.mixer.music.get_busy():
    time.sleep(0.5)

pygame.mixer.quit()
