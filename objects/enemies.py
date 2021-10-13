from objects.entity import AnimateEntity, Explosion
from pygame import Rect, image 
from pygame.transform import scale
from objects.game import Game
import random

from objects.stopwatch import Stopwatch

class Enemies(AnimateEntity):
    def __init__(self, x, y, speed):
        self.timer = Stopwatch()
        super().__init__(x, y, speed)
        if not Game.players:
            return
        self.target_player = random.choice(Game.players)
        self.speed = 1.2

    def move(self, to):
        if self.timer.time_elapsed() > 5_000:
            if not Game.players:
                return
            self.target_player = random.choice(Game.players)
            self.timer.reset()
        for i in self.hit_test(self.Rect, filter(lambda x: isinstance(x, Explosion), Game.entities)):
            try: # Sometimes the explosion accidentally delete an enemy twice, this is a simple fix for that
                Game.enemies.remove(self)
                i.player.kills += 1
                Game.genomes[Game.players.index(i.player)].fitness += 50
            except Exception as E:
                # print(E)
                pass
        [x, y] = to
        if self.x > x:
            self.x -= self.speed
        if self.x < x:
            self.x += self.speed
        if self.y > y:
            self.y -= self.speed
        if self.y < y:
            self.y += self.speed


class Harpies(Enemies):
    animations = [
        scale(
            image.load("assets/images/enemies/harpies/{}.png".format(x+1)),
            (Game.settings["game.tileSize"],
            Game.settings["game.tileSize"])
        ) for x in range(3)
    ]

    sprite = animations[0]
    def __init__(self, x, y, speed):
        super().__init__(x, y, speed)
        self.Rect = Rect(self.x, self.y, Harpies.sprite.get_width(), Harpies.sprite.get_height())



    def animate(self):
        self.Rect = Rect(self.x, self.y, Harpies.sprite.get_width(), Harpies.sprite.get_height())
        self.move((self.target_player.x, self.target_player.y))
        if self.frame < len(Harpies.animations) - 1:
            self.frame += 1
            Harpies.sprite = Harpies.animations[self.frame]
        else:
            self.frame = 0
        Game.surface.blit(Harpies.sprite, (self.x, self.y))

class Hunter(Enemies):
    sprite = scale(
        image.load("assets/images/enemies/hunter/1.png"), 
        (Game.settings["game.tileSize"], 
        Game.settings["game.tileSize"])
    )
    def __init__(self, x, y, speed):
        super().__init__(x, y, speed)
        self.Rect = Rect(self.x, self.y, Hunter.sprite.get_width(), Hunter.sprite.get_height())

    def animate(self):
        self.Rect = Rect(self.x, self.y, Hunter.sprite.get_width(), Hunter.sprite.get_height())
        self.move((self.target_player.x, self.target_player.y))
        Game.surface.blit(Hunter.sprite, (self.x, self.y))
