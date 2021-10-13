import math
from pygame.transform import scale
from objects.stopwatch import Stopwatch
from objects.game import Game
from objects.entity import AnimateEntity, BombActive, BombItem, Box, Explosion, InanimateEntity, Wall
from objects.text import Text
from pygame import Rect, image
import random
class Player(AnimateEntity):
    animations = {
        1: {
            "move": [
                * [image.load("assets/images/player1/move/move1.png").convert() for _ in range(5)], 
                * [image.load("assets/images/player1/move/move2.png").convert() for _ in range(2)]
                ],
            "idle": [
                * [image.load("assets/images/player1/idle/idle1.png").convert() for _ in range(10)], 
                * [image.load("assets/images/player1/idle/idle6.png").convert() for _ in range(5)]
                ]
        },
        2: {
            "move": [
                * [image.load("assets/images/player2/move/move1.png").convert() for _ in range(5)],
                * [image.load("assets/images/player2/move/move2.png").convert() for _ in range(2)]
                ],
            "idle": [
                * [image.load("assets/images/player2/idle/idle1.png").convert() for _ in range(10)],
                * [image.load("assets/images/player2/idle/idle6.png").convert() for _ in range(5)]
                ]
        },
        3: {
            "move": [
                * [image.load("assets/images/player3/move/move1.png").convert() for _ in range(5)], 
                * [image.load("assets/images/player3/move/move2.png").convert() for _ in range(2)]
                ],
            "idle": [
                * [image.load("assets/images/player3/idle/idle1.png").convert() for _ in range(10)], 
                * [image.load("assets/images/player3/idle/idle6.png").convert() for _ in range(5)]
                ]
        },
        4: {
            "move": [
                * [image.load("assets/images/player4/move/move1.png").convert() for _ in range(5)], 
                * [image.load("assets/images/player4/move/move2.png").convert() for _ in range(2)]
                ],
            "idle": [
                * [image.load("assets/images/player4/idle/idle1.png").convert() for _ in range(10)], 
                * [image.load("assets/images/player4/idle/idle6.png").convert() for _ in range(5)]
                ]
        },
    }
    def __init__(self, x, y, speed, id):
        super().__init__(x, y, speed)
        self.prev_tile = [self.tile_x, self.tile_y] 
        self.id = id
        self.neat_output = []
        self.super_bomb_limit = 0
        self.speed_boost_limit = 0
        self.noclip_time_limit = 0
        self.move_timer = Stopwatch()

        self.idle = True
        self.facingUp = False
        self.facingDown = False
        self.facingLeft = False
        self.facingRight = False
        self.timer = Stopwatch()
        self.speed_timer = Stopwatch()
        self.noclip_timer = Stopwatch()
        self.noClip = False 
        self.is_bot = False
        self.idleAnimation = Player.animations[1]["idle"]
        self.moveAnimation = Player.animations[1]["move"]
        self.kills = 0
        self.sprite = self.idleAnimation[0]
        self.Rect = Rect(self.x+3, self.y+3, self.sprite.get_width()-3, self.sprite.get_height()-3)
        self.has_bomb = True

    def get_nearest_enemy(self):
        nearest_enemy_coordinates = (0, 0)
        nearest_enemy_distance = float("inf")
        for enemy in Game.enemies:
            enemy_distance = self.get_distance((enemy.x, enemy.y)) 
            if enemy_distance < nearest_enemy_distance:
                nearest_enemy_distance = enemy_distance
                nearest_enemy_coordinates = (enemy.x, enemy.y)
        return nearest_enemy_coordinates

    def renderIdTag(self):
        if not self.is_bot:
            id_label = Text(str(self.id), size="xs", fg="white", bg="black", align=[self.x + self.sprite.get_width()/2, self.y], display=False)
        else: 
            id_label = Text(str(self.id), size="xs", fg="red", bg="black", align=[self.x + self.sprite.get_width()/2, self.y], display=False)
        id_label.y -= id_label.text.get_height() + 5
        id_label.x -= id_label.text.get_width()/2
        Game.surface.blit(id_label.text, (id_label.x, id_label.y))

    def placeBomb(self):
        if self.has_bomb:
            self.has_bomb = False
            self.timer.reset()
            Game.entities.append(BombActive(
                self.tile_x,
                self.tile_y,
                self.x_offset,
                self.y_offset,
                self
            ))

    def moveBot(self):
        if not self.neat_output:
            return
        if self.neat_output[0] > 0.5:
            self.faceUp = True
        if self.neat_output[1] > 0.5:
            self.faceDown = True
        if self.neat_output[2] > 0.5:
            self.faceRight = True
        if self.neat_output[3] > 0.5:
            self.faceLeft = True
        if self.neat_output[4] > 0.5:
            self.placeBomb()

        if [x for x in Game.map_item if isinstance(x, (Wall, Box)) and self.Rect.colliderect(x.Rect)]:
            Game.genomes.pop(Game.players.index(self))
            Game.players.remove(self)

        # if self.faceUp and isinstance(Game.map_item[Game.change2Dto1DIndex(self.tile_x, self.tile_y - 1)], (Wall, Box)):
        #     Game.genomes.pop(Game.players.index(self))
        #     Game.players.remove(self)
        #     return
        # if self.faceDown and isinstance(Game.map_item[Game.change2Dto1DIndex(self.tile_x, self.tile_y + 1)], (Wall, Box)):
        #     Game.genomes.pop(Game.players.index(self))
        #     Game.players.remove(self)
        #     return
        # if self.faceLeft and isinstance(Game.map_item[Game.change2Dto1DIndex(self.tile_x - 1, self.tile_y)], (Wall, Box)):
        #     Game.genomes.pop(Game.players.index(self))
        #     Game.players.remove(self)
        #     return
        # if self.faceRight and isinstance(Game.map_item[Game.change2Dto1DIndex(self.tile_x + 1, self.tile_y)], (Wall, Box)):
        #     Game.genomes.pop(Game.players.index(self))
        #     Game.players.remove(self)
        #     return

        if (not self.faceUp) and (not self.faceDown) and (not self.faceLeft) and (not self.faceRight):
            Game.genomes[Game.players.index(self)].fitness -= 0.05
            if self.move_timer.time_elapsed() > 1000:
                Game.genomes.pop(Game.players.index(self))
                Game.players.remove(self)
                return
        else:
            self.move_timer.reset()
                
        if self.get_distance(self.get_nearest_enemy()) < 50:
            Game.genomes[Game.players.index(self)].fitness -= 0.1
            

        self.move()
        self.faceUp = False
        self.faceDown = False
        self.faceLeft = False
        self.faceRight = False

    def animate(self):
        if self.is_bot:
            self.moveBot()
        
        # Handle bomb placement
        if self.timer.time_elapsed() > 3_000:
            self.has_bomb = True
        item = BombItem(
            (self.x-self.x_offset)/Game.settings["game.tileSize"], 
            (self.y-self.y_offset)/Game.settings["game.tileSize"],
            self.x_offset,
            self.y_offset)

        # Animations
        if self.idle:
            if self.frame < len(self.idleAnimation) - 1:
                self.frame += 1
            else:
                self.frame = 0
            self.sprite = self.idleAnimation[self.frame]
        else:
            if self.frame < len(self.moveAnimation) - 1:
                self.frame += 1
            else:
                self.frame = 0 
            self.sprite = self.moveAnimation[self.frame]
        self.sprite.set_colorkey((0, 0, 0, 0))

        Game.surface.blit(self.sprite, (self.Rect.x, self.Rect.y))
        if self.has_bomb:
            if self.super_bomb_limit > 0:
                Game.surface.blit(BombItem.ignited_sprite, (item.x, item.y)) 
            else:
                Game.surface.blit(BombItem.sprite, (item.x, item.y))

        self.renderIdTag() # Number tag
        
        for i in self.hit_test(self.Rect, filter(lambda x: isinstance(x, Explosion), Game.entities)):
            if i.player != self:
                try: # Sometimes the explosion accidentally delete a player twice, this is a simple fix for that
                    Game.players.remove(self)
                    for player in Game.players:
                        if player.is_bot:
                            player.target = random.choice(Game.players)
                            while player.target == player and len(Game.players) > 1:
                                player.target = random.choice(Game.players)
                    i.player.kills += 1
                    Game.genomes[Game.players.index(i)].fitness += 100
                except:
                    pass
        for enemy in Game.enemies:
            if self.Rect.colliderect(enemy.Rect) == 1:
                try: # Sometimes the explosion accidentally delete a player twice, this is a simple fix for that
                    Game.players.remove(self)
                    if Game.players:
                        for enemy in Game.enemies:
                            enemy.target_player = random.choice(Game.players)
                except:
                    pass

        if self.speed_boost_limit > 0:
            if self.speed_timer.time_elapsed() < self.speed_boost_limit:
                Game.surface.blit(
                    scale(
                        image.load("assets/images/speed_boost.png"),
                        (int(InanimateEntity.tile_size * 0.5), int(InanimateEntity.tile_size * 0.5))
                    ),
                    (self.x, self.y)
                )
            else:
                self.speed_timer.reset()
                self.speed_boost_limit = 0
        else:
            self.speed_timer.reset()

        if self.noclip_time_limit > 0:
            if self.noclip_timer.time_elapsed() < self.noclip_time_limit:
                Game.surface.blit(
                    scale(
                        image.load("assets/images/noclip.png"),
                        (int(InanimateEntity.tile_size * 0.5), int(InanimateEntity.tile_size * 0.5))
                        ),
                    (self.x + Game.settings["game.tileSize"] - 15, self.y + Game.settings["game.tileSize"] - 15)
                )
            else:
                self.noclip_timer.reset()
                self.noclip_time_limit = 0
                self.noClip = False
        else:
            self.noclip_timer.reset()
            
