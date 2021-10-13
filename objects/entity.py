from objects.text import Text
from objects.game import Game
from pygame import Rect, image
from pygame.transform import scale
from objects.stopwatch import Stopwatch
import math

class InanimateEntity(Game):
    sprite = None
    tile_size = Game.settings["game.tileSize"]

    def __init__(self, tileX, tileY, offsetX, offsetY):
        self.tileX = tileX
        self.tileY = tileY
        self.x, self.y = (tileX * InanimateEntity.tile_size) + offsetX, (tileY * InanimateEntity.tile_size) + offsetY
        self.Rect = Rect(self.x, self.y, Game.settings["game.tileSize"], Game.settings["game.tileSize"])
        super().__init__()
class PowerUp(InanimateEntity):
    sprite = scale(
        image.load("assets/images/power_up.png"),
        (Game.settings["game.tileSize"],
        Game.settings["game.tileSize"])
    ).convert() 
    def __init__(self, tileX, tileY, offsetX, offsetY):
        super().__init__(tileX, tileY, offsetX, offsetY)
        self.player = None
        
    def modify(self):
        pass

    def update(self):
        player = [x for x in Game.players if self.Rect.colliderect(x.Rect)]
        if player:
            self.player = player[0]
            self.modify()
            Game.map_item[Game.change2Dto1DIndex(self.tileX, self.tileY)] = Empty(self.tileX, self.tileY, self.x_offset, self.y_offset)
class SuperExplosive(PowerUp):
    sprite = scale(
        image.load("assets/images/ignited_bomb.png"),
        (Game.settings["game.tileSize"], Game.settings["game.tileSize"])
    )
    def modify(self):
        self.player.super_bomb_limit = 5

    def update(self):
        super().update()
        Game.surface.blit(SuperExplosive.sprite, (self.x, self.y))

class NoClip(PowerUp):
    sprite = scale(
        image.load("assets/images/noclip.png"),
        (Game.settings["game.tileSize"], Game.settings["game.tileSize"])
    )
    def modify(self):
        self.player.noclip_time_limit = 5_000
        self.player.noClip = True

    def update(self):
        super().update()
        Game.surface.blit(NoClip.sprite, (self.x, self.y))
class Speed2x(PowerUp):
    sprite = scale(
        image.load("assets/images/speed_boost.png"),
        (Game.settings["game.tileSize"], Game.settings["game.tileSize"])
    )
    def modify(self):
        self.player.speed_boost_limit = 5_000

    def update(self):
        super().update()
        Game.surface.blit(Speed2x.sprite, (self.x, self.y))
class AnimateEntity(Game):
    def __init__(self, x, y, speed):
        # Speed will be in tile per second
        self.x = x
        self.y = y
        self.tile_x = round((self.x-self.x_offset)/Game.settings["game.tileSize"])
        self.tile_y = round((self.y-self.y_offset)/Game.settings["game.tileSize"])
        self.__speed = speed
        self.Rect = Rect(0, 0, 0, 0)
        self.speed = 0
        self.sprite = None
        self.idle = False
        self.faceUp = False
        self.faceDown = False
        self.faceLeft = False
        self.faceRight = False
        self.moveSprite = []
        self.idleSprite = []
        self.kills = 0
        self.id = 0
        self.frame = 0
        self.prev_pos = [self.x, self.y]
        self.is_bot = False
        super().__init__()

    def collission_test(self, rect, tiles):
        return [tile for tile in tiles if rect.colliderect(tile.Rect) == 1 and not isinstance(tile, (Empty, PowerUp))]

    def hit_test(self, rect, explosions):
        return [fire for fire in explosions if rect.colliderect(fire.Rect) == 1]

    def get_distance(self, target):
        [x, y] = target
        return math.sqrt((x-self.x)**2+(y-self.y)**2)

    def get_nearest_player(self, for_bot=False):
        nearest_player_coordinates = (0, 0)
        nearest_player_tile_coordinates = (0, 0)
        nearest_player_distance = float("inf")
        for player in Game.players:
            player_distance = self.get_distance((player.x, player.y)) 
            if player_distance < nearest_player_distance and player.id != self.id:
                nearest_player_distance = player_distance
                nearest_player_coordinates = (player.x, player.y)
                nearest_player_tile_coordinates = (player.tile_x + 1, player.tile_y + 1)
        if for_bot:
            return nearest_player_tile_coordinates
        return nearest_player_coordinates
        
    def move(self):
        self.prev_pos = [self.x, self.y]
        self.tile_x = round((self.x-self.x_offset)/Game.settings["game.tileSize"])
        self.tile_y = round((self.y-self.y_offset)/Game.settings["game.tileSize"])
        if Game.framerate > 0:
            self.speed = self.__speed * Game.settings["game.tileSize"] / Game.framerate
            if self.speed_timer.time_elapsed() < self.speed_boost_limit:
                self.speed *= 3
            if self.is_bot:
                self.speed *= 0.7
        else:
            self.speed = 0
        if self.noClip:
            if self.faceUp and self.y - self.speed > Game.y_offset + Game.settings["game.tileSize"]:
                self.y -= self.speed
            if self.faceDown and self.y + self.speed < Game.y_offset + (Game.map_height*Game.settings["game.tileSize"]) - Game.settings["game.tileSize"]:
                self.y += self.speed
            if self.faceLeft and self.x - self.speed > Game.x_offset + Game.settings["game.tileSize"]:
                self.x -= self.speed
            if self.faceRight and self.x + self.speed < Game.x_offset +( Game.map_width*Game.settings["game.tileSize"]) - Game.settings["game.tileSize"]:
                self.x += self.speed

            self.Rect.y = self.y 
            self.Rect.x = self.x
            self.tile_x = round((self.x-self.x_offset)/Game.settings["game.tileSize"])
            self.tile_y = round((self.y-self.y_offset)/Game.settings["game.tileSize"])
            return 

        if self.faceUp:
            self.y -= self.speed
        if self.faceDown:
            self.y += self.speed
        self.Rect.y = self.y 
        hit_list = self.collission_test(self.Rect, Game.map_item)
        for tile in hit_list:
            if self.faceUp:
                self.Rect.top  = tile.Rect.bottom
            if self.faceDown:
                self.Rect.bottom = tile.Rect.top
        if self.faceLeft:
            self.x -= self.speed
        if self.faceRight:
            self.x += self.speed
        self.Rect.x = self.x
        hit_list = self.collission_test(self.Rect, Game.map_item)
        for tile in hit_list:
            if self.faceLeft:
                self.Rect.left = tile.Rect.right
            if self.faceRight:
                self.Rect.right = tile.Rect.left

        self.x = self.Rect.x
        self.y = self.Rect.y
        self.tile_x = round((self.x-self.x_offset)/Game.settings["game.tileSize"])
        self.tile_y = round((self.y-self.y_offset)/Game.settings["game.tileSize"])

class Empty(InanimateEntity):
    def __str__(self):
        return "Empty<{}({}),{}({})>".format(self.x, self.tileX, self.y, self.tileY)


class Box(InanimateEntity):
    sprite = scale(image.load("assets/images/box.png").convert(),
                   (InanimateEntity.tile_size, InanimateEntity.tile_size))

    def __str__(self):
        return "Box<{}({}),{}({})>".format(self.x, self.tileX, self.y, self.tileY)


class Wall(InanimateEntity):
    sprite = scale(image.load("assets/images/wall.png").convert(),
                   (InanimateEntity.tile_size, InanimateEntity.tile_size))

    def __str__(self):
        return "Wall<{}({}),{}({})>".format(self.x, self.tileX, self.y, self.tileY)

class BombItem(InanimateEntity):
    ignited_sprite = scale(image.load("assets/images/ignited_bomb.png").convert(), (int(InanimateEntity.tile_size * 0.5), int(InanimateEntity.tile_size * 0.5)))
    sprite = scale(image.load("assets/images/regular_bomb.png").convert(), (int(InanimateEntity.tile_size * 0.5), int(InanimateEntity.tile_size * 0.5)))
    sprite.set_colorkey((0, 0, 0, 0))
    ignited_sprite.set_colorkey((0, 0, 0, 0))
    def __init__(self, tileX, tileY, offsetX, offsetY):
        super().__init__(tileX, tileY, offsetX, offsetY)
        self.x += (self.tile_size - BombItem.sprite.get_width())/2 
        self.x += (self.tile_size - BombItem.sprite.get_height())/2 
        self.player = None

class Explosion(InanimateEntity):
    animations = []
    animations += [scale(image.load("assets/images/explosions/explosion{}.png".format(j + 1)).convert(), (InanimateEntity.tile_size, InanimateEntity.tile_size)) for j in range(6)]
    def __init__(self, tileX, tileY, offsetX, offsetY, placed_by):
        super().__init__(tileX, tileY, offsetX, offsetY)
        self.timer = Stopwatch()
        self.timer.reset()
        self.player = placed_by
        self.animations = []
        self.frame = 0
        
        Explosion.sprite = Explosion.animations[0]

    def animate(self):
        if self.frame < len(Explosion.animations) - 1:
            self.frame += 1
        else:
            self.frame = 0
        Explosion.sprite = Explosion.animations[self.frame]
        Game.surface.blit(Explosion.sprite, (self.x, self.y))
        if self.timer.time_elapsed() > 200:
            Game.entities.pop(0)


class BombActive(InanimateEntity):
    sprite = scale(image.load("assets/images/regular_bomb.png").convert(), (InanimateEntity.tile_size, InanimateEntity.tile_size))
    animations = []
    animations += [scale(image.load("assets/images/regular_bomb.png").convert(), (InanimateEntity.tile_size, InanimateEntity.tile_size)) for _ in range(10)]
    animations += [scale(image.load("assets/images/ignited_bomb.png").convert(), (InanimateEntity.tile_size, InanimateEntity.tile_size)) for _ in range(10)]

    def __init__(self, tileX, tileY, offsetX, offsetY, placed_by):
        super().__init__(tileX, tileY, offsetX, offsetY)
        self.player = placed_by
        self.timer = Stopwatch()
        self.timer.reset()
        self.text = Text(str(self.player.id), size="xs", fg="white", bg="black", align=[self.player.x, self.player.y], display=False)
        self.text.x += (BombActive.sprite.get_width() - self.text.text.get_width())/2
        self.text.y += (BombActive.sprite.get_height() - self.text.text.get_height())/2
        self.frame = 0
        self.stop_propagation = {
            "UP": False,
            "DOWN": False,
            "LEFT": False,
            "RIGHT": False,
        }

    def __add_explosion_up(self, i):
        item = Game.map_item[Game.change2Dto1DIndex(self.tileX, self.tileY - i)]
        if isinstance(item, (Wall, PowerUp)):
            self.stop_propagation["UP"] = True
        else:
            if isinstance(Game.change2Dto1DIndex(self.tileX, self.tileY - i), Box):
                Game.genomes[Game.players.index(self.player)].fitness += 10
            Game.map_item[Game.change2Dto1DIndex(self.tileX, self.tileY - i)] = Empty(self.tileX, self.tileY - i, self.x_offset, self.y_offset)
            Game.entities.append(Explosion(self.tileX, self.tileY - i, self.x_offset, self.y_offset, self.player)) # Up

    def __add_explosion_down(self, i):
        item = Game.map_item[Game.change2Dto1DIndex(self.tileX, self.tileY + i)]
        if isinstance(item, (Wall, PowerUp)):
            self.stop_propagation["DOWN"] = True
        else:
            if isinstance(Game.change2Dto1DIndex(self.tileX, self.tileY - i), Box):
                Game.genomes[Game.players.index(self.player)].fitness += 10
            Game.map_item[Game.change2Dto1DIndex(self.tileX, self.tileY + i)] = Empty(self.tileX, self.tileY + i, self.x_offset, self.y_offset)
            Game.entities.append(Explosion(self.tileX, self.tileY + i, self.x_offset, self.y_offset, self.player)) # Down

    def __add_explosion_left(self, i):
        item = Game.map_item[Game.change2Dto1DIndex(self.tileX - i, self.tileY)]
        if isinstance(item, (Wall, PowerUp)):
            self.stop_propagation["LEFT"] = True
        else:
            if isinstance(Game.change2Dto1DIndex(self.tileX, self.tileY - i), Box):
                Game.genomes[Game.players.index(self.player)].fitness += 10
            Game.map_item[Game.change2Dto1DIndex(self.tileX - i, self.tileY)] = Empty(self.tileX - i, self.tileY, self.x_offset, self.y_offset)
            Game.entities.append(Explosion(self.tileX - i, self.tileY, self.x_offset, self.y_offset, self.player)) # Left

    def __add_explosion_right(self, i):
        item = Game.map_item[Game.change2Dto1DIndex(self.tileX + i, self.tileY)]
        if isinstance(item, (Wall, PowerUp)):
            self.stop_propagation["RIGHT"] = True
        else:
            if isinstance(Game.change2Dto1DIndex(self.tileX, self.tileY - i), Box):
                Game.genomes[Game.players.index(self.player)].fitness += 10
            Game.map_item[Game.change2Dto1DIndex(self.tileX + i, self.tileY)] = Empty(self.tileX + i, self.tileY, self.x_offset, self.y_offset)
            Game.entities.append(Explosion(self.tileX + i, self.tileY, self.x_offset, self.y_offset, self.player)) # Right

    def explode(self):
        Game.entities.append(Explosion(self.tileX, self.tileY, self.x_offset, self.y_offset, self.player))
        if self.player.super_bomb_limit <= 0:
            for i in range(1, 3):
                if not self.stop_propagation["UP"]:
                    self.__add_explosion_up(i)
                if not self.stop_propagation["DOWN"]:
                    self.__add_explosion_down(i)
                if not self.stop_propagation["LEFT"]:
                    self.__add_explosion_left(i)
                if not self.stop_propagation["RIGHT"]:
                    self.__add_explosion_right(i)
        else:
            for i in range(1, 5):              
                if not self.stop_propagation["UP"]:
                    self.__add_explosion_up(i)
                if not self.stop_propagation["DOWN"]:
                    self.__add_explosion_down(i)
                if not self.stop_propagation["LEFT"]:
                    self.__add_explosion_left(i)
                if not self.stop_propagation["RIGHT"]:
                    self.__add_explosion_right(i)
            self.player.super_bomb_limit -= 1
        Game.entities.pop(0)
        self.stop_propagation["UP"] = False
        self.stop_propagation["DOWN"] = False
        self.stop_propagation["LEFT"] = False
        self.stop_propagation["RIGHT"] = False

    def animate(self):
        if self.frame < len(BombActive.animations) - 1:
            self.frame += 1
        else:
            self.frame = 0
        BombActive.sprite = BombActive.animations[self.frame]
        Game.surface.blit(BombActive.sprite, (self.x, self.y))
        Game.surface.blit(self.text.text, (self.text.x, self.text.y))
        if self.timer.time_elapsed() > 1_000:
            self.explode()

    def __str__(self):
        return "Bomb<time:{}>".format(1_000 - self.timer.time_elapsed())

    