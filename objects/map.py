from objects.enemies import Enemies, Harpies, Hunter
from pygame import Surface, image
from objects.game import Game
from objects.entity import AnimateEntity, Box, Empty, Wall, PowerUp, Speed2x, SuperExplosive, NoClip 
from objects.player import Player
import random
import sys

class MapFactory(Game):
    test = 0
    def __init__(self):
        MapFactory.test += 1
        self.test = MapFactory.test
        super().__init__()
        size = self.settings["game.mapSize"]
        self.width, self.height = size, size
        self.EMPTY = 'e'
        self.WALL = 'w'
        self.BOX = 'b'
        self.maze = {}
        self.spaceCells = set()
        self.connected = set()
        self.walls = set()
        self._map = self.__generateMaze()
        Game.modifyGameData("map", {"map": self._map})

    @staticmethod
    def adjacent(cell):
        i, j = cell
        for (y, x) in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            yield (i+y, j+x), (i+2*y, j+2*x)

    def __initialize(self):
        for i in range(self.height):
            for j in range(self.width):
                self.maze[(i, j)] = self.EMPTY if (
                    (i % 2 == 1) and (j % 2 == 1)
                ) else self.WALL

    def __borderFill(self):
        for i in range(self.height):
            self.maze[(i, 0)] = self.WALL
            self.maze[(i, self.width-1)] = self.WALL
        for j in range(self.width):
            self.maze[(0, j)] = self.WALL
            self.maze[(self.height-1, j)] = self.WALL
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[(i, j)] == self.EMPTY:
                    self.spaceCells.add((i, j))
                if self.maze[(i, j)] == self.WALL:
                    self.walls.add((i, j))

    def __primAlg(self, verbose=True):
        originalSize = len(self.spaceCells)
        self.connected.add((1, 1))
        while len(self.connected) < len(self.spaceCells):
            doA, doB = None, None
            cns = list(self.connected)
            random.shuffle(cns)
            for (i, j) in cns:
                if doA is not None:
                    break
                for A, B in self.adjacent((i, j)):
                    if A not in self.walls:
                        continue
                    if (B not in self.spaceCells) or (B in self.connected):
                        continue
                    doA, doB = A, B
                    break
            A, B = doA, doB
            self.maze[A] = self.BOX if random.randrange(
                1, 5) == 4 else self.EMPTY
            self.maze[B] = self.BOX if random.randrange(
                1, 5) == 4 else self.EMPTY
            self.walls.remove(A)
            self.spaceCells.add(A)
            self.connected.add(A)
            self.connected.add(B)
            if verbose:
                cs, ss = len(self.connected), len(self.spaceCells)
                cs += (originalSize - ss)
                ss += (originalSize - ss)

    def __generateMaze(self):
        self.width += 2
        self.height += 2
        self.__initialize()
        self.__borderFill()
        self.__primAlg(False)
        _map = [''.join(self.maze[(i, j)] for j in range(self.width))
                for i in range(self.height)]
        for i, row in enumerate(_map):
            _map[i] = list(row)
        for c, a in enumerate(_map):
            for d, b in enumerate(a):
                if c > 0 and c < len(_map) - 1 and d > 0 and d < len(a) - 1:
                    # if random.randrange(1, 40) == 4 and b == "w":
                    #     _map[c][d] = "m"
                    if random.randrange(1, 5) == 3 and b == "w":
                        _map[c][d] = "e"
                    if random.randrange(1, 60) == 2 and b == "e":
                        _map[c][d] = "sb" # Speed Boost
                    if random.randrange(1, 60) == 2 and b == "e":
                        _map[c][d] = "ub" # Ultra Bomb
                    # if random.randrange(1, 60) == 5 and b == "e":
                    #     _map[c][d] = "nc" # No Clip
        return _map


class MapRenderer(Game):
    bomb = image.load("assets/images/regular_bomb.png")
    def __init__(self):
        super().__init__()
        self.wall_sprite = Surface((Game.resolution[0], Game.resolution[1])).convert()
        self.start()

    def start(self):
        self.wall_sprite = Surface((Game.resolution[0], Game.resolution[1])).convert()
        Game.map_item = []
        Game.enemies = []
        Game.entities = []
        harpies_count = 10
        hunter_count = 10
        space = 15
        for i, column in enumerate(Game.map["map"]):
            for j, item in enumerate(column):
                if item == "b":
                    Game.map_item.append(Box(i, j, Game.x_offset, Game.y_offset))
                elif item == "e":
                    Game.map_item.append(Empty(i, j, Game.x_offset, Game.y_offset))
                    if space > 0:
                        space -= 1
                    else:
                        space = 20
                        if harpies_count > 0:
                            harpies_count -= 1
                            Game.enemies.append(
                                Harpies(
                                    i*Game.settings["game.tileSize"] + self.x_offset, 
                                    ((j*Game.settings["game.tileSize"]) + self.y_offset),
                                    1
                                ))
                            continue
                        elif hunter_count > 0 and harpies_count == 0:
                            hunter_count -= 1
                            Game.enemies.append(
                                Hunter(
                                    i*Game.settings["game.tileSize"] + self.x_offset, 
                                    ((j*Game.settings["game.tileSize"]) + self.y_offset),
                                    1.5
                                ))
                            continue


                # elif item == "nc":
                #     Game.map_item.append(NoClip(i, j, Game.x_offset, Game.y_offset))
                elif item == "sb":
                    Game.map_item.append(Speed2x(i, j, Game.x_offset, Game.y_offset))
                elif item == "ub":
                    Game.map_item.append(SuperExplosive(i, j, Game.x_offset, Game.y_offset))
                elif item == "w":
                    wall = Wall(i, j, Game.x_offset, Game.y_offset)
                    self.wall_sprite.blit(Wall.sprite, (wall.x, wall.y))
                    Game.map_item.append(wall)

    def render(self):
        Game.surface.blit(self.wall_sprite, (0, 0))
        for item in Game.map_item:
            if isinstance(item, Box):
                Game.surface.blit(Box.sprite, (item.x, item.y))
            elif isinstance(item, PowerUp):
                item.update()

        [entity.animate() or (isinstance(entity, AnimateEntity) and entity.move()) for entity in Game.entities]
        [player.animate() or player.move() for player in Game.players]
        [enemy.animate() for enemy in Game.enemies]