from io import IncrementalNewlineDecoder
from objects.text import Text
from pygame import display, image
from objects.game import Game
from objects.entity import Box, Empty, Mimic, Wall
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
        print("Generating new map")
        Game.modifyGameData("map", {"map": self._map, "test": self.test})

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
                if cs % 10 == 1:
                    print('%s/%s cells self.connected ...' %
                          (cs, ss), file=sys.stderr)

    def __generateMaze(self):
        print("Generating Maze...")
        self.width += 2
        self.height += 2
        self.__initialize()
        self.__borderFill()
        self.__primAlg()
        _map = [''.join(self.maze[(i, j)] for j in range(self.width))
                for i in range(self.height)]
        for i, row in enumerate(_map):
            _map[i] = list(row)
        for c, a in enumerate(_map):
            for d, b in enumerate(a):
                if c > 0 and c < len(_map) - 1 and d > 0 and d < len(a) - 1:
                    if random.randrange(1, 40) == 4 and b == "w":
                        _map[c][d] = "m"
                    if random.randrange(1, 5) == 3 and b == "w":
                        _map[c][d] = "e"

        return _map


class MapRenderer(Game):
    bomb = image.load("assets/images/regular_bomb.png")
    def __init__(self):
        print("Map renderer initialized...")
        super().__init__()
        self.map_item = []
        self.players = []
        self.map_width = len(Game.map["map"])
        self.map_height = len(Game.map["map"][0])
        Game.x_offset = Game.resolution[0]/2 / Game.settings["game.tileSize"] - self.map_width/2
        Game.y_offset = Game.resolution[1]/2 / Game.settings["game.tileSize"] - self.map_height/2
        self.start()

    def start(self):
        self.map_item = []
        player_count = Game.gameConf["game.player.count"]
        bot_count = Game.gameConf["game.bot.count"]
        space = 5
        for i, column in enumerate(Game.map["map"]):
            for j, item in enumerate(column):
                if item == "b":
                    self.map_item.append(Box(i + self.x_offset, j + self.y_offset))
                elif item == "w":
                    self.map_item.append(Wall(i + self.x_offset, j + self.y_offset))
                elif item == "m":
                    self.map_item.append(Mimic(i + self.x_offset, j + self.y_offset))
                elif item == "e":
                    if space > 0 or player_count == 0:
                        self.map_item.append(Empty(i + self.x_offset, j + self.y_offset))
                        space -= 1
                    else:
                        space = 5
                        self.players.append(Player((i + self.x_offset) * Game.settings["game.tileSize"], (j + self.y_offset) * Game.settings["game.tileSize"], 5, player_count))
                        player_count -= 1


    def render(self):
        # Game.surface.blit(self.player1.sprite, ((self.player1.x + Game.x_offset)*Game.settings["game.tileSize"], ((self.player1.y+Game.y_offset)*Game.settings["game.tileSize"])))
        # Game.surface.blit(self.player2.sprite, ((self.player2.x + Game.x_offset)*Game.settings["game.tileSize"], ((self.player2.y+Game.y_offset)*Game.settings["game.tileSize"]))) 
        # self.player1.facingRight = True
        # self.player1.idle = False
        # self.player1.move()
        # Game.surface.blit(MapRenderer.bomb, ((self.player1.x + self.x_offset)*Game.settings["game.tileSize"] + 5, ((self.player1.y+self.y_offset)*Game.settings["game.tileSize"])+10))
        for item in self.map_item:
            if isinstance(item, Wall):
                Game.surface.blit(Wall.sprite, (item.x, item.y))
            elif isinstance(item, Box):
                Game.surface.blit(Box.sprite, (item.x, item.y))
            elif isinstance(item, Mimic):
                Game.surface.blit(Mimic.sprite, (item.x, item.y))
        for item in self.players:
            if isinstance(item, Player):
                item.animate()
                item.move()

