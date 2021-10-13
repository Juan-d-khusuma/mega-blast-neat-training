from objects.entity import Explosion, Wall
from objects.game import Game
from objects.map import MapFactory, MapRenderer
from objects.player import Player
from objects.stopwatch import Stopwatch
from objects.text import Text
from pygame import init, display, event, QUIT, mixer, mouse, quit, time, mixer
import sys, os, neat, gym, random, pickle, multiprocessing

# env = gym.make("Bomberman-v0")

mapFactory = MapFactory()
init()
mixer.init()

display.set_caption("Mega Blast")
clock = time.Clock()
round = 1

timer = Stopwatch()

target_player = Player(
    (19 * Game.settings["game.tileSize"] + Game.x_offset), 
    (19 * Game.settings["game.tileSize"] + Game.y_offset), 
    Game.settings["game.playerSpeed"], 999
)

player_count_options = [x + 1 for x in range(4)]
bot_count_options = [x for x in range(4)]
win_limit_options = [x + 1 for x in range(9)]
player_count_selected = Game.gameConf["game.player.count"]
bot_count_selected = Game.gameConf["game.bot.count"]
win_limit_selected = Game.gameConf["game.win.limit"]

round_winner = None

def main(genomes, config):  # sourcery no-metrics

    """ INPUTS:
    - distance to nearest player
    - distance to nearest enemy
    - 5 surrounding tiles
    
    """
    Game.players = []
    j = 5
    for i, (_, genome) in enumerate(genomes, start=1):
        genome.fitness = 0
        while not Game.map["map"][i][j] == "e":
            if j < Game.map_width - 1:
                j += 1
            else:
                j = 1
                i += 1
        player = Player(
            (i * Game.settings["game.tileSize"] + Game.x_offset), 
            (j * Game.settings["game.tileSize"] + Game.y_offset), 
            Game.settings["game.playerSpeed"], i
        )
        j += 1
        player.is_bot = True
        Game.players.append(player)
        Game.genomes.append(genome)
        Game.networks.append(neat.nn.FeedForwardNetwork.create(genome, config))

    mapRenderer = MapRenderer()
    global round_over, round
    while True:
        if timer.time_elapsed() > 30_000 or not Game.players:
            MapFactory()
            mapRenderer.start()
            round += 1
            timer.reset()
            break

        for i, player in enumerate(Game.players):
            if isinstance(player, Player):
                # Game.genomes[i].fitness += 0.1
                upper_tile = 1 if isinstance(Game.map_item[Game.change2Dto1DIndex(player.tile_x, player.tile_y-1)], Wall) else 0
                lower_tile = 1 if isinstance(Game.map_item[Game.change2Dto1DIndex(player.tile_x, player.tile_y+1)], Wall) else 0
                left_tile = 1 if isinstance(Game.map_item[Game.change2Dto1DIndex(player.tile_x-1, player.tile_y)], Wall) else 0
                right_tile = 1 if isinstance(Game.map_item[Game.change2Dto1DIndex(player.tile_x+1, player.tile_y)], Wall) else 0
                target_distance = player.get_distance((target_player.x, target_player.y))
                Game.genomes[i].fitness += 10/target_distance
                if target_player.hit_test(target_player.Rect, filter(lambda x: isinstance(x, Explosion), Game.entities)):
                    Game.genomes[i].fitness += 200
                    break
                output = Game.networks[i].activate((
                    target_player.x,
                    target_player.y,
                    player.x, 
                    player.y, 
                    *player.get_nearest_player(), 
                    *player.get_nearest_enemy(),
                    upper_tile,
                    lower_tile,
                    left_tile,
                    right_tile,
                    player.get_distance(player.get_nearest_player()),
                    player.get_distance(player.get_nearest_enemy()),
                    target_distance,
                ))
                player.neat_output = output

        for e in event.get():
            if e.type == QUIT:
                quit()
                sys.exit()

        Game.surface.fill("black")


        mapRenderer.render()
        Game.framerate = int(clock.get_fps())
        Text(str(Game.framerate), size="xl", fg="white", bg="black", align="top-right")
        Text("Round {}".format(round), size="xl", fg="white", bg="black", align="top-center")
        target_player.animate()
        display.update()
        clock.tick(30)

def run(config_file):
    config = neat.config.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        config_file
    )

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-9")
    winner = p.run(main, 100)   
    with open("winner", "wb") as x:
        pickle.dump(winner, x)
        print("Dumping the last fittest_gene")
    print(winner)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)