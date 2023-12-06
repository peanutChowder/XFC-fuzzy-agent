import time
import math
import EasyGA

from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from examples.test_controller import TestController
from examples.scott_dick_controller import ScottDickController
from examples.smart_controller import SmartController
from examples.graphics_both import GraphicsBoth

def evaluate_fitness(chromosome):
    """
    Evaluates the fitness of the current individual on a set of scenarios.

    inputs:
    1. chromosome
    2. scenarios (set of scenarios to be iterated)

    return: total_score (calculated based on asteroid kills and deaths of current controller in the scenarios.)
    """

    total_score = 0
    # SmartController class still needs to be modified to take individual as an input
    MyController = SmartController(chromosome)

    for scenario in scenarios:
        # evaluate the game
        pre = time.perf_counter()
        score,perf_data = game.run(scenario=scenario, controllers=[MyController, ScottDickController()])
        # calculate the current fitness and add it to the total score
        # my controller is set as first controller when running the game -> access via index 0
        asteroid_kills = score.teams[0].asteroids_hit
        deaths = score.teams[0].deaths

        fitness = asteroid_kills - 30*deaths
        total_score += fitness
    
    return total_score

def generate_chromosome():
    """
    Creates a chromosome to initialize the SmartController with.

    return: chromosome
    """
    chromosome = {}

    # generate fuzzy sets for antecedents
    # values hardcoded for a first testing
    # after successful testing -> generate random values
    chromosome['bullet_time'] = [0,0,0.05,0,0.07,0.15,0.0,0.2]
    chromosome['theta_delta'] = [-1*math.pi, -5/9 * math.pi, -3/4 * math.pi, -1/2*math.pi, -1/4 * math.pi, -1/2 * math.pi, -1/4 * math.pi,0, -1/180 * math.pi, 0, 1/180 * math.pi, 0, 1/4 * math.pi, 1/2 * math.pi, 1/4 * math.pi, 1/2 * math.pi, 3/4 * math.pi, 5/9 * math.pi, math.pi]
    chromosome['asteroidDistance'] = [0, 0, 200, 100, 150, 200, 200, 350]
    chromosome['asteroidSpeed'] = [0, 50, 30, 70]
    chromosome['currVelocity'] = [-300, -250, -100, -150, -70, 5, -5, 0, 5, 100, 250, 300, 5, 90, 200]

    # generate fuzzy sets for consequents
    chromosome['ship_turn'] = [-180, -100, -135, -120, -45, -90, -60, 0, -1, 0, 1, 0, 60, 90, 45, 120, 135, 100, 180]
    chromosome['ship_fire'] = [-1,-1,0.0,0.0,1,1]
    chromosome['thrust'] = [-300, -300, -150, -200, -100, 50, -5, 0, 5, 150, 300, 300, 50, 100, 200]

    return chromosome



# Define game scenario
my_training_scenario = Scenario(name='Train Scenario',
                            num_asteroids=10,
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=60,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

# Define Game Settings
game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 1,
                 'graphics_obj': None,
                 'frequency': 30}

# set game as global variable for fitness function
global game
#game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

# set scenario as global variable for fitness function
global scenarios
scenarios = [my_training_scenario]


# initialize population
#population = generate_chromosome()
# set parameters of genetic algorithm
ga = EasyGA.GA()
ga.gene_impl = lambda: generate_chromosome()
ga.chromosome_length = 1
ga.population_size = 5
ga.target_fitness_type = 'max'
ga.generation_goal = 1
# need to see what the syntax is when two parameters are passed
ga.fitness_function_impl = evaluate_fitness
ga.evolve()
ga.print_best_chromosome()

# get best chromosome

# create controller with best chromosome
