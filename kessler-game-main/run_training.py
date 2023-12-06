import time
import math
import EasyGA

from src.kesslergame import Scenario, KesslerGame, GraphicsType
from examples.test_controller import TestController
from examples.scott_dick_controller import ScottDickController
from examples.smart_controller import SmartController
from examples.graphics_both import GraphicsBoth

def evaluate_fitness(chromosome, scenarios):
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
        score,perf_data = game.run(scenario=scenario, controllers=[MyController(), ScottDickController()])
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
    chromosome['bullet_time'] = [0,0,0.05,0,0.05,0.1,0.0,0.1]
    chromosome['theta_delta'] = [-1*math.pi/3,-1*math.pi/6,-1*math.pi/3,-1*math.pi/6,0,-1*math.pi/6,0,math.pi/6,0,math.pi/6,math.pi/3,math.pi/6,math.pi/3]
    chromosome['nearestAsteroidDistance'] = [0, 0, 150,120, 200, 300,200, 400, 400]
    chromosome['currVelocity'] = [-300, -200, -100,-200, -100, 0,-100, 0, 100,100, 200, 300,0, 100, 200]

    # generate fuzzy sets for consequents
    chromosome['ship_turn'] = [-180,-180,-30,-90,-30,0,-30,0,30,0,30,90,30,180,180]
    chromosome['ship_fire'] = [-1,-1,0.0,0.0,1,1]
    chromosome['thrust'] = [-300, -200, -100,-200, -100, 0,-100, 0, 100,100, 200, 300,0, 100, 200]

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

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

# initialize population
#population = generate_chromosome()
# set parameters of genetic algorithm
ga = EasyGA.GA()
ga.gene_impl = lambda: generate_chromosome()
ga.chromosome_length = 1
ga.population_size = 100
ga.target_fitness_type = 'min'
ga.generation_goal = 3
# need to see what the syntax is when two parameters are passed
ga.fitness_function_impl = evaluate_fitness
ga.evolve()
ga.print_best_chromosome()

# started to implement this approach based on pseudo code; might delete
""" # evaluate the fitness on the different chromosomes on the training scenario
# still need to think of stopping conditions
while stopping_conditions_not_met:
    for individual in population:
        # evaluate fitness of current individual on training scenario
        evaluate_fitness(individual,my_training_scenario)
        # modify population """


# leftover from scenario_test.py; might delete
""" # Print out some general info about the result
print('Scenario eval time: '+str(time.perf_counter()-pre))
print(score.stop_reason)
print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
print('Deaths: ' + str([team.deaths for team in score.teams]))
print('Accuracy: ' + str([team.accuracy for team in score.teams]))
print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams])) """