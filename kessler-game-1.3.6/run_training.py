######################################################################################################
# ECE 449
# Group 28
# Authors: Jacob Feng, Kristoffer Kussmann, Ismail Altaf Hussain
# 
# This file trains and optimizes the fuzzy system controller for the Kessler game. The training is done
# by passing chromosomes to the 'smart_controller.py' file, where our (non-hardcoded) controller
# accepts a chromosome input.
######################################################################################################


import time
import math
import EasyGA
import json
import numpy as np

from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from examples.test_controller import TestController
from examples.scott_dick_controller import ScottDickController
from examples.smart_controller import SmartController
from examples.graphics_both import GraphicsBoth

game = None

def evaluate_fitness(chromosome, mode="train"):
    """
    Evaluates the fitness of the current individual on a set of scenarios.

    inputs:
    1. chromosome

    return: total_score (calculated based on asteroid kills and deaths of current controller in the scenarios.)
    """

    total_score = 0
    MyController = SmartController(chromosome,mode)

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
    chromosome['bullet_time'] = [0,0,0.05,0,np.random.uniform(0,0.15),0.15,0.0,0.2]
    chromosome['theta_delta'] = [-1*math.pi, -5/9 * math.pi, -3/4 * math.pi, np.random.uniform(-3/4 * math.pi,-1/4 * math.pi), -1/4 * math.pi, -1/2 * math.pi, np.random.uniform(-1/2 * math.pi,0),0, -1/180 * math.pi, np.random.uniform(-1/180 * math.pi,1/180 * math.pi), 1/180 * math.pi, 0, np.random.uniform(0,1/2 * math.pi), 1/2 * math.pi, 1/4 * math.pi, np.random.uniform(1/4 * math.pi,3/4 * math.pi), 3/4 * math.pi, 5/9 * math.pi, math.pi]
    chromosome['asteroidDistance'] = [0, 0, 200, 100, np.random.uniform(100,200), 200, 200, 350]
    chromosome['asteroidSpeed'] = [0, np.random.uniform(35,65), 30, np.random.uniform(55,85)]
    chromosome['currVelocity'] = [-300, -250, -100, -150, np.random.uniform(-150,5), 5, -5, np.random.uniform(-5,5), 5, 100, np.random.uniform(100,300), 300, 5, 90, 200]

    # generate fuzzy sets for consequents
    chromosome['ship_turn'] = [-180, -100, -135, np.random.uniform(-135,-45), -45, -90, np.random.uniform(-90,0), 0, -1, np.random.uniform(-1,1), 1, 0, np.random.uniform(0,90), 90, 45, np.random.uniform(45,135), 135, 100, 180]
    chromosome['ship_fire'] = [-1,-1,0.0,0.0,1,1]
    chromosome['thrust'] = [-300, -300, -150, -200, np.random.uniform(-200,50), 50, -5, np.random.uniform(-5,5), 5, 150, 300, 300, 50, np.random.uniform(50,200), 200]

    return chromosome

def generate_static_chromosome():
    """
    Return a hardcoded chromosome to compare the GA-optimized fuzzy system with
    """
    chromosome = {}

    # generate fuzzy sets for antecedents
    # values hardcoded for a first testing
    # after successful testing -> generate random values
    chromosome['bullet_time'] = [0,0,0.05,0,0.07,0.15,0.0,0.2]
    chromosome['theta_delta'] = [-1*math.pi, -5/9 * math.pi, -3/4 * math.pi, -1/2*math.pi, -1/4 * math.pi, -1/2 * math.pi, -1/3 * math.pi,0, -1/180 * math.pi, 0, 1/180 * math.pi, 0, 1/4 * math.pi, 1/2 * math.pi, 1/4 * math.pi, 1/2 * math.pi, 3/4 * math.pi, 5/9 * math.pi, math.pi]
    chromosome['asteroidDistance'] = [0, 0, 200, 100, 150, 200, 200, 350]
    chromosome['asteroidSpeed'] = [0, 50, 30, 70]
    chromosome['currVelocity'] = [-300, -290, -100, -150, -70, 5, -5, 0, 5, 100, 250, 270, 5, 120, 200]

    # generate fuzzy sets for consequents
    chromosome['ship_turn'] = [-180, -100, -135, -120, -45, -90, -60, 0, -1, 0, 1, 0, 60, 90, 45, 120, 135, 100, 180]
    chromosome['ship_fire'] = [-1,-1,0.0,0.0,1,1]
    chromosome['thrust'] = [-300, -300, -150, -200, -100, 50, -5, 0, 5, 150, 300, 300, 50, 100, 200]

    return chromosome

def displayCompareWithStatic(best_fitness):
    chromosome = generate_static_chromosome()

    staticFitness =  evaluate_fitness(chromosome, "test")
    print("==================== Summary ================================")
    print(f"Fitness of an unoptimized fuzzy controller: {staticFitness}")
    print(f"Fitness of a GA-optimized fuzzy controller: {best_fitness} ")
    print("=============================================================")

def main():
    global game
    global scenarios

    # Define game scenario
    my_training_scenario = Scenario(name='Train Scenario',
                                num_asteroids=10,
                                ship_states=[
                                    {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1},
                                    {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2},
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
    scenarios = [my_training_scenario]

    game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

    # init and
    # set parameters of genetic algorithm
    ga = EasyGA.GA()
    ga.gene_impl = lambda: generate_chromosome()
    ga.chromosome_length = 1
    ga.population_size = 20
    ga.target_fitness_type = 'max'
    ga.generation_goal = 4
    ga.fitness_function_impl = evaluate_fitness
    ga.evolve()
    ga.print_best_chromosome()

    # get best chromosome
    best_fitness = ga.database.get_highest_chromosome()
    best_fitness = best_fitness if type(best_fitness) == float else max(best_fitness) # Sometimes best fitness is a list, other times a float

    best_chromosome = ga.database.query_all(
        f"""
        SELECT chromosome
        FROM data
        WHERE fitness = {best_fitness}
        """
    )

    displayCompareWithStatic(best_fitness)

    # Sometimes SQLite returns a list, sometimes not. This is to handle that
    best_chromosome = best_chromosome[0] if type(best_chromosome) == list else best_chromosome

    # create controller with best chromosome
    best_chromosome = best_chromosome.replace("'", "\"")
    best_chromosome = json.loads(best_chromosome)
    BestController = SmartController(best_chromosome[0],'test')

    # run game with the best controller
    final_game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
    pre = time.perf_counter()
    score,perf_data = final_game.run(scenario=my_training_scenario, controllers=[BestController, ScottDickController()])



main()