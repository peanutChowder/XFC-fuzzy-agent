# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.
import time
from statistics import mean
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from examples.test_controller import TestController
from examples.scott_dick_controller import ScottDickController
from examples.smart_controller import SmartController
from examples.graphics_both import GraphicsBoth


def runSimulation(numRuns):
    team1Scores = {"hits": [], "deaths": [], "accuracy": []}
    team2Scores= {"hits": [], "deaths": [], "accuracy": []}
    for i in range(numRuns):

        # game = KesslerGame(settings=game_settings) # Use this to visualize the game scenario
        game = TrainerEnvironment(settings=game_settings) # Use this for max-speed, no-graphics simulation
        pre = time.perf_counter()
        score, perf_data = game.run(scenario=my_test_scenario, controllers = [SmartController(), ScottDickController()])

        hits = [team.asteroids_hit for team in score.teams]
        deaths = [team.deaths for team in score.teams]
        accuracy = [team.accuracy for team in score.teams]

        print(f"============== Game {i + 1} =================")
        print('Scenario eval time: '+str(time.perf_counter()-pre))
        print(score.stop_reason)
        print('Asteroids hit: ' + str(hits))
        print('Deaths: ' + str(deaths))
        print('Accuracy: ' + str(accuracy))
        print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
        # print('Evaluated frames: ' + str([controller.eval_frames for controller in score.final_controllers]))

        team1Scores["hits"].append(hits[0])
        team1Scores["deaths"].append(deaths[0])
        team1Scores["accuracy"].append(deaths[0])
        team2Scores["hits"].append(hits[1])
        team2Scores["deaths"].append(deaths[1])
        team2Scores["accuracy"].append(deaths[1])




    print("Final Stats")
    print(f"Avg Hits:\t\t {mean(team1Scores['hits'])}\t\t{mean(team2Scores['hits'])}")
    print(f"Avg Deaths:\t\t {mean(team1Scores['deaths'])}\t\t{mean(team2Scores['deaths'])}")
    print(f"Avg Accuracy:\t\t {mean(team1Scores['accuracy'])}\t\t{mean(team2Scores['accuracy'])}")


def runVisual():
    game = KesslerGame(settings=game_settings) # Use this to visualize the game scenario
    # game = TrainerEnvironment(settings=game_settings) # Use this for max-speed, no-graphics simulation
    pre = time.perf_counter()
    score, perf_data = game.run(scenario=my_test_scenario, controllers = [SmartController(), TestController()])
    print('Scenario eval time: '+str(time.perf_counter()-pre))
    print(score.stop_reason)
    print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
    print('Deaths: ' + str([team.deaths for team in score.teams]))
    print('Accuracy: ' + str([team.accuracy for team in score.teams]))
    print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
    print('Evaluated frames: ' + str([controller.eval_frames for controller in score.final_controllers]))

my_test_scenario = Scenario(name='Test Scenario',
 num_asteroids=10,
ship_states=[
 {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1},
 {'position': (600, 400), 'angle': 90, 'lives': 3, 'team': 2},
 ],
map_size=(1000, 800),
 time_limit=60,
ammo_limit_multiplier=0,
stop_if_no_ammo=False)
game_settings = {'perf_tracker': True,
 'graphics_type': GraphicsType.Tkinter,
 'realtime_multiplier': 3,
 'graphics_obj': None}

runSimulation(10)
