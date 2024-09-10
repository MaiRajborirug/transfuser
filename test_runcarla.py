from __future__ import print_function

import carla
import random
# import time



import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
# import carla
import signal

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

# from leaderboard.scenarios.scenario_manager_local import ScenarioManager
# from leaderboard.scenarios.route_scenario_local import RouteScenario
# from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
# from leaderboard.autoagents.agent_wrapper_local import  AgentWrapper, AgentError
# from leaderboard.utils.statistics_manager_local import StatisticsManager
# from leaderboard.utils.route_indexer import RouteIndexer

# NOTE: record times
import time
import logging
import os

def main():
    try:
        # Connect to the CARLA server
        client = carla.Client('localhost', 2000)  # Replace 'localhost' with the IP of your CARLA server if needed
        client.set_timeout(10.0)  # Timeout for server connection

        # Load a specific town (e.g., Town02)
        world = client.load_world('Town02')  # Change this to other maps like 'Town01', 'Town03', etc.

        # Get the blueprint library, which contains all vehicle models
        blueprint_library = world.get_blueprint_library()

        # Find a vehicle blueprint, e.g., a Tesla Model 3
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

        # Get a random spawn point from the map
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        # Spawn the vehicle at the selected spawn point
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        print(f'Created {vehicle.type_id} at {spawn_point.location}')

        # Move the vehicle forward for 5 seconds
        vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
        time.sleep(5)

        # Stop the vehicle
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(2)

    finally:
        # Clean up by destroying the vehicle when the script ends
        if vehicle is not None:
            print("Destroying vehicle")
            vehicle.destroy()

if __name__ == '__main__':
    main()