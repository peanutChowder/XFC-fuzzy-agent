# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt




class SmartController(KesslerController):
    
    
        
    def __init__(self):
        self.eval_frames = 0 #What is this?
        self.asteroids = []

        # self.targetingControl is the targeting rulebase, which is static in this controller.      
        # Declare variables

        self.targetingControl = None
        self.thrustControl = None

        self.initTargetControl()
        self.initMoveControl()

    def initTargetControl(self):
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi,math.pi,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)
        
        #Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/3,-1*math.pi/6)
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/3,-1*math.pi/6,0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/6,0,math.pi/6])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0,math.pi/6,math.pi/3])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,math.pi/6,math.pi/3)
        
        #Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-70,-30,5])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-10,0,10])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [5,30,70])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [60,180,180])
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 
                
        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['N']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PL'], ship_fire['N']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))   
        rule6 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NL'], ship_fire['Y']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))    
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PL'], ship_fire['Y']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule11 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule12 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NL'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule14 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PL'], ship_fire['Y']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
     
        #DEBUG
        #bullet_time.view()
        #theta_delta.view()
        #ship_turn.view()
        #ship_fire.view()
     
     
        
        # Declare the fuzzy controller, add the rules 
        # This is an instance variable, and thus available for other methods in the same object. See notes.                         
        # self.targetingControl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
             
        self.targetingControl = ctrl.ControlSystem()
        self.targetingControl.addrule(rule1)
        self.targetingControl.addrule(rule2)
        self.targetingControl.addrule(rule3)
        self.targetingControl.addrule(rule4)
        self.targetingControl.addrule(rule5)
        self.targetingControl.addrule(rule6)
        self.targetingControl.addrule(rule7)
        self.targetingControl.addrule(rule8)
        self.targetingControl.addrule(rule9)
        self.targetingControl.addrule(rule10)
        self.targetingControl.addrule(rule11)
        self.targetingControl.addrule(rule12)
        self.targetingControl.addrule(rule13)
        self.targetingControl.addrule(rule14)
        self.targetingControl.addrule(rule15)

    def initMoveControl(self):
        nearestAsteroidDistance = ctrl.Antecedent(np.arange(0, 1000, 2), "asteroid_distance")
        currVelocity = ctrl.Antecedent(np.arange(-300, 300, 1), 'curr_velocity')
        thrust = ctrl.Consequent(np.arange(-300, 300, 1), 'ship_thrust')

        # C = close, M = medium, F = far
        nearestAsteroidDistance["C"] = fuzz.trimf(nearestAsteroidDistance.universe, [0, 0, 200])
        nearestAsteroidDistance["M"] = fuzz.trimf(nearestAsteroidDistance.universe, [100, 150, 300])
        nearestAsteroidDistance["F"] = fuzz.smf(nearestAsteroidDistance.universe, 200, 350)
        # nearestAsteroidDistance["F"] = fuzz.trimf(nearestAsteroidDistance.universe, [200, 1000, 1000])

        # first letter: F = forwards, R = reverse
        # Second letter is F = Fast, S = Slow
        # St = stationary
        currVelocity["RF"] = fuzz.trimf(currVelocity.universe, [-300, -250, -120])
        currVelocity["RS"] = fuzz.trimf(currVelocity.universe, [-150, -100, 0])
        currVelocity["St"] = fuzz.trimf(currVelocity.universe, [-50, 0, 50])
        currVelocity["FF"] = fuzz.trimf(currVelocity.universe, [120, 250, 300])
        currVelocity["FS"] = fuzz.trimf(currVelocity.universe, [0, 100, 150])

        # first letter: F = forwards, R = reverse
        # Second letter is F = Fast, S = Slow
        # St = stationary
        thrust["RF"] = fuzz.trimf(thrust.universe, [-300, -300, -70])
        thrust["RS"] = fuzz.trimf(thrust.universe, [-200, -100, 50])
        thrust["St"] = fuzz.trimf(thrust.universe, [-30, 0, 30])
        thrust["FF"] = fuzz.trimf(thrust.universe, [200, 300, 300])
        thrust["FS"] = fuzz.trimf(thrust.universe, [50, 100, 200])

        rule1 = ctrl.Rule(nearestAsteroidDistance["C"] & currVelocity['RF'], thrust["St"])
        rule2 = ctrl.Rule(nearestAsteroidDistance["C"] & currVelocity['RS'], thrust["RS"])
        rule3 = ctrl.Rule(nearestAsteroidDistance["C"] & currVelocity["St"], thrust["RF"])
        rule4 = ctrl.Rule(nearestAsteroidDistance["C"] & (currVelocity["FF"] | currVelocity["FS"]), thrust["RF"])

        rule5 = ctrl.Rule(nearestAsteroidDistance["M"] & currVelocity["RF"], thrust["FF"])
        rule6 = ctrl.Rule(nearestAsteroidDistance["M"] & currVelocity["RS"], thrust["FS"])
        rule7 = ctrl.Rule(nearestAsteroidDistance["M"] & currVelocity["St"], thrust["FF"])
        rule8 = ctrl.Rule(nearestAsteroidDistance["M"] & (currVelocity["FF"] | currVelocity["FS"]), thrust["RS"])

        rule9 = ctrl.Rule(nearestAsteroidDistance["F"] & currVelocity["RF"], thrust["FF"])
        rule10 = ctrl.Rule(nearestAsteroidDistance["F"] & currVelocity["RS"], thrust["FF"])
        rule11 = ctrl.Rule(nearestAsteroidDistance["F"] & currVelocity["St"], thrust["FS"])
        rule12 = ctrl.Rule(nearestAsteroidDistance["F"] & currVelocity["FS"], thrust["FS"])
        rule13 = ctrl.Rule(nearestAsteroidDistance["F"] & currVelocity["FF"], thrust["St"])

        self.thrustControl = ctrl.ControlSystem()

        self.thrustControl.addrule(rule1)
        self.thrustControl.addrule(rule2)
        self.thrustControl.addrule(rule3)  
        self.thrustControl.addrule(rule4)  
        self.thrustControl.addrule(rule5)
        self.thrustControl.addrule(rule6)
        self.thrustControl.addrule(rule7)
        self.thrustControl.addrule(rule8)
        self.thrustControl.addrule(rule9)
        self.thrustControl.addrule(rule10)
        self.thrustControl.addrule(rule11)
        self.thrustControl.addrule(rule12)
        self.thrustControl.addrule(rule13)  
        
    def getClosestAsteroid(self, ship_pos_x, ship_pos_y):
        # Find the closest asteroid (disregards asteroid velocity)      
        closestAsteroid = self.asteroids[0]

        closestAsteroid = min(
            self.asteroids,
            key=lambda asteroid: math.sqrt((ship_pos_x - asteroid["position"][0])**2 + (ship_pos_y - asteroid["position"][1])**2)
        )

        return {"aster": closestAsteroid, "dist": math.sqrt((ship_pos_x - closestAsteroid["position"][0])**2 + (ship_pos_y - closestAsteroid["position"][1])**2)}
    
    def getCollidingAsteroids(self, shipX, shipY, shipVelX, shipVelY, maxTime):
        collisionThreshold = 10
        collisionAsteroids = []

        for asteroid in self.asteroids:
            asterX, asterY = asteroid["position"]
            asterVelX, asterVelY = asteroid["velocity"]

            # Calculate time to collision 
            timeToCollision = (asterX - shipX) / (shipVelX - asterVelX)

            # Ignore asteroids that will collide in a long time
            if timeToCollision > maxTime:
                continue;

            # Calculate future positions at the time of collision
            shipFutureX = shipX + shipVelX * timeToCollision
            shipFutureY = shipY + shipVelY * timeToCollision

            asterFutureX = asterX + asterVelX * timeToCollision
            asterFutureY = asterY + asterVelY * timeToCollision

            # Check if the positions intersect
            if abs(shipFutureX - asterFutureX) < collisionThreshold and abs(shipFutureY - asterFutureY) < collisionThreshold:
                collisionAsteroids.append(asteroid)

        return collisionAsteroids


    def getAsteroidsSortedByDistance(self, shipX, shipY):   
        return sorted(
            self.asteroids, 
            key=lambda a: math.sqrt((shipX - a["position"][0])**2 + (shipY - a["position"][1])**2)
        )
    
    def getMaxThreatAsteroid(self, shipX: int, shipY: int, shipVelX: int , shipVelY: int):
        distanceThreshold = 300

        collidingAsteroids = self.getCollidingAsteroids(shipX, shipY, shipVelX, shipVelY, 500)
        sortedAsteroids = self.getAsteroidsSortedByDistance(shipX, shipY)

        for asteroid in sortedAsteroids:
            if asteroid in collidingAsteroids:
                return {
                    "aster": asteroid, 
                    "dist": math.sqrt((shipX - asteroid["position"][0])**2 + (shipY - asteroid["position"][1])**2)
                    }
            
        return {
            "aster": sortedAsteroids[0], 
            "dist": math.sqrt((shipX - sortedAsteroids[0]["position"][0])**2 + (shipY - sortedAsteroids[0]["position"][1])**2)
            }
    
    def getShootingInputs(self, ship_pos_x, ship_pos_y, ship_state, closest_asteroid):
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * bullet_t
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * bullet_t
        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        return bullet_t, shooting_theta
    
    def getRelativeVelocity(self, absVelX, absVelY, shipHeading):
        """Get the velocity as a magnitude relative to where the ship is facing.
        i.e. moving backwards is -1, forwards is +1.

        Args:
            absVelX (int): Absolute x-velocity 
            absVelY (int): Absolute y-velocity
            shipHeading (int): Ship heading in degrees

        Returns:
            relVel (int): velocity with respect to heading
        """
        # Convert heading to radians
        heading_rad = math.radians(shipHeading)
        
        # Calculate the relative velocity component along the heading direction
        relVel = (absVelX * math.cos(heading_rad)) + (absVelY * math.sin(heading_rad))

        return relVel

    def actions(self, ship_state: Dict, game_state: Dict):
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?
        
        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity. 
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.
        

        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation 
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.


        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
        self.asteroids = game_state['asteroids']

        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]  

        biggestAsteroidThreat = self.getClosestAsteroid(ship_pos_x, ship_pos_y)
        bullet_t, shooting_theta = self.getShootingInputs(ship_pos_x, ship_pos_y, ship_state, biggestAsteroidThreat)

        relativeVelocity = self.getRelativeVelocity(ship_state["velocity"][0], ship_state["velocity"][1], ship_state["heading"])
        
        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targetingControl,flush_after_run=1)
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta

        # Pass inputs to movement control
        thrust = ctrl.ControlSystemSimulation(self.thrustControl, flush_after_run=1)
        thrust.input["asteroid_distance"] = biggestAsteroidThreat["dist"]
        thrust.input["curr_velocity"] = relativeVelocity
        
        shooting.compute()
        thrust.compute()
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
               
        # And return your three outputs to the game simulation. Controller algorithm complete.
        applyThrust = thrust.output["ship_thrust"]
        
        self.eval_frames +=1

        # added due to error of missing return argument in kesslergame.py, line 125#
        drop_mine = False
        
        #DEBUG
        # print(thrust, bullet_t, shooting_theta, turn_rate, fire, drop_mine)
        
        return applyThrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Smart Controller"
    
if __name__ == "__main__":
    sc = SmartController()

    sc.initMoveControl()