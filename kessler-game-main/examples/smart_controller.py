# Jacob Feng
# ECE 449

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np




class SmartController(KesslerController):
    
    
        
    def __init__(self, chromosome, mode):
        self.eval_frames = 0 #What is this?
        self.asteroids = []

        # self.targetingControl is the targeting rulebase, which is static in this controller.      
        # Declare variables

        self.targetingControl = None
        self.thrustControl = None
        self.currTargetAsteroid = None

        self.chromosome = chromosome
        self.mode = mode

        self.initTargetControl()
        self.initMoveControl()

    def initTargetControl(self):
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi,math.pi,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        if self.mode == 'test':
            #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
            bullet_time['S'] = fuzz.trimf(bullet_time.universe, self.chromosome['bullet_time'][0:3])
            bullet_time['M'] = fuzz.trimf(bullet_time.universe, self.chromosome['bullet_time'][3:6])
            bullet_time['L'] = fuzz.smf(bullet_time.universe,self.chromosome['bullet_time'][6], self.chromosome['bullet_time'][7])
            
            #Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
            theta_delta['NL'] = fuzz.zmf(theta_delta.universe, self.chromosome['theta_delta'][0], self.chromosome['theta_delta'][1])
            theta_delta['NM'] = fuzz.trimf(theta_delta.universe, self.chromosome['theta_delta'][2:5])
            theta_delta['NS'] = fuzz.trimf(theta_delta.universe, self.chromosome['theta_delta'][5:8])
            theta_delta['Z'] = fuzz.trimf(theta_delta.universe, self.chromosome['theta_delta'][8:11])
            theta_delta['PS'] = fuzz.trimf(theta_delta.universe, self.chromosome['theta_delta'][11:14])
            theta_delta['PM'] = fuzz.trimf(theta_delta.universe, self.chromosome['theta_delta'][14:17])
            theta_delta['PL'] = fuzz.smf(theta_delta.universe,self.chromosome['theta_delta'][17], self.chromosome['theta_delta'][18])
            
            #Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
            ship_turn['NL'] = fuzz.zmf(ship_turn.universe, self.chromosome['ship_turn'][0], self.chromosome['ship_turn'][1])
            ship_turn['NM'] = fuzz.trimf(ship_turn.universe, self.chromosome['ship_turn'][2:5])
            ship_turn['NS'] = fuzz.trimf(ship_turn.universe, self.chromosome['ship_turn'][5:8])
            ship_turn['Z'] = fuzz.trimf(ship_turn.universe, self.chromosome['ship_turn'][8:11])
            ship_turn['PS'] = fuzz.trimf(ship_turn.universe, self.chromosome['ship_turn'][11:14])
            ship_turn['PM'] = fuzz.trimf(ship_turn.universe, self.chromosome['ship_turn'][14:17])
            ship_turn['PL'] = fuzz.smf(ship_turn.universe, self.chromosome['ship_turn'][17], self.chromosome['ship_turn'][18])
            
            #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
            #   and returned as the boolean 'fire'
            ship_fire['N'] = fuzz.trimf(ship_fire.universe, self.chromosome['ship_fire'][0:3])
            ship_fire['Y'] = fuzz.trimf(ship_fire.universe, self.chromosome['ship_fire'][3:6])
        
        if self.mode == 'train':
            #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
            bullet_time['S'] = fuzz.trimf(bullet_time.universe, self.chromosome[0].value['bullet_time'][0:3])
            bullet_time['M'] = fuzz.trimf(bullet_time.universe, self.chromosome[0].value['bullet_time'][3:6])
            bullet_time['L'] = fuzz.smf(bullet_time.universe,self.chromosome[0].value['bullet_time'][6], self.chromosome[0].value['bullet_time'][7])
            
            #Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
            theta_delta['NL'] = fuzz.zmf(theta_delta.universe, self.chromosome[0].value['theta_delta'][0], self.chromosome[0].value['theta_delta'][1])
            theta_delta['NM'] = fuzz.trimf(theta_delta.universe, self.chromosome[0].value['theta_delta'][2:5])
            theta_delta['NS'] = fuzz.trimf(theta_delta.universe, self.chromosome[0].value['theta_delta'][5:8])
            theta_delta['Z'] = fuzz.trimf(theta_delta.universe, self.chromosome[0].value['theta_delta'][8:11])
            theta_delta['PS'] = fuzz.trimf(theta_delta.universe, self.chromosome[0].value['theta_delta'][11:14])
            theta_delta['PM'] = fuzz.trimf(theta_delta.universe, self.chromosome[0].value['theta_delta'][14:17])
            theta_delta['PL'] = fuzz.smf(theta_delta.universe,self.chromosome[0].value['theta_delta'][17], self.chromosome[0].value['theta_delta'][18])
            
            #Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
            ship_turn['NL'] = fuzz.zmf(ship_turn.universe, self.chromosome[0].value['ship_turn'][0], self.chromosome[0].value['ship_turn'][1])
            ship_turn['NM'] = fuzz.trimf(ship_turn.universe, self.chromosome[0].value['ship_turn'][2:5])
            ship_turn['NS'] = fuzz.trimf(ship_turn.universe, self.chromosome[0].value['ship_turn'][5:8])
            ship_turn['Z'] = fuzz.trimf(ship_turn.universe, self.chromosome[0].value['ship_turn'][8:11])
            ship_turn['PS'] = fuzz.trimf(ship_turn.universe, self.chromosome[0].value['ship_turn'][11:14])
            ship_turn['PM'] = fuzz.trimf(ship_turn.universe, self.chromosome[0].value['ship_turn'][14:17])
            ship_turn['PL'] = fuzz.smf(ship_turn.universe, self.chromosome[0].value['ship_turn'][17], self.chromosome[0].value['ship_turn'][18])
            
            #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
            #   and returned as the boolean 'fire'
            ship_fire['N'] = fuzz.trimf(ship_fire.universe, self.chromosome[0].value['ship_fire'][0:3])
            ship_fire['Y'] = fuzz.trimf(ship_fire.universe, self.chromosome[0].value['ship_fire'][3:6])
        
                
        #Declare each fuzzy rule
        rules = [
            ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])), 

            ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),    
            ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),

            ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),
        ]

             
        self.targetingControl = ctrl.ControlSystem()
        for rule in rules:
            self.targetingControl.addrule(rule)


    def initMoveControl(self):
        asteroidDistance = ctrl.Antecedent(np.arange(0, 1000, 2), "asteroid_distance")
        asteroidSpeed = ctrl.Antecedent(np.arange(0, 300, 1), 'asteroid_speed')
        currVelocity = ctrl.Antecedent(np.arange(-300, 300, 1), 'curr_velocity')
        thrust = ctrl.Consequent(np.arange(-300, 300, 1), 'ship_thrust')

        if self.mode == 'test':
            # C = close, M = medium, F = far
            asteroidDistance["C"] = fuzz.trimf(asteroidDistance.universe, self.chromosome['asteroidDistance'][0:3])
            asteroidDistance["M"] = fuzz.trimf(asteroidDistance.universe, self.chromosome['asteroidDistance'][3:6])
            asteroidDistance["F"] = fuzz.smf(asteroidDistance.universe, self.chromosome['asteroidDistance'][6], self.chromosome['asteroidDistance'][7])

            # LR = Low Risk, HR = High Risk
            asteroidSpeed["LR"] = fuzz.zmf(asteroidSpeed.universe, self.chromosome['asteroidSpeed'][0], self.chromosome['asteroidSpeed'][1])
            asteroidSpeed["HR"] = fuzz.smf(asteroidSpeed.universe, self.chromosome['asteroidSpeed'][2], self.chromosome['asteroidSpeed'][3])

            # first letter: F = forwards, R = reverse
            # Second letter is F = Fast, S = Slow
            # St = stationary
            currVelocity["RF"] = fuzz.trimf(currVelocity.universe, self.chromosome['currVelocity'][0:3])
            currVelocity["RS"] = fuzz.trimf(currVelocity.universe, self.chromosome['currVelocity'][3:6])
            currVelocity["St"] = fuzz.trimf(currVelocity.universe, self.chromosome['currVelocity'][6:9])
            currVelocity["FF"] = fuzz.trimf(currVelocity.universe, self.chromosome['currVelocity'][9:12])
            currVelocity["FS"] = fuzz.trimf(currVelocity.universe, self.chromosome['currVelocity'][12:15])

            # first letter: F = forwards, R = reverse
            # Second letter is F = Fast, S = Slow
            # St = stationary
            thrust["RF"] = fuzz.trimf(thrust.universe, self.chromosome['thrust'][0:3])
            thrust["RS"] = fuzz.trimf(thrust.universe, self.chromosome['thrust'][3:6])
            thrust["St"] = fuzz.trimf(thrust.universe, self.chromosome['thrust'][6:9])
            thrust["FF"] = fuzz.trimf(thrust.universe, self.chromosome['thrust'][9:12])
            thrust["FS"] = fuzz.trimf(thrust.universe, self.chromosome['thrust'][12:15])

        if self.mode == 'train':
            # C = close, M = medium, F = far
            asteroidDistance["C"] = fuzz.trimf(asteroidDistance.universe, self.chromosome[0].value['asteroidDistance'][0:3])
            asteroidDistance["M"] = fuzz.trimf(asteroidDistance.universe, self.chromosome[0].value['asteroidDistance'][3:6])
            asteroidDistance["F"] = fuzz.smf(asteroidDistance.universe, self.chromosome[0].value['asteroidDistance'][6], self.chromosome[0].value['asteroidDistance'][7])

            # LR = Low Risk, HR = High Risk
            asteroidSpeed["LR"] = fuzz.zmf(asteroidSpeed.universe, self.chromosome[0].value['asteroidSpeed'][0], self.chromosome[0].value['asteroidSpeed'][1])
            asteroidSpeed["HR"] = fuzz.smf(asteroidSpeed.universe, self.chromosome[0].value['asteroidSpeed'][2], self.chromosome[0].value['asteroidSpeed'][3])

            # first letter: F = forwards, R = reverse
            # Second letter is F = Fast, S = Slow
            # St = stationary
            currVelocity["RF"] = fuzz.trimf(currVelocity.universe, self.chromosome[0].value['currVelocity'][0:3])
            currVelocity["RS"] = fuzz.trimf(currVelocity.universe, self.chromosome[0].value['currVelocity'][3:6])
            currVelocity["St"] = fuzz.trimf(currVelocity.universe, self.chromosome[0].value['currVelocity'][6:9])
            currVelocity["FF"] = fuzz.trimf(currVelocity.universe, self.chromosome[0].value['currVelocity'][9:12])
            currVelocity["FS"] = fuzz.trimf(currVelocity.universe, self.chromosome[0].value['currVelocity'][12:15])

            # first letter: F = forwards, R = reverse
            # Second letter is F = Fast, S = Slow
            # St = stationary
            thrust["RF"] = fuzz.trimf(thrust.universe, self.chromosome[0].value['thrust'][0:3])
            thrust["RS"] = fuzz.trimf(thrust.universe, self.chromosome[0].value['thrust'][3:6])
            thrust["St"] = fuzz.trimf(thrust.universe, self.chromosome[0].value['thrust'][6:9])
            thrust["FF"] = fuzz.trimf(thrust.universe, self.chromosome[0].value['thrust'][9:12])
            thrust["FS"] = fuzz.trimf(thrust.universe, self.chromosome[0].value['thrust'][12:15])

        rules = [
            ctrl.Rule(asteroidDistance["C"] & currVelocity['RF'], thrust["RF"]),
            ctrl.Rule(asteroidDistance["C"] & currVelocity['RS'], thrust["RF"]),
            ctrl.Rule(asteroidDistance["C"] & currVelocity["St"], thrust["RF"]),
            ctrl.Rule(asteroidDistance["C"] & (currVelocity["FF"] | currVelocity["FS"]), thrust["RF"]),

            ctrl.Rule(asteroidDistance["M"] & currVelocity["RF"], thrust["FF"]),
            ctrl.Rule(asteroidDistance["M"] & currVelocity["RS"], thrust["FS"]),
            ctrl.Rule(asteroidDistance["M"] & currVelocity["St"], thrust["St"]),
            ctrl.Rule(asteroidDistance["M"] & asteroidSpeed["LR"] & currVelocity["FS"] | currVelocity["FF"], thrust["FS"]),
            ctrl.Rule(asteroidDistance["M"] & asteroidSpeed["HR"] & currVelocity["FS"] | currVelocity["FF"], thrust["RF"]),

            ctrl.Rule(asteroidDistance["F"] & currVelocity["RF"], thrust["FF"]),
            ctrl.Rule(asteroidDistance["F"] & currVelocity["RS"], thrust["FF"]),
            ctrl.Rule(asteroidDistance["F"] & currVelocity["St"], thrust["FF"]),
            ctrl.Rule(asteroidDistance["F"] & currVelocity["FS"], thrust["FF"]),
            ctrl.Rule(asteroidDistance["F"] & currVelocity["FF"], thrust["FF"]),
        ]
        
        self.thrustControl = ctrl.ControlSystem()
        for rule in rules:
            self.thrustControl.addrule(rule)
        
    def getClosestAsteroid(self, ship_pos_x, ship_pos_y):
        # Find the closest asteroid (disregards asteroid velocity)      
        closestAsteroid = self.asteroids[0]

        closestAsteroid = min(
            self.asteroids,
            key=lambda asteroid: math.sqrt((ship_pos_x - asteroid["position"][0])**2 + (ship_pos_y - asteroid["position"][1])**2)
        )

        return {"aster": closestAsteroid, "dist": math.sqrt((ship_pos_x - closestAsteroid["position"][0])**2 + (ship_pos_y - closestAsteroid["position"][1])**2)}
    
    def inDanger(self, shipX, shipY, radius):
        for asteroid in self.asteroids:
            asteroidX, asteroidY = asteroid["position"]

            # Calculate the distance between the ship and the current asteroid
            distance_to_ship = math.sqrt((shipX - asteroidX)**2 + (shipY - asteroidY)**2)

            # Check if the asteroid is within the specified radius
            if distance_to_ship <= radius:
                return True  # At least one asteroid is within the danger radius

        return False  # No asteroids are within the danger radius

    
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
    
    def getRelativeAsteroidVelocity(self, shipX, shipY, asteroid):
        # Extract asteroid position and velocity
        asteroidX, asteroidY = asteroid["position"]
        asteroidVelX, asteroidVelY = asteroid["velocity"]

        # Calculate relative velocity components
        relativeVelX = asteroidVelX - (shipX - asteroidX)
        relativeVelY = asteroidVelY - (shipY - asteroidY)

        # Calculate the dot product to determine the direction of the relative velocity
        dotProduct = relativeVelX * (shipX - asteroidX) + relativeVelY * (shipY - asteroidY)

        # Calculate the magnitude of the relative velocity with sign
        if dotProduct < 0:
            # Negative dot product indicates the asteroid is moving away from the ship
            relativeVelocityMagnitude = -((relativeVelX ** 2 + relativeVelY ** 2) ** 0.5)
        else:
            # Positive dot product indicates the asteroid is moving towards the ship
            relativeVelocityMagnitude = (relativeVelX ** 2 + relativeVelY ** 2) ** 0.5

        return relativeVelocityMagnitude

    def actions(self, ship_state: Dict, game_state: Dict):
        
        self.asteroids = game_state['asteroids']

        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]  


        # Only change targets if there is no current target or if other asteroids are within the danger radius
        if not self.currTargetAsteroid or self.currTargetAsteroid["aster"] not in self.asteroids or self.inDanger(ship_pos_x, ship_pos_y, 20):
            self.currTargetAsteroid = self.getClosestAsteroid(ship_pos_x, ship_pos_y)

        bullet_t, shooting_theta = self.getShootingInputs(ship_pos_x, ship_pos_y, ship_state, self.currTargetAsteroid)

        relativeVelocity = self.getRelativeVelocity(ship_state["velocity"][0], ship_state["velocity"][1], ship_state["heading"])
        relativeAsteroidVelocity = self.getRelativeAsteroidVelocity(ship_pos_x, ship_pos_y, self.currTargetAsteroid["aster"])
        
        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targetingControl,flush_after_run=1)
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta

        # Pass inputs to movement control
        thrust = ctrl.ControlSystemSimulation(self.thrustControl, flush_after_run=1)
        thrust.input["asteroid_speed"] = relativeAsteroidVelocity
        thrust.input["asteroid_distance"] = self.currTargetAsteroid["dist"]
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

        # added due to error in kessler_game.py, line 125
        drop_mine = False
        
        return applyThrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Smart Controller"
    
if __name__ == "__main__":
    sc = SmartController()

    sc.initMoveControl()