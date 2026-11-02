import pygame
import math
from Colors import Color


class Action:
    TURN_LEFT = 0
    TURN_RIGHT = 1
    ACCELERATE = 2
    BRAKE = 3


class Car:
    def __init__(self, _STARTING_POINT, _STARTING_ANGLE, _nn, _computeNetwork, _COLLISIONS_MASK, car_dim,
                 _total_width, _total_height, _segment_width=None, _sensor_step=1):

        self.CAR_WIDTH = car_dim[0]
        self.CAR_HEIGHT = car_dim[1]

        self.MINIMUM_SPEED = 0
        self.TURN_SPEED = 3
        self.ACCELERATION = 2

        # Sensor samples every SENSOR_STEP pixels.  Set once in main.py and passed
        # in by EvolutionEngine (the same value goes to NumbaSim / BatchedSim).
        self.SENSOR_STEP = _sensor_step
        self.MAX_SPEED = 4

        self.COLLISION_SURFACE_COLOR = Color.GREEN
        self.DRAW_SENSORS = False
        self.SENSORS_DRAW_DISTANCE = 5000
        self.SENSOR_ANGLES = 25
        self.CAR_SPRITE_PATH = "assets/"

        self.STARTING_POINT = _STARTING_POINT
        self.STARTING_ANGLE = _STARTING_ANGLE
        self.TOTAL_WIDTH = _total_width
        self.TOTAL_HEIGHT = _total_height
        self.nn = _nn
        self.computeNetwork = _computeNetwork
        self.COLLISIONS_MASK = _COLLISIONS_MASK

        # Width of one track segment, used for sensor normalisation and fitness
        self.SEGMENT_WIDTH = _segment_width if _segment_width is not None else _total_width

        self.baseSprite = pygame.image.load(self.CAR_SPRITE_PATH + "car_sprite.png").convert_alpha()
        self.baseSprite = pygame.transform.scale(self.baseSprite, (self.CAR_WIDTH, self.CAR_HEIGHT))

        self.deadSprite = pygame.image.load(self.CAR_SPRITE_PATH + "car_dead_sprite.png").convert_alpha()
        self.deadSprite = pygame.transform.scale(self.deadSprite, (self.CAR_WIDTH, self.CAR_HEIGHT))

        self.sprite = self.baseSprite.copy()

        self.fitness = 0
        self.speed = self.MINIMUM_SPEED
        self.position = self.STARTING_POINT.copy()
        self.angle = self.STARTING_ANGLE

        self.sensorValues = []
        self.sensorHits = []
        self.inputVector = []

        self.alive = True

        self.startX = self.STARTING_POINT[0]
        self.maxXProgress = self.startX
        self.xAtLastReset = self.startX

        self.deadFlag = False
        self.lastFitnessProgress = 0
        self.lastFastSpeed = 0

        # Minimum fitness gain (in fitness units) required within the no-progress
        # window to keep the car alive.  Prevents cars from resetting the timer by
        # tiny oscillations without actually advancing.
        # 1 fitness unit = SEGMENT_WIDTH / 10 pixels of forward progress.
        self.MIN_FITNESS_PROGRESS = 0.1

    def reset(self):
        self.sprite = self.baseSprite.copy()
        self.fitness = 0
        self.speed = self.MINIMUM_SPEED
        self.position = self.STARTING_POINT.copy()
        self.angle = self.STARTING_ANGLE
        self.sensorValues = []
        self.sensorHits = []
        self.alive = True
        self.maxXProgress = self.startX
        self.xAtLastReset = self.startX
        self.deadFlag = False
        self.lastFitnessProgress = 0
        self.lastFastSpeed = 0
        self.deadSprite = pygame.image.load(self.CAR_SPRITE_PATH + "car_dead_sprite.png").convert_alpha()
        self.deadSprite = pygame.transform.scale(self.deadSprite, (self.CAR_WIDTH, self.CAR_HEIGHT))

    def updateSensors(self):
        collision_map = self.COLLISIONS_MASK
        step = self.SENSOR_STEP
        map_w = collision_map.shape[0]
        map_h = collision_map.shape[1]
        self.sensorValues.clear()
        self.sensorHits.clear()
        for i in range(0, 5):
            sensorAngle = self.angle + (i - 2) * self.SENSOR_ANGLES
            radians = math.radians((360 - sensorAngle))
            xStep = math.cos(radians) * step
            yStep = math.sin(radians) * step
            length = 0

            sensorX: float = self.position[0]
            sensorY: float = self.position[1]

            while True:
                ix, iy = int(sensorX), int(sensorY)
                if ix < 0 or ix >= map_w or iy < 0 or iy >= map_h:
                    break  # treat out-of-bounds as wall hit
                if collision_map[ix][iy]:
                    break
                sensorX += xStep
                sensorY += yStep
                length += step

            self.sensorValues.append(length)
            self.sensorHits.append([sensorX, sensorY])

    def computeReaction(self):
        self.inputVector.clear()
        self.inputVector.extend([v / self.SEGMENT_WIDTH for v in self.sensorValues])
        self.inputVector.append(self.speed / self.MAX_SPEED)    # speed normalised to 0..1

        output = self.computeNetwork(self.nn, self.inputVector)

        def normalize_pair(a, b):
            if a > 0 and b > 0:
                s = a + b
                return a / s, b / s
            elif a < 0 and b < 0:
                a *= -1
                b *= -1
                (a, b) = (b, a)
                s = a + b
                return a / s, b / s

            abimn = min(a, b)
            a -= abimn
            b -= abimn
            s = a + b
            return (a / s, b / s) if s != 0 else (0, 0)

        tl, tr = normalize_pair(output[Action.TURN_LEFT], output[Action.TURN_RIGHT])
        acc, brk = normalize_pair(output[Action.ACCELERATE], output[Action.BRAKE])

        self.turn_left(tl)
        self.turn_right(tr)
        self.accelerate(acc)
        self.brake(brk)

    def updatePosition(self):
        radians = math.radians(360 - self.angle)
        self.position[0] += math.cos(radians) * self.speed
        self.position[1] += math.sin(radians) * self.speed

    def checkCollision(self):
        self.sprite = pygame.transform.rotate(self.baseSprite, self.angle)

        if (self.position[0] < 0 or self.position[0] >= self.TOTAL_WIDTH or
                self.position[1] < 0 or self.position[1] >= self.TOTAL_HEIGHT):
            self.alive = False
            return

        col_map = self.COLLISIONS_MASK
        map_w, map_h = col_map.shape[0], col_map.shape[1]

        hw, hh = self.CAR_WIDTH / 2, self.CAR_HEIGHT / 2
        rad = math.radians(360 - self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        cx, cy = self.position[0], self.position[1]

        for dx, dy in [(0, 0), (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            px = cx + dx * cos_a - dy * sin_a
            py = cy + dx * sin_a + dy * cos_a
            ix, iy = int(px), int(py)
            if 0 <= ix < map_w and 0 <= iy < map_h:
                if col_map[ix, iy]:
                    self.alive = False
                    return
            else:
                self.alive = False
                return

    def updateFitness(self, time):
        newX = self.position[0]

        if newX > self.maxXProgress:
            self.maxXProgress = newX
            gained = (self.maxXProgress - self.xAtLastReset) / self.SEGMENT_WIDTH * 10
            if gained >= self.MIN_FITNESS_PROGRESS:
                self.lastFitnessProgress = time
                self.xAtLastReset = self.maxXProgress

        if time - self.lastFitnessProgress > 12000:
            self.alive = False
            return

        total_track_width = self.TOTAL_WIDTH
        progress = (self.maxXProgress - self.startX) / self.SEGMENT_WIDTH * 10

        if self.maxXProgress >= total_track_width - self.SEGMENT_WIDTH * 0.1:
            # Completed the full assembled track
            self.alive = False
            elapsed_seconds = time / 120
            self.fitness = progress + 50000.0 / (elapsed_seconds + 1)
        else:
            self.fitness = progress

        if self.speed > 0.001:
            self.lastFastSpeed = time
        if time - self.lastFastSpeed > 1200:
            self.alive = False

    def update(self, time):
        self.updateSensors()
        self.computeReaction()
        self.updatePosition()
        self.checkCollision()
        if self.alive:
            self.updateFitness(time)

    def draw(self, screen, camera_x=0.0) -> None:
        screen_x = self.position[0] - camera_x

        if self.alive:
            screen.blit(self.sprite, (screen_x - self.sprite.get_width() / 2,
                                      self.position[1] - self.sprite.get_height() / 2))
        else:
            if not self.deadFlag:
                self.deadFlag = True
                self.deadSprite = pygame.transform.rotate(self.deadSprite, self.angle)
            screen.blit(self.deadSprite, (screen_x - self.deadSprite.get_width() / 2,
                                          self.position[1] - self.deadSprite.get_height() / 2))

        if self.DRAW_SENSORS and self.alive:
            for position in self.sensorHits:
                pygame.draw.line(screen, Color.WHITE, (screen_x, self.position[1]),
                                 (position[0] - camera_x, position[1]), 2)
                pygame.draw.circle(screen, Color.MAGENTA,
                                   (int(position[0] - camera_x), int(position[1])), 4)

    def accelerate(self, force):
        self.speed += self.ACCELERATION * force
        if self.speed > self.MAX_SPEED:
            self.speed = self.MAX_SPEED

    def brake(self, force):
        self.speed -= self.ACCELERATION * force
        if self.speed < self.MINIMUM_SPEED:
            self.speed = self.MINIMUM_SPEED

    def turn_left(self, force):
        self.angle += self.TURN_SPEED * force

    def turn_right(self, force):
        self.angle -= self.TURN_SPEED * force
