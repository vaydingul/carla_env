from xml.sax.handler import DTDHandler
import numpy as np

AIR_DENSITY = 1.225  # kg/m^3
GRAVITY = 9.81  # m/s^2

# LATERAL DYNAMICS PROPERTIES
CORNERING_STIFFNESS_FRONT = 35000 
CORNERING_STIFFNESS_REAR = 40000
MAX_STEER_ANGLE = np.deg2rad(70)  # radians

# INERTIAL PROPERTIES
MASS = 2404 # OK
INERTIA = 5000

# LONGITUDINAL DYNAMICS PROPERTIES
AERODYNAMIC_DRAG_COEFFICIENT = 0.3 # OK
ROLLING_DRAG_COEFFICIENT = 0.02


WHEELBASE = 2.87
WHEELBASE_FRONT = WHEELBASE * 0.5
WHEELBASE_REAR = WHEELBASE * 0.5

class KinematicBicycleModel(object):
	"""
	Define a kinematic bicycle model
	"""
	def __init__(self, x = 0, y = 0, yaw = 0, v = 0):
		
		self.x = x
		self.y = y
		self.yaw = yaw
		self.v = v
		

	def step(self, acceleration, steer, dt):
		"""
		Update the state of the vehicle.
		"""

		delta = steer * MAX_STEER_ANGLE #* np.pi * 0.5
		#delta = np.clip(delta, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)


		self.x += self.v * np.cos(self.yaw) * dt 
		self.y += self.v * np.sin(self.yaw) * dt
		self.yaw += self.v / WHEELBASE * np.tan(delta) * dt
		self.v += acceleration * dt

		return self.x, self.y, self.yaw, self.v

class DynamicBicycleModel(object):
	"""
	Define a dynamic bicycle model
	"""
	def __init__(self, x = 0, y = 0, yaw = 0, velocity_x = 0, velocity_y = 0, omega = 0):
		
		self.x = x
		self.y = y
		self.yaw = yaw
		self.velocity_x = velocity_x
		self.velocity_y = velocity_y
		self.omega = omega
		

		
	

	def _lateral_dynamics_step(self, steer, dt):
		"""
		Update the lateral dynamics of the vehicle.
		"""
		# TODO : Implement lateral dynamics!

		self.y += -self.velocity_y * dt		
		self.yaw += self.omega * dt

		if np.abs(self.velocity_x) > 1e-2:
			
			_delta_vf = (self.velocity_y + WHEELBASE_FRONT * self.omega) / (self.velocity_x)
			_delta_vr = (self.velocity_y - WHEELBASE_REAR * self.omega) / (self.velocity_x)

		else:

			_delta_vf = 0
			_delta_vr = 0

		_F_yf = 2 * CORNERING_STIFFNESS_FRONT * (steer - _delta_vf)
		_F_yr = 2 * CORNERING_STIFFNESS_REAR * (-_delta_vr)

		self.velocity_y += -self.velocity_x * self.omega + (1 / MASS) * (_F_yf + _F_yr) * dt
		self.omega += (1 / INERTIA) * (WHEELBASE_FRONT * _F_yf - WHEELBASE_REAR * _F_yr) * dt



	def _longitudinal_dynamics_step(self, acceleration_x, dt):
		"""
		Update the longitudinal dynamics of the vehicle.
		"""
		# TODO : Implement longitudinal dynamics!
		self.x += self.velocity_x * dt

		# Aerodynamic drag force
		#_frontal_area = 1.6 + 0.00056 * (MASS - 765)
		#_aerodynamic_drag_force = 0.5 * AIR_DENSITY * _frontal_area * AERODYNAMIC_DRAG_COEFFICIENT * self.velocity_x ** 2

		# Longitudinal tire force
	
		self.velocity_x += acceleration_x * dt


	def step(self, acceleration_x, steer, dt):
		"""
		Update the state of the vehicle.
		"""
		steer = MAX_STEER_ANGLE * steer
		self._lateral_dynamics_step(steer, dt)
		self._longitudinal_dynamics_step(acceleration_x, dt)
	

	# def step(self, acceleration, steer, dt):
	# 	"""
	# 	Update the state of the vehicle.
	# 	"""
	# 	delta = steer * MAX_STEER_ANGLE

	# 	self.x = self.x + self.velocity_x * np.cos(self.yaw) * dt - self.velocity_y * np.sin(self.yaw) * dt
	# 	self.y = self.y + self.velocity_x * np.sin(self.yaw) * dt + self.velocity_y * np.cos(self.yaw) * dt
	# 	self.yaw = self.yaw + self.omega * dt
	# 	self.yaw = normalize_angle(self.yaw)

	# 	if self.velocity_x != 0:
	# 		Ffy = -CORNERING_STIFFNESS_FRONT * np.arctan2(((self.velocity_y + WHEELBASE_FRONT * self.omega) / self.velocity_x - delta), 1.0)
	# 		Fry = -CORNERING_STIFFNESS_REAR * np.arctan2((self.velocity_y - WHEELBASE_REAR * self.omega) / self.velocity_x, 1.0)
	# 	else:
	# 		Ffy = 0
	# 		Fry = 0

	# 	R_x = ROLLING_DRAG_COEFFICIENT * self.velocity_x
	# 	F_aero = AERODYNAMIC_DRAG_COEFFICIENT * self.velocity_x ** 2
	# 	F_load = F_aero + R_x


	# 	self.velocity_x = self.velocity_x + (acceleration - Ffy * np.sin(delta) / MASS - F_load/MASS + self.velocity_y * self.omega) * dt
	# 	self.velocity_y = self.velocity_y + (Fry / MASS + Ffy * np.cos(delta) / MASS - self.velocity_x * self.omega) * dt
	# 	self.omega = self.omega + (Ffy * WHEELBASE_FRONT * np.cos(delta) - Fry * WHEELBASE_REAR) / INERTIA * dt	
	

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle