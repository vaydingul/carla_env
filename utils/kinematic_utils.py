
def throttle_brake_to_acceleration(throttle, brake):
	"""Convert throttle and brake to acceleration."""

	return throttle - brake


def acceleration_to_throttle_brake(acceleration):
	"""Convert acceleration to throttle and brake."""

	throttle = max(0, acceleration)
	brake = max(0, -acceleration)
	return throttle, brake
