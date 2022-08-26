import unittest
import time
from carla_env.modules.client import client
from carla_env.modules.server import server


class ClientTest(unittest.TestCase):
	def setUp(self):
		self.server = server.ServerModule(None)
		self.server._start()
		time.sleep(5.0)
		self.client = client.ClientModule(None)

	def test_start(self):
		self.client._start()
		self.assertFalse(self.client.settings.synchronous_mode)
		self.assertEqual(self.client.settings.fixed_delta_seconds, 0.1)
		time.sleep(10.0)
		self.server._stop()
