import unittest
import time
from carla_env.modules.server import server

class ServerTest(unittest.TestCase):
	def setUp(self):
		self.server = server.ServerModule(None)

	def test_start(self):
		self.server._start()
		self.assertTrue(self.server._is_running())
		time.sleep(5.0)
		self.server._stop()
		
	def test_stop(self):
		self.server._start()
		time.sleep(5.0)
		self.server._stop()
		time.sleep(1.0)
		self.assertFalse(self.server._is_running())


