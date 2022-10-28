import unittest
from unittest import TestCase

import model  # noqa: F401,F403


class TestMutable(TestCase):

    def test_use_masternet(self):
        config_str = "SuperConvK3BNRELU(3,32,2,1)SuperResK3K3(32,64,2,32,1)\
                      SuperResK3K3(64,128,2,64,1)SuperResK3K3(128,256,2,128,1)\
                      SuperResK3K3(256,512,2,256,1)SuperConvK1BNRELU(256,512,1,1)"
        
