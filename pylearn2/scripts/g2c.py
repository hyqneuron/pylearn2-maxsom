#!/usr/bin/env python
"""
Script to take a GPU model and convert it to CPU model, so that the 
model can be unpickled and examined on a non-GPU machine.
Obtained from: https://groups.google.com/forum/#!msg/theano-users/el74vKWgBPA/jjCY487lEtMJ
"""

import os
import sys
import numpy as np
from pylearn2.utils import serial
import pylearn2.config.yaml_parse as yaml_parse

class FakeChannel(object):
	def __init__(self, channel):
		self.val_record = np.asarray(channel.val_record)
		self.example_record=np.asarray(channel.example_record)
		self.batch_record=np.asarray(channel.batch_record)
		self.epoch_record=np.asarray(channel.epoch_record)
		self.time_record =np.asarray(channel.time_record)

class FakeMonitor(object):
	def __init__(self, monitor):
		channels = {}
		for channel in monitor.channels:
			channels[channel] = FakeChannel(monitor.channels[channel])
		self.channels = channels
			


if __name__=="__main__":
	_, in_path, out_path = sys.argv
	os.environ['THEANO_FLAGS']="device=cpu"
  
	model = serial.load(in_path)
 
	model2 = yaml_parse.load(model.yaml_src)
	model2.set_param_values(model.get_param_values())
	model2.dataset_yaml_src = model.dataset_yaml_src
		
	model2.monitor = FakeMonitor(model.monitor)
 
	serial.save(out_path, model2)
