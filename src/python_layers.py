from __future__ import print_function
from util import addCaffePath
addCaffePath()
import caffe
import numpy as np

# Helper class that allows python layers to be added to NetSpec easily. Just use
#    Py.YourLayer(bottom1, bottom2, ..., parameter1=.., ...)
# parameter1 will automatically be passed to YourLayer defined below
class PY:
	def _parse_kwargs(self, layer, kwargs):
		l = getattr(self.py_module, layer)
		if not 'param_str' in kwargs:
			py_args = {}
			for a in list(kwargs.keys()):
				if hasattr(l, a):
					py_args[a] = kwargs.pop(a)
			kwargs['param_str'] = str(py_args)
		if hasattr(l, 'N_TOP'):
			kwargs['ntop'] = l.N_TOP
		return kwargs


	def __init__(self, module):
		import importlib
		self.module = module
		self.py_module = importlib.import_module(module)

	def __getattr__(self, name):
		return lambda *args, **kwargs: caffe.layers.Python(*args, module=self.module, layer=name, **self._parse_kwargs(name, kwargs))
Py = PY('python_layers')


class PyLayer(caffe.Layer):
	def setup(self, bottom, top):
		if self.param_str:
			params = eval(self.param_str)
			if isinstance(params, dict):
				for p,v in params.items():
					setattr(self, p, v)


class SigmoidCrossEntropyLoss(PyLayer):
	ignore_label = None
	def reshape(self, bottom, top):
		assert len(bottom) == 2
		assert len(top) == 1
		top[0].reshape(1)
	
	def forward(self, bottom, top):
		f, df, t = bottom[0].data, bottom[0].diff, bottom[1].data
		mask = (t != self.ignore_label)
		lZ  = np.log(1+np.exp(-np.abs(f))) * mask
		dlZ = np.exp(np.minimum(f,0))/(np.exp(np.minimum(f,0))+np.exp(-np.maximum(f,0))) * mask
		top[0].data[0] = np.mean(lZ + ((f>0)-t)*f * mask)
		df[...] = (dlZ - t*mask) / lZ.size
	
	def backward(self, top, prop, bottom):
		bottom[0].diff[...] *= top[0].diff[0]

class Print(PyLayer):
	def reshape(self, bottom, top):
		pass
	
	def forward(self, bottom, top):
		print( bottom[0].data )
	
	def backward(self, top, prop, bottom):
		pass
