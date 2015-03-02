from pylearn2.models.mlp import MLP
from pylearn2.models.maxout import Maxout
from pylearn2.training_algorithms.sgd import SGD
import logging
import warnings
import sys

import numpy as np
from theano.compat import six
from theano import config
from theano import function
from theano.gof.op import get_debug_values
import theano.tensor as T

from pylearn2.compat import OrderedDict, first_key
from pylearn2.monitor import Monitor
from pylearn2.space import CompositeSpace, NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor \
        as LRMomentumAdjustor
from pylearn2.utils.iteration import is_stochastic, has_uniform_batch_size
from pylearn2.utils import py_integer_types, py_float_types
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils import sharedX
from pylearn2.utils import contains_nan
from pylearn2.utils import contains_inf
from pylearn2.utils import isfinite
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.timing import log_timing
from pylearn2.utils.rng import make_np_rng


log = logging.getLogger(__name__)

class TestAlgo(SGD):
    # this train function mainly to hack into weight tracking
    def train(self, dataset):
        """
        Runs one epoch of SGD training on the specified dataset.

        Parameters
        ----------
        dataset : Dataset
        """

        self.first = False
        rng = self.rng
        if not is_stochastic(self.train_iteration_mode):
            rng = None

        data_specs = self.cost.get_data_specs(self.model)

        # The iterator should be built from flat data specs, so it returns
        # flat, non-redundent tuples of data.
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)
        if len(space_tuple) == 0:
            # No data will be returned by the iterator, and it is impossible
            # to know the size of the actual batch.
            # It is not decided yet what the right thing to do should be.
            raise NotImplementedError("Unable to train with SGD, because "
                    "the cost does not actually use data from the data set. "
                    "data_specs: %s" % str(data_specs))
        flat_data_specs = (CompositeSpace(space_tuple), source_tuple)

        iterator = dataset.iterator(mode=self.train_iteration_mode,
                batch_size=self.batch_size,
                data_specs=flat_data_specs, return_tuple=True,
                rng = rng, num_batches = self.batches_per_iter)

        """
        if not hasattr(self, 'batch_count'):
            self.batch_count=0
            self.param_records=[]
        print "Going into first batch"
        param_init = self.model.get_param_values()
        """
        

        on_load_batch = self.on_load_batch
        for batch in iterator:
            for callback in on_load_batch:
                callback(*batch)
            self.sgd_update(*batch)
            # iterator might return a smaller batch if dataset size
            # isn't divisible by batch_size
            # Note: if data_specs[0] is a NullSpace, there is no way to know
            # how many examples would actually have been in the batch,
            # since it was empty, so actual_batch_size would be reported as 0.
            actual_batch_size = flat_data_specs[0].np_batch_size(batch)
            self.monitor.report_batch(actual_batch_size)
            for callback in self.update_callbacks:
                callback(self)


            """
            param_first = self.model.get_param_values()
            with log_timing(log, "Saving initial param and first param"):
                serial.save("param_init_first.pkl", (param_init, param_first))
            sys.exit(0)
            # Now, we record the weights every 50 minibatches
            # So 10 records per epoch
            self.batch_count+=1
            if self.batch_count%50==0:
                self.param_records.append(self.model.get_param_values())
                # for every 2 epochs, we save the param_records
                if self.batch_count%(50*20)==0:
                    record_path = './mytest/'+str(self.batch_count)+'.pkl'
                    print "We are now about to same lots of param records"
                    with log_timing(log, 'Saving param records to'+record_path):
                        serial.save(record_path, self.param_records)
                    self.param_records=[]
            """

class SOMaxout(Maxout):
    """
    A SOM-Maxout layer based on Maxout

    Each maxout unit is a group, and units within the same group learn
    "together" by copying each other's update in an SOM-like manner. 
    
    Usually, in a maxout group, if a unit is winning/maxing all the time, the
    other units in its group will never be used, never get updated, and thus get
    stuck forever. This wastes maxout's capacity.

    SOM-Maxout solves this problem by asking units within the same somaxout
    group to be each others' buddies. The winners will help their neighbours to
    learn "together". That is, if the winner gets a delta w, it will ask its
    neighbours to get a SOM_factor * delta w.
    
        decay_rate
    """
    def __init__(self, *args, **kwargs):
        super(SOMaxout, self).__init__(*args, **kwargs)
        matrix_value = 0.5 * np.eye(self.num_pieces) + \
                       0.5 * np.ones([self.num_pieces,self.num_pieces])
        self.SOM_copy_matrix = sharedX(matrix_value)
        print "SOM_copy_matrix established"
        print matrix_value

    def modify_grads(self, grads):
        """
        W is a matrix n-input by n-maxout unit
        The objective of this function is to ask nearby units in the same SOM
        group to learn from each other by asking them to copy each other's
        grads
        [1, 0.8]
        [0.8, 1]
        """
        W, = self.transformer.get_params()
        grad_old = grads[W]
        npi = self.num_pieces
        # within each Maxout unit, we perform a within-group copy of grads.
        # each within-group copy produces an input-size by num_pieces matrix.
        grad_list= [ T.dot(grad_old[:, i*npi:(i+1)*npi ], self.SOM_copy_matrix)
                         for i in xrange(self.num_units)]
        # we then concatenate all those matrices into an input-size by
        # num_units*num_pieces matrix
        grads[W] = T.concatenate(grad_list, axis=1)
        print "Gradients for layer "+self.layer_name+" modified."

    def _modify_updates(self, updates):
        """
        At each update, make sure all units in the same somaxout group has equal
        norm
        """
        pass
