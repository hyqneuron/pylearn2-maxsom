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
    def __init__(self, **kwargs):
        """
        apart from the ordinary arguments of SGD, we accept an
        stage_2: indicate if we are in the "continued" stage of training. If we
        are, modify_grads and modify_updates will choose not to set up SOM.
        """
        self.stage_2 = False
        if 'stage_2' in kwargs:
            self.stage_2 = kwargs['stage_2']
            kwargs.pop('stage_2',None)
        super(TestAlgo, self).__init__(**kwargs)

    def setup(self, model, dataset):
        for layer in model.layers:
            layer.stage_2=self.stage_2
        return super(TestAlgo, self).setup(model, dataset)
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

        # update layers' epoch count and new SOM_copy_matrix
        for layer in self.model.layers:
            layer.set_epoch(self.model.monitor.get_epochs_seen())
            layer.set_SOM_copy_matrix(layer.compute_SOM_copy_matrix())

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
    
    init_SOM_var: initial SOM variance when epoch=0
    final_SOM_var: final SOM variance
    saturate_at_epoch: the epoch at which SOM variance reaches final_SOM_var
    """
    def __init__(self, 
                init_SOM_var=1.5,
                final_SOM_var=1.5,
                saturate_at_epoch=10**8,
                **kwargs):
        super(SOMaxout, self).__init__(**kwargs)
        self.init_SOM_var = init_SOM_var
        self.final_SOM_var = final_SOM_var
        self.saturate_at_epoch = saturate_at_epoch
        self.SOM_var = init_SOM_var
        self.standardize_norm = True

        print "Initializing mytest6."+self.layer_name
        # compute and create SOM_cmatrix
        self.SOM_copy_matrix = sharedX(np.zeros(
            [self.num_pieces, self.num_pieces]), 'SOM_cmatrix_'+self.layer_name)
        self.set_SOM_copy_matrix(self.compute_SOM_copy_matrix())
    def set_epoch(self, epoch): # called by TestAlg.train()
        self.epoch = epoch

    def compute_SOM_copy_matrix(self):
        self.SOM_var = (self.init_SOM_var + (self.final_SOM_var -
            self.init_SOM_var)*(float(self.epoch)/self.saturate_at_epoch))
        matrix_value = np.zeros([self.num_pieces, self.num_pieces])
        for i in range(self.num_pieces):
            for j in range(self.num_pieces):
                matrix_value[i,j] = np.exp( - ((i-j)/self.SOM_var)**2 )
        return matrix_value
    def set_SOM_copy_matrix(self, matrix_value):
        self.SOM_copy_matrix.set_value(matrix_value)
        print "Layer "+self.layer_name+" has received new SOM_cmatrix:"
        self.print_SOM_copy_matrix()
        

    def print_SOM_copy_matrix(self):
        np.set_printoptions(precision=4,linewidth=100)
        if not hasattr(self, "SOM_copy_matrix"):
            print self.layer_name+" has no som cmatrix"
        else:
            print "Epoch={}, SOM_var={}".format(self.epoch, self.SOM_var)
            print self.SOM_copy_matrix.get_value()

    def modify_grads(self, grads):
        """
        W is a matrix n-input by n-maxout unit
        The objective of this function is to ask nearby units in the same SOM
        group to learn from each other by asking them to copy each other's
        grads
        [1, 0.8]
        [0.8, 1]
        """
        if self.stage_2:
            print "Gradient left untouched"
            return
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
        if self.stage_2:
            print "Norm Standardization cancelled"
            return
        W, = self.transformer.get_params()
        update_old = updates[W]
        npi = self.num_pieces
        if self.standardize_norm:
            norms = T.sqrt(T.sum(T.sqr(update_old), axis=0))
            norm_mean = norms.reshape([self.num_units, self.num_pieces]).mean(axis=1)
            norm_desired=T.repeat(norm_mean, npi)
            if self.max_col_norm is not None:
                norm_desired = T.clip(norm_desired, 0, self.max_col_norm)
            updates[W] = update_old * norm_desired / norms 
            print "Updates for layer "+self.layer_name+" modified with within-group norm standardization"

