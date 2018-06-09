from contextlib import contextmanager
import collections
import h5py
import hashlib
import json
import numpy as np
import sys
import os
import os.path
#import tables, warnings; warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

#print(os.path.dirname(os.path.abspath(__file__)))

#curdir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(curdir)

import util

import theano
from theano import tensor

# Global variable scoping utility, modeled after Tensorflow
_curr_active_scope = None

class variable_scope(object):
    def __init__(self, name):
        global _curr_active_scope
        self.name = name
        if self.name is None:
            # This is the root scope
            assert _curr_active_scope is None
            self.parent = None
            self.fullname = ''
        else:
            # This is not the root scope
            assert _curr_active_scope is not None
            assert '/' not in name
            self.parent = _curr_active_scope
            assert self.name not in self.parent.children, 'Scope {} already exists in parent scope {}'.format(self.name, self.parent.fullname)
            self.parent.children[self.name] = self
            self.fullname = self.parent.fullname + '/' + self.name

        self.children = collections.OrderedDict()
        self.vars = collections.OrderedDict()

    def __enter__(self):
        global _curr_active_scope
        _curr_active_scope = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _curr_active_scope
        assert _curr_active_scope == self
        _curr_active_scope = self.parent

    def get_child_variables(self, trainable_only):
        vs = [v for v, trainable in self.vars.itervalues() if (not trainable_only or trainable)]
#        vs = [v for v, trainable in self.vars.values() if (not trainable_only or trainable)]
        for c in self.children.itervalues():
#        for c in self.children.values():
            vs += c.get_child_variables(trainable_only)
        return vs

    # to be called by get_variable
    def _register_variable(self, name, init_value, broadcastable, trainable):
        assert '/' not in name
        assert name not in self.vars, 'Variable name {} already registered in scope {}'.format(name, self.fullname)
        v = theano.shared(value=init_value, name=self.fullname + '/' + name, broadcastable=broadcastable)
        self.vars[name] = (v, trainable)
        return v

_curr_active_scope = variable_scope(None) # this is the root scope
def get_variable(name, init_value, broadcastable=None, trainable=True):
    global _curr_active_scope
    return _curr_active_scope._register_variable(name, init_value, broadcastable, trainable)

def reset_global_scope():
    global _curr_active_scope
    _curr_active_scope = None
    _curr_active_scope = variable_scope(None)


def _hash_name2array(name2array):
    '''
    Hashes a list of (name,array) tuples.
    The hash is invariant to permutations of the list.
    '''
    def hash_array(a):
        return '%.10f,%.10f,%d' % (np.mean(a), np.var(a), np.argmax(a))
    return hashlib.sha1('|'.join('%s %s' for n, h in sorted([(name, hash_array(a)) for name, a in name2array]))).hexdigest()


import abc
class Model(object):
    '''
    A model abstraction. Stores variables and can save/load them to HDF5 files.
    '''
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def varscope(self): pass

    def get_all_variables(self):
        return self.varscope.get_child_variables(trainable_only=False)

    def get_trainable_variables(self):
        return self.varscope.get_child_variables(trainable_only=True)

    def print_trainable_variables(self):
        for v in self.get_trainable_variables():
            util.header('- %s (%d parameters)' % (v.name, v.get_value().size))
        util.header('Total: %d parameters' % (self.get_num_params(),))

    def get_num_params(self):
        return sum(v.get_value().size for v in self.get_trainable_variables())

    ### Utilities for getting/setting flattened parameter vectors ###
    def set_params(self, x):
        # print 'setting param vars:\n{}'.format('\n'.join([v.name for v in self.get_trainable_variables()]))
        assert x.ndim == 1
        pos = 0
        for v in self.get_trainable_variables():
            val = v.get_value()
            s = val.size
            v.set_value(x[pos:pos+s].reshape(val.shape))
            pos += s
        assert pos == x.shape[0]

    def get_params(self):
        return util.flatcat([v.get_value() for v in self.get_trainable_variables()])

    @contextmanager
    def try_params(self, x):
        orig_x = self.get_params()
        self.set_params(x)
        yield
        self.set_params(orig_x)

    # HDF5 saving and loading
    # The hierarchy in the HDF5 file reflects the hierarchy in the Tensorflow graph.

    def savehash(self):
        return _hash_name2array([(v.name, v.get_value()) for v in self.get_all_variables()])

    def save_h5(self, h5file, key, extra_attrs=None):
        with h5py.File(h5file, 'a') as f:
            if key in f:
                util.warn('WARNING: key %s already exists in %s' % (key, h5file))
                dset = f[key]
            else:
                dset = f.create_group(key)

            for v in self.get_all_variables():
                dset[v.name] = v.get_value()

            dset.attrs['hash'] = self.savehash()
            if extra_attrs is not None:
                for k, v in extra_attrs:
                    if k in dset.attrs:
                        util.warn('Warning: attribute %s already exists in %s' % (k, dset.name))
                    dset.attrs[k] = v

    def load_h5(self, h5file, key):
        with h5py.File(h5file, 'r') as f:
            dset = f[key]

            for v in self.get_all_variables():
                assert v.name[0] == '/'; vname = v.name[1:]
                print ('Reading', vname)
                if vname in dset:
                    v.set_value(dset[vname][...])
                elif vname+':0' in dset:
                    # Tensorflow saves variables with :0 appended to the name,
                    # so try this for backwards compatibility
                    v.set_value(dset[vname+':0'][...])
                else:
                    raise RuntimeError('Variable %s not found in %s' % (vname, dset))

            h = self.savehash()
            assert h == dset.attrs['hash'], 'Checkpoint hash %s does not match loaded hash %s' % (dset.attrs['hash'], h)


# Layers for feedforward networks

class Layer(Model):
    @abc.abstractproperty
    def output(self):
        pass

    @abc.abstractproperty
    def output_shape(self):
        '''Shape refers to the shape without the batch axis, which always implicitly goes first'''
        pass


class ReshapeLayer(Layer):
    def __init__(self, input_, new_shape):
        self._output_shape = tuple(new_shape)
        util.header('Reshape(new_shape=%s)' % (str(self._output_shape),))
        with variable_scope(type(self).__name__) as self.__varscope:
            self._output = input_.reshape((-1,)+self._output_shape)
    @property
    def varscope(self): return self.__varscope
    @property
    def output(self): return self._output
    @property
    def output_shape(self): return self._output_shape

class AffineLayer(Layer):
    def __init__(self, input_B_Di, input_shape, output_shape, initializer, sequential_input=False):
        assert len(input_shape) == len(output_shape) == 1
        util.header('Affine(in=%d, out=%d)' % (input_shape[0], output_shape[0]))
        self._output_shape = (output_shape[0],)
#        if type(input_B_Di) == 3:
#            n_steps = input_B_Di.shape[1]
#        else:
#            n_steps = 0
        with variable_scope(type(self).__name__) as self.__varscope:
            if initializer is None:
                # initializer = np.random.randn(input_shape[0], output_shape[0]) * np.sqrt(2./input_shape[0])

                # Glorot/Bengio 2010
                s = np.sqrt(6. / (input_shape[0] + output_shape[0]))
                initializer = np.random.uniform(low=-s, high=s, size=(input_shape[0],output_shape[0]))

            else:
                assert initializer.shape == (input_shape[0], output_shape[0])
            self.W_Di_Do = get_variable('W', initializer.astype(theano.config.floatX))

            if not sequential_input:
                self.b_1_Do = get_variable('b', np.zeros((1, output_shape[0]), dtype=theano.config.floatX), broadcastable=(True,False))
            # add to support sequential model
            else:
                self.b_1_Do = get_variable('b', np.zeros((1, 1, output_shape[0]), dtype=theano.config.floatX), broadcastable=(True,True,False))
#            if n_steps > 0:
#                self.b_1_Do = get_variable('b', np.zeros((1, output_shape[0]), dtype=theano.config.floatX), broadcastable=(True,False))
#            else:
#                self.b_1_Do = get_variable('b', np.zeros((1,1,output_shape[0]), dtype=theano.config.floatX), broadcastable=(True,True,False))

            self._output_B_Do = input_B_Di.dot(self.W_Di_Do) + self.b_1_Do
    @property
    def varscope(self): return self.__varscope
    @property
    def output(self): return self._output_B_Do
    @property
    def output_shape(self): return self._output_shape


class NonlinearityLayer(Layer):
    def __init__(self, input_B_Di, output_shape, func):
        util.header('Nonlinearity(func=%s)' % func)
        self._output_shape = output_shape
        with variable_scope(type(self).__name__) as self.__varscope:
            self._output_B_Do = {
                'relu': tensor.nnet.relu,
                'lrelu': lambda x: tensor.nnet.relu(x, .01),
                'elu': tensor.nnet.elu,
                'tanh': tensor.tanh,
                'sigmoid': tensor.nnet.sigmoid
            }[func](input_B_Di)
    @property
    def varscope(self): return self.__varscope
    @property
    def output(self): return self._output_B_Do
    @property
    def output_shape(self): return self._output_shape


def _check_keys(d, keys, optional):
    s = set(d.keys())
    if not (s == set(keys) or s == set(keys+optional)):
        raise RuntimeError('Got keys %s, but expected keys %s with optional keys %s' % (str(s, str(keys), str(optional))))

def _parse_initializer(layerspec):
    if 'initializer' not in layerspec:
        return None
    initspec = layerspec['initializer']
    raise NotImplementedError('Unknown layer initializer type %s' % initspec['type'])




def _ortho_weight(ndim, mdim):
    nmdim = max(ndim, mdim)
    W = np.random.randn(nmdim, nmdim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)[:ndim, :mdim]


def _xavier_weight(ndim, mdim):
    return np.random.uniform(low=-np.sqrt(6. / (ndim + mdim)), high=np.sqrt(6. / (ndim + mdim)), size=(ndim, mdim)).astype(theano.config.floatX)



def _init_lstm_weights_4(ndim, mdim, init_weight_func, couple_axis=1):
    values = np.concatenate([init_weight_func(ndim, mdim),
        init_weight_func(ndim, mdim),
        init_weight_func(ndim, mdim),
        init_weight_func(ndim, mdim)], axis=couple_axis)
    return values.astype(theano.config.floatX)


""" Add an LSTM Layer for LSTM policy and classifier """
class LSTMLayer(Layer):
    def __init__(self, input_B_T_Di, mask_B_T, hidden_B_Do, cell_B_Do, input_shape, output_shape, initializer):
        assert len(input_shape) == len(output_shape) == 1
        
        util.header('LSTMLayer(in=%d, out=%d)' %(input_shape[0], output_shape[0]))

#        print("input_B_T_Di dtype:", input_B_T_Di.dtype)
#        print("mask_B_T_Di dtype:", mask_B_T.dtype)
       
        self._input_shape = (input_shape[0],)
        self._output_shape = (output_shape[0],)

        self.initializer = initializer

        # record hidden and cell of samples
        self.n_samples = None
        self._sample_hidden = None
        self._sample_cell = None

        with variable_scope(type(self).__name__) as self.__varscope:
            if initializer is None:
                # Glorot/Bengio 2010
                W_initializer = _init_lstm_weights_4(input_shape[0], output_shape[0], init_weight_func=_xavier_weight, couple_axis=1)
                U_initializer = _init_lstm_weights_4(output_shape[0], output_shape[0], init_weight_func=_ortho_weight, couple_axis=1)
                b_initializer = np.zeros((1, 1, 4 * output_shape[0]))

            else:
                assert False, 'Not Impelement different initializer for LSTMLayer'
            self.W_Di_Do = get_variable('W', W_initializer.astype(theano.config.floatX))
            self.U_Do_Do = get_variable('U', U_initializer.astype(theano.config.floatX))
            self.b_1_1_Do = get_variable('b', b_initializer.astype(theano.config.floatX), broadcastable=(True,True,False))


        input_T_B_Di = tensor.transpose(input_B_T_Di, [1,0,2])
        mask_T_B = tensor.transpose(mask_B_T, [1,0])
        input_actvn_T_B_Do = tensor.dot(input_T_B_Di, self.W_Di_Do) + self.b_1_1_Do

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim : (n + 1) * dim]
            return _x[:, n * dim : (n + 1) * dim]

        def _step(m_, x_, h_, c_):
#            print("Debug: _step")
#            print("m_ dtype:", m_.dtype, "x_ dtype:", x_.dtype, "h_ dtype:", h_.dtype, "c_ dtype:", c_.dtype)
            preact = tensor.dot(h_, self.U_Do_Do) + x_
#            print("preact dtype:", preact.dtype)
            i = tensor.nnet.sigmoid(_slice(preact, 0, output_shape[0]))
            f = tensor.nnet.sigmoid(_slice(preact, 1, output_shape[0]))
            o = tensor.nnet.sigmoid(_slice(preact, 2, output_shape[0]))
#            print("i dtype:", i.dtype, "f dtype:", f.dtype, "o dtype:", o.dtype)
            c = tensor.tanh(_slice(preact, 3, output_shape[0]))

#            print("c dtype:", c.dtype)
            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_
            # convert data type to follow theano config
            c = c.astype(theano.config.floatX)
#            print("c dtype:", c.dtype)

            h = o * tensor.tanh(c)
#            print("h dtype:", h.dtype)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_
            # convert data type to follow theano config
            h = h.astype(theano.config.floatX)
#            print("h dtype:", h.dtype)
            return h, c

        rval, updates = theano.scan(_step,
                                    sequences=[mask_T_B, input_actvn_T_B_Do],
                                    outputs_info=[hidden_B_Do, cell_B_Do],
                                    name='layer',
#                                    n_steps=self.n_steps)
                                    )
        _output_T_B_Do = rval[0]
        _cell_T_B_Do = rval[1]
        self._output_B_T_Do = tensor.transpose(_output_T_B_Do, [1,0,2])
        self._cell_B_T_Do = tensor.transpose(_cell_T_B_Do, [1,0,2])
#        self._updates = updates

    def restart_hidden_cell(self, n_samples=None):
#        print("Debug: restart hidden state:")
        if n_samples is None:
            self._sample_hidden.fill(0)
            self._sample_cell.fill(0)
        else:
            self.n_samples = n_samples
            self._sample_hidden = np.zeros((n_samples, self._output_shape[0]), dtype=theano.config.floatX)
            self._sample_cell = np.zeros((n_samples, self._output_shape[0]), dtype=theano.config.floatX)

    
    @property
    def varscope(self): return self.__varascope
    @property
    def output(self): return self._output_B_T_Do
    @property
    def output_shape(self): return self._output_shape
    @property
    def cell(self): return self._cell_B_T_Do
    @property
    def sample_hidden(self): return self._sample_hidden
    @property
    def sample_cell(self): return self._sample_cell
#    @property
#    def updates(self): return self._updates






class FeedforwardNet(Layer):
    def __init__(self, input_B_Di, input_shape, layerspec_json):
        '''
        Args:
            layerspec (string): JSON string describing layers
        '''
        assert len(input_shape) >= 1
        self.input_B_Di = input_B_Di

        layerspec = json.loads(layerspec_json)
#        print 'Debug: layerspec type:', type(layerspec)
#        print 'Debug: layerspec[0] type:', type(layerspec[0])
#        print 'Debug: layerspec[0] keys:', layerspec[0].keys()
#        print 'Debug: layerspec[0][n] type:', type(layerspec[0]['n'])
        util.header('Loading feedforward net specification')
        
        print (json.dumps(layerspec, indent=2, separators=(',', ': ')))

        self.layers = []
        with variable_scope(type(self).__name__) as self.__varscope:

            prev_output, prev_output_shape = input_B_Di, input_shape

            for i_layer, ls in enumerate(layerspec):
                with variable_scope('layer_%d' % i_layer):
                    if ls['type'] == 'reshape':
                        _check_keys(ls, ['type', 'new_shape'], [])
                        self.layers.append(ReshapeLayer(prev_output, ls['new_shape']))

                    elif ls['type'] == 'fc':
                        _check_keys(ls, ['type', 'n'], ['initializer'])
                        self.layers.append(AffineLayer(
                            prev_output, prev_output_shape, output_shape=(ls['n'],), initializer=_parse_initializer(ls)))

                    elif ls['type'] == 'nonlin':
                        _check_keys(ls, ['type', 'func'], [])
                        self.layers.append(NonlinearityLayer(prev_output, prev_output_shape, ls['func']))

                    else:
                        raise NotImplementedError('Unknown layer type %s' % ls['type'])

                prev_output, prev_output_shape = self.layers[-1].output, self.layers[-1].output_shape
        self._output, self._output_shape = prev_output, prev_output_shape

    @property
    def varscope(self): return self.__varscope
    @property
    def output(self): return self._output
    @property
    def output_shape(self): return self._output_shape





# TODO add dropout layer
""" Add an LSTM Net for LSTM policy and classifier """ 
class LSTMNet(Layer):
    def __init__(self, input_B_T_Di, mask_B_T, hidden_B_Dh, cell_B_Dh, input_shape, layerspec_json):
        '''
        Args:
            layerspec (string): JSON string describing LSTM
        '''
        assert len(input_shape) >= 1

        self._input_shape = input_shape
        layerspec = json.loads(layerspec_json)

        util.header('Loading LSTM net specification')

        print (json.dumps(layerspec, indent=2, separators=(',', ': ')))

        self.layers = []
        B, T = input_B_T_Di.shape[0], input_B_T_Di.shape[1]
        with variable_scope(type(self).__name__) as self.__varscope:

            prev_output, prev_output_shape = input_B_T_Di, input_shape

            for i_layer, ls in enumerate(layerspec):
                with variable_scope('layer_%d' % i_layer):
                    # not implemented yet
                    if ls['type'] == 'reshape':
                        _check_keys(ls, ['type', 'new_shape'], [])
                        self.layers.append(ReshapeLayer(prev_output, ls['new_shape']))

                    # add sequential input setting
                    elif ls['type'] == 'fc':
                        _check_keys(ls, ['type', 'n'], ['initializer'])
                        self.layers.append(AffineLayer(
                            prev_output, prev_output_shape, output_shape=(ls['n'],), initializer=_parse_initializer(ls), sequential_input=True))

                    elif ls['type'] == 'nonlin':
                        _check_keys(ls, ['type', 'func'], [])
                        self.layers.append(NonlinearityLayer(prev_output, prev_output_shape, ls['func']))

                    # add lstm layer
                    elif ls['type'] == 'lstm':
                        _check_keys(ls, ['type', 'n'], [])
                        self.layers.append(LSTMLayer(prev_output, mask_B_T, hidden_B_Dh, cell_B_Dh, prev_output_shape, output_shape=(ls['n'],), initializer=None))

                    else:
                        raise NotImplementedError('Unknown layer type %s' % ls['type'])

                prev_output, prev_output_shape = self.layers[-1].output, self.layers[-1].output_shape
            self._output, self._output_shape = prev_output, prev_output_shape


    @property
    def varscope(self): return self.__varscpoe
    @property
    def output(self): return self._output
    @property
    def output_shape(self): return self._output_shape
    @property
    def input_shape(self): return self._input_shape






def _printfields(fields, sep=' | ', width=8, precision=4, print_header=True):
    names, vals, fmts = [], [], []
    for name, val, typeinfo in fields:
        names.append(name)
        if val is None:
            # display Nones as empty entries
            vals.append('')
            fmts.append('{:%ds}' % width)
        else:
            vals.append(val)
            if typeinfo is int:
                fmts.append('{:%dd}' % width)
            elif typeinfo is float:
                fmts.append('{:%d.%df}' % (width, precision))
            else:
                raise NotImplementedError(typeinfo)
    if print_header:
        header = ((('{:^%d}' % width) + sep) * len(names))[:-len(sep)].format(*names)
        print ('-'*len(header))
        print (header)
        print ('-'*len(header))
    print (sep.join(fmts).format(*vals))

def _type_to_col(t, pos):
    if t is int: return tables.Int32Col(pos=pos)
    if t is float: return tables.Float32Col(pos=pos)
    raise NotImplementedError(t)

def _type_to_str(t):
    if t is int: return 'i4'
    if t is float: return 'f4'
    raise NotImplementedError(t)


class TrainingLog(object):
    '''A training log backed by PyTables. Stores diagnostic numbers over time and model snapshots.'''

    def __init__(self, filename, attrs):
        if filename is None:
            util.warn('Warning: not writing log to any file!')
            self.f = None
        else:
            if os.path.exists(filename):
                raise RuntimeError('Log file %s already exists' % filename)
            self.f = tables.open_file(filename, mode='w')
            for k, v in attrs: self.f.root._v_attrs[k] = v
            self.log_table = None

        self.schema = None # list of col name / types for display

    def close(self):
        if self.f is not None: self.f.close()

    def write(self, kvt, display=True, **kwargs):
        # Write to the log
        if self.f is not None:
            if self.log_table is None:
                desc = {k: _type_to_col(t, pos) for pos, (k, _, t) in enumerate(kvt)}
                self.log_table = self.f.create_table(self.f.root, 'log', desc)

            row = self.log_table.row
            for k,v,_ in kvt: row[k] = v
            row.append()

            self.log_table.flush()

        if display:
            if self.schema is None:
                self.schema = [(k,t) for k,_,t in kvt]
            else:
                # If we are missing columns, fill them in with Nones
                nonefilled_kvt = []
                kvt_dict = {k:(v,t) for k,v,t in kvt}
                for schema_k, schema_t in self.schema:
                    if schema_k in kvt_dict:
                        v, t = kvt_dict[schema_k]
                        nonefilled_kvt.append((schema_k, v, t)) # check t == schema_t too?
                    else:
                        nonefilled_kvt.append((schema_k, None, schema_t))
                kvt = nonefilled_kvt
            _printfields(kvt, **kwargs)

    def write_snapshot(self, model, key_iter):
        if self.f is None: return

        # Save all variables into this group
        snapshot_root = '/snapshots/iter%07d' % key_iter

        for v in model.get_all_variables():
            assert v.name[0] == '/'
            fullpath = snapshot_root + v.name
            groupname, arrayname = fullpath.rsplit('/', 1)
            self.f.create_array(groupname, arrayname, v.get_value(), createparents=True)

        # Store the model hash as an attribute
        self.f.getNode(snapshot_root)._v_attrs.hash = model.savehash()

        self.f.flush()



class BasicTrainingLog(object):
    '''A basic training log by h5py. Stores diagnostic numbers over time and model snapshots.'''
    
    def __init__(self, filename, attrs):
        if filename is None:
            util.warn('Warning: not writing log to any file!')
            self.f = None
        else:
            if os.path.exists(filename):
                raise RuntimeError('Log file %s already exists' % filename)
            self.f = h5py.File(filename, mode='w')
        for k, v in attrs:
            self.f.attrs[k] = v
        self.plot_params = ['x', 'y', 'z', 'exdatax', 'exdatay', 'pdatax', 'pdatay']
        self.logs = None
        self.log_attrs = None
        self.log_dtype = None
        
        self.schema = None # list of col name / types for display

    def close(self):
        if self.f is not None:
            self.f.close()

    def add_log(self, log, display=False, **kwargs):
        # add new log
        if self.f is not None:
            if self.logs is None:
                self.logs = []
                self.log_attrs = {k: pos for pos, (k,v,_) in enumerate(log)}
                self.log_dtype = [_type_to_str(t) for k,_,t in log]
                self.log_dtype = ', '.join(self.log_dtype)
                self.log_dtype = np.dtype(self.log_dtype)
            row = [None for _ in xrange(len(self.log_attrs))]
            for k,v,_ in log:
                row[self.log_attrs[k]] = v
            self.logs.append(tuple(row))
        
        if display:
            if self.schema is None:
                self.schema = [(k,t) for k,_,t in log]
            else:
                # If we are missing columns, fill them in with None
                nonefilled_log = []
                log_dict = {k:(v,t) for k,v,t in log}
                for schema_k, schema_t in self.schema:
                    if schema_k in log_dict:
                        v, t = log_dict[schema_k]
                        nonefilled_log.append((schema_k, v, t))
                    else:
                        nonefilled_log.append((schema_k, None, schema_t))
                log = nonefilled_log
            _printfields(log, **kwargs)

    def write_snapshot(self, model, key_iter):
        if self.f is None: return 

        # Save all variables into this group 
        snapshot_root = '/snapshots/iter%07d' % key_iter

        for v in model.get_all_variables():
            assert v.name[0] == '/'
            fullpath = snapshot_root + v.name
            value = v.get_value()
            groupname, dsetname = fullpath.rsplit('/', 1)
            if groupname not in self.f:
                group = self.f.create_group(groupname)
            else:
                group = self.f[groupname]
            group.create_dataset(dsetname, value.shape, dtype=np.float32, data=value)
        # Store the model hash as an attribute
        self.f[snapshot_root].attrs['hash'] = model.savehash()

        self.f.flush()

    
    def write_plot(self, plot, key_iter):
        if self.f is None: return 

        # Save all ploting data into this group
        plot_root = '/plots/iter%07d' % key_iter
        group = self.f.create_group(plot_root)
        for col,param in enumerate(self.plot_params):
#            dsetname = plot_root + '/' + param
            group.create_dataset(param, plot[col].shape, dtype=plot[col].dtype, data=plot[col])
        self.f.flush()
            

    def write_log(self):
        # create log dataset
        if self.logs is not None:
            log_dset = self.f.create_dataset('log', (len(self.logs),), dtype=self.log_dtype, data=np.array(self.logs, dtype=self.log_dtype))
            for k,p in self.log_attrs.iteritems():
#            for k,p in self.log_attrs.items():
                log_dset.attrs['FIELD_{}_NAME'.format(p)] = k
            self.f.flush()



class NoOpStandardizer(object):
    def __init__(self, dim, eps=1e-6): pass
    def update(self, points_N_D): pass
    def standardize_expr(self, x_B_D): return x_B_D
    def unstandardize_expr(self, y_B_D): return y_B_D
    def standardize(self, x_B_D): return x_B_D
    def unstandardize(self, y_B_D): return y_B_D

class Standardizer(Model):
    def __init__(self, dim, eps=1e-6, init_count=0, init_mean=0., init_meansq=1.):
        '''
        Args:
            dim: dimension of the space of points to be standardized
            eps: small constant to add to denominators to prevent division by 0
            init_count, init_mean, init_meansq: initial values for accumulators

        Note:
            if init_count is 0, then init_mean and init_meansq have no effect beyond
            the first call to update(), which will ignore their values and
            replace them with values from a new batch of data.
        '''
        self._eps = eps
        self._dim = dim
#        print 'Debug: nn'
        with variable_scope(type(self).__name__) as self.__varscope:
            self._count = get_variable('count', np.array(float(init_count)).astype(theano.config.floatX), trainable=False)
            self._mean_1_D = get_variable('mean_1_D', np.full((1, self._dim), init_mean).astype(theano.config.floatX), broadcastable=(True,False), trainable=False)
            self._meansq_1_D = get_variable('meansq_1_D', np.full((1, self._dim), init_meansq).astype(theano.config.floatX), broadcastable=(True,False), trainable=False)
            # Convert data type to follow theano config
#            self._mean_1_D = self._mean_1_D.astype(theano.config.floatX)
#            self._meansq_1_D = self._meansq_1_D.astype(theano.config.floatX)
#            print 'Debug: _mean_1_D dtype:', self._mean_1_D.dtype
#            print 'Debug: _meansq_1_D dtype:', self._meansq_1_D.dtype
        self._stdev_1_D = tensor.sqrt(tensor.nnet.relu(self._meansq_1_D - tensor.square(self._mean_1_D)))
        # Relu ensures inside is nonnegative. maybe the better choice would have been to
        # add self._eps inside the square root, but I'm keeping things this way to preserve
        # backwards compatibility with existing saved models.

        self.get_mean = self._mean_1_D.get_value
        self.get_stdev = theano.function([], self._stdev_1_D[0,:]) # TODO: return with shape (1,D)

    @property
    def varscope(self): return self.__varscope

    def update(self, points_N_D):
        assert points_N_D.ndim == 2 and points_N_D.shape[1] == self._dim
        num = points_N_D.shape[0]
        count = float(self._count.get_value())
        a = count/(count+num)
        self._mean_1_D.set_value(a*self._mean_1_D.get_value() + (1.-a)*points_N_D.mean(axis=0, keepdims=True))
        self._meansq_1_D.set_value(a*self._meansq_1_D.get_value() + (1.-a)*(points_N_D**2).mean(axis=0, keepdims=True))
        self._count.set_value(count + num)

    def standardize_expr(self, x_B_D):
        return (x_B_D - self._mean_1_D) / (self._stdev_1_D + self._eps)

    def unstandardize_expr(self, y_B_D):
        return y_B_D*(self._stdev_1_D + self._eps) + self._mean_1_D

    def standardize(self, x_B_D):
        assert x_B_D.ndim == 2
        return (x_B_D - self.get_mean()) / (self.get_stdev() + self._eps)

    def unstandardize(self, y_B_D):
        assert y_B_D.ndim == 2
        return y_B_D*(self.get_stdev() + self._eps) + self.get_mean()



class SeqStandardizer(Model):
    def __init__(self, dim, eps=1e-6, init_count=0, init_mean=0., init_meansq=1.):
        self._eps = eps
        self._dim = dim

        with variable_scope(type(self).__name__) as self.__varscope:
            self._count = get_variable('count', np.array(float(init_count)).astype(theano.config.floatX), trainable=False)
            self._mean_1_1_D = get_variable('mean_1_1_D', np.full((1, 1, self._dim), init_mean).astype(theano.config.floatX), broadcastable=(True,True,False), trainable=False)
            self._meansq_1_1_D = get_variable('meansq_1_1_D', np.full((1, 1, self._dim), init_meansq).astype(theano.config.floatX), broadcastable=(True,True,False), trainable=False)
            # Convert data type to follow theano config
        self._stdev_1_1_D = tensor.sqrt(tensor.nnet.relu(self._meansq_1_1_D - tensor.square(self._mean_1_1_D)))
        
        self.get_mean = self._mean_1_1_D.get_value
        self.get_stdev = theano.function([], self._stdev_1_1_D[0,:,:]) # TODO return with shape (1,1,D)

    @property
    def varscope(self): return self.__varscope

    def update(self, points_N_D):
#        print("Debug: points_N_D:", points_N_D.ndim, points_N_D.shape)
#        print("Debug: dim:", self._dim)
        assert points_N_D.ndim == 3 and points_N_D.shape[2] == self._dim
        
        num = points_N_D.shape[0]
        count = float(self._count.get_value())
        a = count/(count+num)
        self._mean_1_1_D.set_value(a*self._mean_1_1_D.get_value() + (1.-a)*np.mean(np.mean(points_N_D, axis=1, keepdims=True), axis=0, keepdims=True))
        self._count.set_value(count + num)
        self._meansq_1_1_D.set_value(a*self._meansq_1_1_D.get_value() + (1.-a)*np.mean(np.mean(points_N_D**2, axis=1, keepdims=True), axis=0, keepdims=True))

    def standardize_expr(self, x_B_T_D):
        return (x_B_T_D - self._mean_1_1_D) / (self._stdev_1_1_D + self._eps)

    def unstandardize_expr(self, y_B_T_D):
        return y_B_T_D*(self._stdev_1_1_D + self._eps) + self._mean_1_1_D

    def standardize(self, x_B_T_D):
        assert x_B_T_D.ndim == 3
        return (x_B_T_D - self.get_mean()) / (self.get_stdev() + self._eps)

    def unstandardize(self, y_B_T_D):
        assert y_B_T_D.ndim == 3
        return y_B_T_D*(self.get_stdev() + self._eps) + self.get_mean()



def test_standardizer():
    D = 10
    s = Standardizer(D, eps=0)

    x_N_D = np.random.randn(200, D)
    s.update(x_N_D)

    x2_N_D = np.random.randn(300, D)
    s.update(x2_N_D)

    allx = np.concatenate([x_N_D, x2_N_D], axis=0)
    assert np.allclose(s._mean_1_D.get_value()[0,:], allx.mean(axis=0))
    assert np.allclose(s.get_stdev(), allx.std(axis=0))
    print ('ok')

if __name__ == '__main__':
    test_standardizer()
