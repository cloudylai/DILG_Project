from . import nn, util, thutil, optim, ContinuousSpace, FiniteSpace, RaggedArray
from collections import namedtuple
from contextlib import contextmanager
import environments
import numpy as np

import theano
from theano import tensor

from abc import abstractmethod



class Policy(nn.Model):
    def __init__(self, obsfeat_space, action_space, num_actiondist_params, enable_obsnorm, varscope_name):
        self.obsfeat_space, self.action_space, self._num_actiondist_params = obsfeat_space, action_space, num_actiondist_params

#        print 'Debug: Policy'
        with nn.variable_scope(varscope_name) as self.__varscope:
            # Action distribution for this current policy
            obsfeat_B_Df = tensor.matrix(name='obsfeat_B_Df', dtype=theano.config.floatX)
#            print 'Debug 0: obsfeat_B_Df dtype:', obsfeat_B_Df.dtype
            with nn.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim)
            normalized_obsfeat_B_Df = self.obsnorm.standardize_expr(obsfeat_B_Df)
#            print 'Debug 0: normalized_obsfeat_B_Df dtype:', normalized_obsfeat_B_Df.dtype
            # Convert (normalized) observations to action distributions
            actiondist_B_Pa = self._make_actiondist_ops(normalized_obsfeat_B_Df) # Pa == parameters of action distribution
#            print 'Debug 0: actiondist_B_Pa dtype:', actiondist_B_Pa.dtype
            self._compute_action_dist_params = thutil.function([obsfeat_B_Df], actiondist_B_Pa)

        # Only code above this line (i.e. _make_actiondist_ops) is allowed to make trainable variables.
        param_vars = self.get_trainable_variables()
#        for i in range(len(param_vars)):
#            print 'Debug 0: param_vars', i, 'name:', param_vars[i].name, 'dtype:', param_vars[i].dtype


        # Reinforcement learning
        input_actions_B_Da = tensor.matrix(name='input_actions_B_Da', dtype=theano.config.floatX if self.action_space.storage_type == theano.config.floatX else 'int64')
        logprobs_B = self._make_actiondist_logprob_ops(actiondist_B_Pa, input_actions_B_Da)
        logprobs_B = logprobs_B.astype(theano.config.floatX)
        # Proposal distribution from old policy
        proposal_actiondist_B_Pa = tensor.matrix(name='proposal_actiondist_B_Pa', dtype=theano.config.floatX)
        proposal_logprobs_B = self._make_actiondist_logprob_ops(proposal_actiondist_B_Pa, input_actions_B_Da)
        proposal_logprobs_B = proposal_logprobs_B.astype(theano.config.floatX)
        # Local RL objective
        advantage_B = tensor.vector(name='advantage_B')
        impweight_B = tensor.exp(logprobs_B - proposal_logprobs_B)
        obj = (impweight_B*advantage_B).mean()
        objgrad_P = thutil.flatgrad(obj, param_vars)
        objgrad_P = objgrad_P.astype(theano.config.floatX)
        # KL divergence from old policy
        kl_B = self._make_actiondist_kl_ops(proposal_actiondist_B_Pa, actiondist_B_Pa)
        kl_B = kl_B.astype(theano.config.floatX)
        kl = kl_B.mean()
        compute_obj_kl = thutil.function([obsfeat_B_Df, input_actions_B_Da, proposal_actiondist_B_Pa, advantage_B], [obj, kl])
        compute_obj_kl_with_grad = thutil.function([obsfeat_B_Df, input_actions_B_Da, proposal_actiondist_B_Pa, advantage_B], [obj, kl, objgrad_P])
        # KL Hessian-vector product
        v_P = tensor.vector(name='v')
        klgrad_P = thutil.flatgrad(kl, param_vars)
        klgrad_P = klgrad_P.astype(theano.config.floatX)
        hvpexpr = thutil.flatgrad((klgrad_P*v_P).sum(), param_vars)
        hvpexpr = hvpexpr.astype(theano.config.floatX)
        # hvpexpr = tensor.Rop(klgrad_P, param_vars, thutil.unflatten_into_tensors(v_P, [v.get_value().shape for v in param_vars]))
        hvp = thutil.function([obsfeat_B_Df, proposal_actiondist_B_Pa, v_P], hvpexpr)
        compute_hvp = lambda _obsfeat_B_Df, _input_actions_B_Da, _proposal_actiondist_B_Pa, _advantage_B, _v_P: hvp(_obsfeat_B_Df, _proposal_actiondist_B_Pa, _v_P)
        # TRPO step
        self._ngstep = optim.make_ngstep_func(self, compute_obj_kl, compute_obj_kl_with_grad, compute_hvp)

        ##### Publicly-exposed functions #####
        # for debugging
        self.compute_internal_normalized_obsfeat = thutil.function([obsfeat_B_Df], normalized_obsfeat_B_Df)

        # Supervised learning objective: log likelihood of actions given state
        bclone_loss = -logprobs_B.mean()
#        print 'Debug 0: bclone_loss dtype:', bclone_loss.dtype
        bclone_lr = tensor.scalar(name='bclone_lr')
#        print 'Debug 0: bclone_lr dtype:', bclone_lr.dtype
        self.step_bclone = thutil.function(
            [obsfeat_B_Df, input_actions_B_Da, bclone_lr],
            bclone_loss,
            updates=thutil.adam(bclone_loss, param_vars, lr=bclone_lr))
        self.compute_bclone_loss_and_grad = thutil.function(
            [obsfeat_B_Df, input_actions_B_Da],
            [bclone_loss, thutil.flatgrad(bclone_loss, param_vars)])
        self.compute_bclone_loss = thutil.function(
            [obsfeat_B_Df, input_actions_B_Da],
            bclone_loss)

        self.compute_action_logprobs = thutil.function(
            [obsfeat_B_Df, input_actions_B_Da],
            logprobs_B)


    @property
    def varscope(self): return self.__varscope

    def update_obsnorm(self, obs_B_Do):
        '''Update observation normalization using a moving average'''
        self.obsnorm.update(obs_B_Do)

    def sample_actions(self, obsfeat_B_Df, deterministic=False):
        '''Samples actions conditioned on states'''
        actiondist_B_Pa = self._compute_action_dist_params(obsfeat_B_Df)
        return self._sample_from_actiondist(actiondist_B_Pa, deterministic), actiondist_B_Pa

    # To be overridden
    @abstractmethod
    def _make_actiondist_ops(self, obsfeat_B_Df): pass
    @abstractmethod
    def _make_actiondist_logprob_ops(self, actiondist_B_Pa, input_actions_B_Da): pass
    @abstractmethod
    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa): pass
    @abstractmethod
    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic): pass
    @abstractmethod
    def _compute_actiondist_entropy(self, actiondist_B_Pa): pass


GaussianPolicyConfig = namedtuple('GaussianPolicyConfig', 'hidden_spec, min_stdev, init_logstdev, enable_obsnorm')
class GaussianPolicy(Policy):
    def __init__(self, cfg, obsfeat_space, action_space, varscope_name):
        assert isinstance(cfg, GaussianPolicyConfig)
        assert isinstance(obsfeat_space, ContinuousSpace) and isinstance(action_space, ContinuousSpace)

        self.cfg = cfg
        Policy.__init__(
            self,
            obsfeat_space=obsfeat_space,
            action_space=action_space,
            num_actiondist_params=action_space.dim*2,
            enable_obsnorm=cfg.enable_obsnorm,
            varscope_name=varscope_name)


    def _make_actiondist_ops(self, obsfeat_B_Df):
        # Computes action distribution mean (of a Gaussian) using MLP
#        print "Debug 1: obsfeat_B_Df dtype:", obsfeat_B_Df.dtype
        with nn.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, (self.obsfeat_space.dim,), self.cfg.hidden_spec)
#            print "Debug 1: net output dtype:", net.output.dtype
        with nn.variable_scope('out'):
            mean_layer = nn.AffineLayer(net.output, net.output_shape, (self.action_space.dim,), initializer=np.zeros((net.output_shape[0], self.action_space.dim)))
#            print "Debug 1: mean_layer output dtype:", mean_layer.output.dtype
            assert mean_layer.output_shape == (self.action_space.dim,)
        means_B_Da = mean_layer.output

        # Action distribution log standard deviations are parameters themselves
        logstdevs_1_Da = nn.get_variable('logstdevs_1_Da', np.full((1, self.action_space.dim), self.cfg.init_logstdev).astype(theano.config.floatX), broadcastable=(True,False))
        stdevs_1_Da = self.cfg.min_stdev + tensor.exp(logstdevs_1_Da) # minimum stdev seems to make density / kl computations more stable
        stdevs_B_Da = tensor.ones_like(means_B_Da)*stdevs_1_Da # "broadcast" to (B,Da)

        actiondist_B_Pa = tensor.concatenate([means_B_Da, stdevs_B_Da], axis=1)
        return actiondist_B_Pa

    def _extract_actiondist_params(self, actiondist_B_Pa):

        means_B_Da = actiondist_B_Pa[:, :self.action_space.dim]
        stdevs_B_Da = actiondist_B_Pa[:, self.action_space.dim:]
        return means_B_Da, stdevs_B_Da

    def _make_actiondist_logprob_ops(self, actiondist_B_Pa, input_actions_B_Da):
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return thutil.gaussian_log_density(means_B_Da, stdevs_B_Da, input_actions_B_Da)

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        proposal_means_B_Da, proposal_stdevs_B_Da = self._extract_actiondist_params(proposal_actiondist_B_Pa)
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return thutil.gaussian_kl(proposal_means_B_Da, proposal_stdevs_B_Da, means_B_Da, stdevs_B_Da)

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic):
        adim = self.action_space.dim
        means_B_Da, stdevs_B_Da = actiondist_B_Pa[:,:adim], actiondist_B_Pa[:,adim:]
        if deterministic:
            return means_B_Da
        stdnormal_B_Da = np.random.randn(actiondist_B_Pa.shape[0], adim)
        stdnormal_B_Da = stdnormal_B_Da.astype(theano.config.floatX)
#        print "Debug 4: means_B_Da dtype:", means_B_Da.dtype
#        print "Debug 4: stdevs_B_Da dtype:", stdevs_B_Da.dtype
#        print "Debug 4: stdnormal_B_Da dtype:", stdnormal_B_Da.dtype
        assert stdnormal_B_Da.shape == means_B_Da.shape == stdevs_B_Da.shape
        return (stdnormal_B_Da*stdevs_B_Da) + means_B_Da

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
#        print "Debug 5: actiondist_B_Pa dtype:", actiondist_B_Pa.dtype
        _, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return util.gaussian_entropy(stdevs_B_Da)

    def compute_actiondist_mean(self, obsfeat_B_Df):
#        print "Debug 6: obsfeat_B_Df dtype:", obsfeat_B_Df.dtype
        actiondist_B_Pa = self._compute_action_dist_params(obsfeat_B_Df)
        means_B_Da, _ = self._extract_actiondist_params(actiondist_B_Pa)
        return means_B_Da

GibbsPolicyConfig = namedtuple('GibbsPolicyConfig', 'hidden_spec, enable_obsnorm')
class GibbsPolicy(Policy):
    def __init__(self, cfg, obsfeat_space, action_space, varscope_name):
        assert isinstance(cfg, GibbsPolicyConfig)
        assert isinstance(obsfeat_space, ContinuousSpace) and isinstance(action_space, FiniteSpace)
        self.cfg = cfg
        Policy.__init__(
            self,
            obsfeat_space=obsfeat_space,
            action_space=action_space,
            num_actiondist_params=action_space.size,
            enable_obsnorm=cfg.enable_obsnorm,
            varscope_name=varscope_name)

    def _make_actiondist_ops(self, obsfeat_B_Df):
        # Computes action distribution using MLP
        # Actiondist consists of the log probabilities
        with nn.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, (self.obsfeat_space.dim,), self.cfg.hidden_spec)
        with nn.variable_scope('out'):
            out_layer = nn.AffineLayer(
                net.output, net.output_shape, (self.action_space.size,),
                initializer=np.zeros((net.output_shape[0], self.action_space.size)))
            assert out_layer.output_shape == (self.action_space.size,)
        scores_B_Pa = out_layer.output # unnormalized (unshifted) log probability
        actiondist_B_Pa = scores_B_Pa - thutil.logsumexp(scores_B_Pa, axis=1) # log probability
        return actiondist_B_Pa

    def _make_actiondist_logprob_ops(self, actiondist_B_Pa, input_actions_B_Da):
        return actiondist_B_Pa[tensor.arange(actiondist_B_Pa.shape[0]), input_actions_B_Da[:,0]]

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        return thutil.categorical_kl(proposal_actiondist_B_Pa, actiondist_B_Pa)

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic):
        probs_B_A = np.exp(actiondist_B_Pa); assert probs_B_A.shape[1] == self.action_space.size
        if deterministic:
            return probs_B_A.argmax(axis=1)[:,None]
        return util.sample_cats(probs_B_A)[:,None]

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        return util.categorical_entropy(np.exp(actiondist_B_Pa))



# XXX use original Policy with memory as SequentialPolicy
# XXX replace LSTM with standard NN
class SequentialPolicy(nn.Model):
    def __init__(self, obsfeat_space, action_space, num_actiondist_params, time_step, enable_obsnorm, enable_actnorm, varscope_name):
        self.obsfeat_space, self.action_space, self._num_actiondist_params, self.time_step = obsfeat_space, action_space, num_actiondist_params, time_step
        self.hidden, self.cell = None, None

        with nn.variable_scope(varscope_name) as self.__varscope:
            # Action distribution for this current policy
            obsfeat_BT_Df = tensor.matrix(name='obsfeat_BT_Df', dtype=theano.config.floatX)
#            hidden_B_Dh = tensor.matrix(name='hidden_B_Dh', dtype=theano.config.floatX)
#            cell_B_Dh = tensor.matrix(name='cell_B_Dh', dtype=theano.config.floatX)

            with nn.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim)
            normalized_obsfeat_BT_Df = self.obsnorm.standardize_expr(obsfeat_BT_Df)
            # Convert (normalized) observations to action distributions
            actiondist_BT_Pa = self._make_actiondist_ops(normalized_obsfeat_BT_Df) # Pa == parameters of action distribution
            self._compute_action_dist_params = thutil.function([obsfeat_BT_Df], actiondist_BT_Pa)
            
#            self._compute_action_dist_with_hidden_cell = thutil.function([obsfeat_B_T_Df, mask_B_T, hidden_B_Dh, cell_B_Dh], [actiondist_B_T_Pa, new_hidden_B_Dh, new_cell_B_Dh])

        # Only code above this line (i.e. _make_actiondist_ops) is allowed to make trainable variables.
        param_vars = self.get_trainable_variables()

        # Reinforcement learning
        input_actions_BT_Da = tensor.matrix(name='input_actions_BT_Da', dtype=theano.config.floatX if self.action_space.storage_type == theano.config.floatX else 'int64')
        logprobs_BT = self._make_actiondist_logprob_ops(actiondist_BT_Pa, input_actions_BT_Da)
        logprobs_BT = logprobs_BT.astype(theano.config.floatX)
        # Proposal distribution from old policy
        proposal_actiondist_BT_Pa = tensor.matrix(name='proposal_actiondist_BT_Pa', dtype=theano.config.floatX)
        proposal_logprobs_BT = self._make_actiondist_logprob_ops(proposal_actiondist_BT_Pa, input_actions_BT_Da)
        proposal_logprobs_BT = proposal_logprobs_BT.astype(theano.config.floatX)
        
        # Local RL objective
        advantage_BT = tensor.vector(name='advantage_BT')
        impweight_BT = tensor.exp(logprobs_BT - proposal_logprobs_BT)
        obj_BT = impweight_BT*advantage_BT
        obj = obj_BT.mean()
        objgrad_P = thutil.flatgrad(obj, param_vars)
        objgrad_P = objgrad_P.astype(theano.config.floatX)
        # KL divergence from old policy
        kl_BT = self._make_actiondist_kl_ops(proposal_actiondist_BT_Pa, actiondist_BT_Pa)
        kl_BT = kl_BT.astype(theano.config.floatX)
        kl = kl_BT.mean()
        compute_obj_kl = thutil.function([obsfeat_BT_Df, input_actions_BT_Da, proposal_actiondist_BT_Pa, advantage_BT], [obj, kl])
        compute_obj_kl_with_grad = thutil.function([obsfeat_BT_Df, input_actions_BT_Da, proposal_actiondist_BT_Pa, advantage_BT], [obj, kl, objgrad_P])
        # KL Hessian-vector product
        v_P = tensor.vector(name='v')
        klgrad_P = thutil.flatgrad(kl, param_vars)
        klgrad_P = klgrad_P.astype(theano.config.floatX)
        hvpexpr = thutil.flatgrad((klgrad_P*v_P).sum(), param_vars)
        hvpexpr = hvpexpr.astype(theano.config.floatX)
        # hvpexpr = tensor.Rop(klgrad_P, param_vars, thutil.unflatten_into_tensors(v_P, [v.get_value().shape for v in param_vars]))
        hvp = thutil.function([obsfeat_BT_Df, proposal_actiondist_BT_Pa, v_P], hvpexpr)
        compute_hvp = lambda _obsfeat_BT_Df, _input_actions_BT_Da, _proposal_actiondist_BT_Pa, _advantage_BT, _v_P: hvp(_obsfeat_BT_Df, _proposal_actiondist_BT_Pa, _v_P)
        # TRPO step 
        self._ngstep = optim.make_ngstep_func(self, compute_obj_kl, compute_obj_kl_with_grad, compute_hvp)


        ##### Publicly-exposed functions #####
        # for debugging
        self.compute_internal_normalized_obsfeat = thutil.function([obsfeat_BT_Df], normalized_obsfeat_BT_Df)
#        self.compute_internal_normalized_prevactrepr = thutil.function([prevact_B_T_Da], normalized_prevactrepr_B_T_Da)

        # Supervised learning objective: log likelihood of actions given state
#        bclone_loss = -logprobs_BT.mean()
#        bclone_lr = tensor.scalar(name='bclone_lr')
#        self.ste_bclone = thutil.function(
#            [obsfeat_BT_Df, input_actions_BT_Da, bclone_lr],
#            bclone_loss,
#            updates=thutil.adam(blcone_loss, param_vars, lr=bclone_lr))
#        self.compute_bclone_loss_and_grad = thutil.function(
#            [obsfeat_BT_Df, input_actions_BT_Da],
#            bclone_loss)

        self.compute_action_logprobs = thutil.function(   
            [obsfeat_BT_Df, input_actions_BT_Da],
            logprobs_BT)


    @property
    def varscope(self): return self.__varscope

    def update_obsnorm(self, obs_BT_Do):
        '''Update observation normalizeation using a moving average'''
        self.obsnorm.update(obs_BT_Do)

#    def update_actnorm(self, act_B_T_Da):
#        '''Update action normalization using a moving average'''
#        self.actnorm.update(act_B_T_Da)

#    def sample_actions(self, obsfeat_B_Df, deterministic=False, restart=False, n_samples=None):
    def sample_actions(self, obsfeat_B_Df, deterministic=False):
        '''Samples actions conditioned on states'''
        actiondist_B_Pa = self._compute_action_dist_params(obsfeat_B_Df)
#        self.hidden = new_hidden_1_Dh
#        self.cell = new_cell_1_Dh
        return self._sample_from_actiondist(actiondist_B_Pa, deterministic), actiondist_B_Pa


    # To be overridden
    @abstractmethod
    def _make_actiondist_ops(self, obsfeat_BT_Df): pass
    @abstractmethod
    def _make_actiondist_logprob_ops(self, actiondist_BT_Pa, input_actions_BT_Da): pass
    @abstractmethod
    def _make_actiondist_kl_ops(self, proposal_actiondist_BT_Pa, actiondist_BT_Pa): pass
    @abstractmethod
    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic): pass
    @abstractmethod
    def _compute_actiondist_entropy(self, actiondist_B_Pa): pass
#    # XXX memory method
#    @abstractmethod
#    def restart_hidden_cell(n_samples): pass
#    @abstractmethod
#    def _compute_action_dist_with_hidden_cell_restart(obsfeat_B_T_Df, mask_B_T, restart, n_samples): pass

#    @abstractmethod
#    def _make_action_representation(self, prevact_B_T_Da): pass




SeqGaussianPolicyConfig = namedtuple('SeqGaussianPolicyConfig', 'hidden_spec, time_step, min_stdev, init_logstdev, enable_obsnorm, enable_actnorm')
class SeqGaussianPolicy(SequentialPolicy):
    def __init__(self, cfg, obsfeat_space, action_space, varscope_name):
        assert isinstance(cfg, SeqGaussianPolicyConfig)
        assert isinstance(obsfeat_space, ContinuousSpace) and isinstance(action_space, ContinuousSpace)

        self.cfg = cfg
        SequentialPolicy.__init__(
            self,
            obsfeat_space=obsfeat_space,
            action_space=action_space,
            num_actiondist_params=action_space.dim*2,
            time_step=cfg.time_step,
            enable_obsnorm=cfg.enable_obsnorm,
            enable_actnorm=cfg.enable_actnorm,    # XXX not implement yet
            varscope_name=varscope_name)

 
    def _make_actiondist_ops(self, obsfeat_BT_Df):
        # Comuptes action distribution mean (of a Gaussian) using MLP
        new_hidden_B_Dh, new_cell_B_Dh = None, None
        with nn.variable_scope('hidden'):
            self.nn_net = nn.FeedforwardNet(obsfeat_BT_Df, (self.obsfeat_space.dim,), self.cfg.hidden_spec)
        with nn.variable_scope('out'):
            self.mean_layer = nn.AffineLayer(
                self.nn_net.output, self.nn_net.output_shape, (self.action_space.dim,), 
                initializer=np.zeros((self.nn_net.output_shape[0], self.action_space.dim)))
            assert self.mean_layer.output_shape == (self.action_space.dim,)
        mean_BT_Da = self.mean_layer.output

        # Action distribution log standard deviations are parameters themselves
        logstdevs_1_Da = nn.get_variable('logstdevs_1_Da', np.full((1, self.action_space.dim).astype(theano.config.floatX), self.cfg.init_logstdev), broadcasetable=(True,False))
        stdevs_1_Da = self.cfg.min_stdev + tensor.exp(logstdevs_1_Da) # minimum stdev seems to make density / kl computations more stable
        stdevs_BT_Da = tensor.ones_like(means_BT_Da)*stdevs_1_Da # "broadcast" to (BT,Da)
        actiondist_BT_Pa = tensor.concatenate([means_BT_Da, stdevs_BT_Da], axis=1)
        # convert data type to fit theano config
        logstdevs_1_Da = logstdevs_1_Da.astype(theano.config.floatX)
        stdevs_1_Da = stdevs_1_Da.astype(theano.config.floatX)
        stdevs_BT_Da = stdevs_BT_Da.astype(theano.config.floatX)
        actiondist_BT_Pa = actiondist_BT_Pa.astype(theano.config.floatX)
#        # XXX get the hidden and cells of lstm 
#        # only handle 1 lstm layer
#        new_hidden_B_Dh = self.lstm_net.layers[0].output[:,-1,:]
#        new_cell_B_Dh = self.lstm_net.layers[0].cell[:,-1,:]
        return actiondist_BT_Pa


    def _extract_actiondist_params(self, actiondist_BT_Pa):
        means_BT_Da = actiondist_BT_Pa[:, :self.action_space.dim]
        stdevs_BT_Da = actiondist_BT_Pa[:, self.action_space.dim:]
        return means_BT_Da, stdevs_BT_Da

    
    def _make_actiondist_logprob_ops(self, actiondist_BT_Pa, input_actions_BT_Da):
        means_BT_Da, stdevs_BT_Da = self._extract_actiondist_params(actiondist_BT_Pa)
        return thutil.gaussian_log_density(means_BT_Da, stdevs_B_Da, input_actions_BT_Da)

    def _make_actiondist_kl_ops(self, proposal_actiondist_BT_Pa, actiondist_BT_Pa):
        proposal_means_BT_Da, proposal_stdevs_BT_Da = self._extract_actiondist_params(proposal_actiondist_BT_Pa)
        means_BT_Da, stdevs_BT_Da = self._extract_actiondist_params(actiondist_BT_Pa)
        return thutil.gaussian_kl(proposal_means_BT_Da, proposal_stdevs_BT_Da, means_BT_Da, stdevs_BT_Da)

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic):
        adim = self.action_space.dim
        means_B_Da, stdevs_B_Da = actiondist_B_Pa[:,:adim], actiondist_B_Pa[:,adim:]
        if deterministic:
            return means_B_Da
        stdnormal_B_Da = np.random.randn(actiondist_B_Pa.shape[0], adim)
        stdnormal_B_Da = stdnormal_B_Da.astype(theano.config.floatX)
        assert stdnormal_B_Da.shape == means_B_Da.shape == stdevs_B_Da.shape
        return (stdnormal_B_Da*stdevs_B_Da) + means_B_Da

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        _, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return util.gaussian_entropy(stdevs_B_Da)

#    # XXX handle restart
#    def restart_hidden_cell(self, n_samples=None):
#        # XXX only handle 1 lstm layer
#        self.lstm_net.layers[0].restart_hidden_cell(n_samples=n_samples)
#        self.hidden = self.lstm_net.layers[0].sample_hidden
#        self.cell = self.lstm_net.layers[0].sample_cell
        
#    def _compute_action_dist_with_hidden_cell_restart(self, obsfeat_B_T_Df, mask_B_T, restart, n_samples=None):
#        if restart:
#            self.restart_hidden_cell(n_samples=n_samples)
#        return self._compute_action_dist_with_hidden_cell(obsfeat_B_T_Df, mask_B_T, self.hidden, self.cell)

#    # no need other representations for continuous action
#    def _make_action_representation(self, prevact_B_T_Da):
#        return prevact_B_T_Da

    def compute_actiondist_mean(self, obsfeat_B_Df):
        actiondist_B_Pa = self._compute_action_dist_params(obsfeat_B_Df)
        means_B_Da, _ = self._extract_actiondist_params(actiondist_B_Pa)
        return means_B_Da





SeqGibbsPolicyConfig = namedtuple('SeqGibbsPolicyConfig', 'hidden_spec, time_step, enable_obsnorm, enable_actnorm')
class SeqGibbsPolicy(SequentialPolicy):
    def __init__(self, cfg, obsfeat_space, action_space, varscope_name):
        assert isinstance(cfg, SeqGibbsPolicyConfig)
        assert isinstance(obsfeat_space, ContinuousSpace) and isinstance(action_space, FiniteSpace)
        self.cfg = cfg
        SequentialPolicy.__init__(
            self,
            obsfeat_space=obsfeat_space,
            action_space=action_space,
            num_actiondist_params=action_space.size,
            time_step=cfg.time_step,
            enable_obsnorm=cfg.enable_obsnorm,
            enable_actnorm=cfg.enable_actnorm,    #XXX not implement yet
            varscope_name=varscope_name)


    def _make_actiondist_ops(self, obsfeat_BT_Df):
        with nn.variable_scope('hidden'):
            self.nn_net = nn.FeedforwardNet(obsfeat_BT_Df, (self.obsfeat_space.dim,), self.cfg.hidden_spec)
        with nn.variable_scope('out'):
            self.out_layer = nn.AffineLayer(
                self.nn_net.output, self.nn_net.output_shape, (self.action_space.size,),
                initializer=np.zeros((self.nn_net.output_shape[0], self.action_space.size)))
            assert self.out_layer.output_shape == (self.action_space.size,)
        scores_BT_Pa = self.out_layer.output # unnormalized (unshifted) log probability
        actiondist_BT_Pa = scores_BT_Pa - thutil.logsumexp(scores_BT_Pa, axis=1) # log probability
#        # XXX get hiddens and cells of lstm
#        # only handle 1 lstm layer
#        new_hidden_B_Dh = self.lstm_net.layers[0].output[:,-1,:]
#        new_cell_B_Dh = self.lstm_net.layers[0].cell[:,-1,:]
#        # reshape: (B*T, Pa) => (B, T, Pa)
#        actiondist_B_T_Pa = tensor.reshape(actiondist_BT_Pa, (B, T, actiondist_BT_Pa.shape[1]), ndim=3)
        return actiondist_BT_Pa


    def _make_actiondist_logprob_ops(self, actiondist_BT_Pa, input_actions_BT_Da):
        return actiondist_BT_Pa[tensor.arange(actiondist_BT_Pa.shape[0]), input_actions_BT_Da[:,0]]

    def _make_actiondist_kl_ops(self, proposal_actiondist_BT_Pa, actiondist_BT_Pa):
        return thutil.categorical_kl(proposal_actiondist_BT_Pa, actiondist_BT_Pa)

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic):
        probs_B_A = np.exp(actiondist_B_Pa); assert probs_B_A.shape[1] == self.action_space.size
        if deterministic:
            return probs_B_A.argmax(axis=1)[:,None]
        return util.sample_cats(probs_B_A)[:,None]

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        return util.categorical_entropy(np.exp(actiondist_B_Pa))

#    def _compute_seq_actiondist_entropy(self, actiondist_B_T_Pa):
#        B, T, Pa = actiondist_B_T_Pa.shape[0], actiondist_B_T_Pa.shape[1], actiondist_B_T_Pa.shape[2]
#        actiondist_BT_Pa = np.reshape(actiondist_B_T_Pa, (B*T, Pa))
#        return util.categorical_entropy(np.exp(actiondist_BT_Pa))

#    def _make_action_representation(self, prevact_B_T_Da):
#        ### use one-hot vector to represent action
#        
#        B, T, Da = prevact_B_T_Da.shape[0], prevact_B_T_Da.shape[1], prevact_B_T_Da.shape[2]
#        prevact_BTDa = tensor.reshape(prevact_B_Da, (B*T*Da, ), ndim=1)
#        prevact_onehot_BTDa = tensor.extra_ops.to_one_hot(prevact_BTDa, self.action_space.size)
#        prevact_onehot_B_T_Da = tensor.reshape(prevact_onehot_BDa, (B, T, self.action_space.size), ndim=3)
#        return prevact_onehot_B_T_Da

    # XXX handle restart
#    def restart_hidden_cell(self, n_samples=None):
#        # XXX only handle 1 lstm layer
#        self.lstm_net.layers[0].restart_hidden_cell(n_samples=n_samples)
#        self.hidden = self.lstm_net.layers[0].sample_hidden
#        self.cell = self.lstm_net.layers[0].sample_cell


#    def _compute_action_dist_with_hidden_cell_restart(self, obsfeat_B_T_Df, mask_B_T, restart, n_samples=None):
#        if restart:
#            self.restart_hidden_cell(n_samples=n_samples)
#        return self._compute_action_dist_with_hidden_cell(obsfeat_B_T_Df, mask_B_T, self.hidden, self.cell)



def compute_qvals(r, gamma):
    assert isinstance(r, RaggedArray)
    trajlengths = r.lengths
    # Zero-fill the rewards on the right, then compute Q values
    rewards_B_T = r.padded(fill=0.)
    qvals_zfilled_B_T = util.discount(rewards_B_T, gamma)
    # Convert data type to follow theano config
    qvals_zfilled_B_T = qvals_zfilled_B_T.astype(theano.config.floatX)
    assert qvals_zfilled_B_T.shape == (len(trajlengths), trajlengths.max())
    return RaggedArray([qvals_zfilled_B_T[i,:l] for i, l in enumerate(trajlengths)]), rewards_B_T

def compute_advantage(r, obsfeat, time, value_func, gamma, lam):
    assert isinstance(r, RaggedArray) and isinstance(obsfeat, RaggedArray) and isinstance(time, RaggedArray)
    trajlengths = r.lengths
    assert np.array_equal(obsfeat.lengths, trajlengths) and np.array_equal(time.lengths, trajlengths)
    B, maxT = len(trajlengths), trajlengths.max()

    # Compute Q values
    q, rewards_B_T = compute_qvals(r, gamma)
    q_B_T = q.padded(fill=np.nan); assert q_B_T.shape == (B, maxT) # q values, padded with nans at the end

    # Time-dependent baseline that cheats on the current batch
    simplev_B_T = np.tile(np.nanmean(q_B_T, axis=0, keepdims=True), (B, 1)); assert simplev_B_T.shape == (B, maxT)
    simplev = RaggedArray([simplev_B_T[i,:l] for i, l in enumerate(trajlengths)])

    # State-dependent baseline (value function)
    v_stacked = value_func.evaluate(obsfeat.stacked, time.stacked); assert v_stacked.ndim == 1
    v = RaggedArray(v_stacked, lengths=trajlengths)

    # Compare squared loss of value function to that of the time-dependent value function
    constfunc_prediction_loss = np.var(q.stacked)
    simplev_prediction_loss = np.var(q.stacked-simplev.stacked) #((q.stacked-simplev.stacked)**2).mean()
    simplev_r2 = 1. - simplev_prediction_loss/(constfunc_prediction_loss + 1e-8)
    vfunc_prediction_loss = np.var(q.stacked-v_stacked) #((q.stacked-v_stacked)**2).mean()
    vfunc_r2 = 1. - vfunc_prediction_loss/(constfunc_prediction_loss + 1e-8)
    # Convert data type to follow theano config
    simplev_r2 = simplev_r2.astype(theano.config.floatX)
    vfunc_r2 = vfunc_r2.astype(theano.config.floatX)

    # Compute advantage -- GAE(gamma, lam) estimator
    v_B_T = v.padded(fill=0.)
    # append 0 to the right
    v_B_Tp1 = np.concatenate([v_B_T, np.zeros((B,1), dtype=theano.config.floatX)], axis=1); assert v_B_Tp1.shape == (B, maxT+1)
    delta_B_T = rewards_B_T + gamma*v_B_Tp1[:,1:] - v_B_Tp1[:,:-1]
    adv_B_T = util.discount(delta_B_T, gamma*lam); assert adv_B_T.shape == (B, maxT)
    # Convert data type to follow theano config
    adv_B_T = adv_B_T.astype(theano.config.floatX)
    adv = RaggedArray([adv_B_T[i,:l] for i, l in enumerate(trajlengths)])
    assert np.allclose(adv.padded(fill=0), adv_B_T)

    return adv, q, vfunc_r2, simplev_r2


# r.shape: B x [(s,T)]
# XXX use the original compute_qvals code
def compute_sequence_qvals(r, gamma):
    assert isinstance(r, RaggedArray)
    trajlengths = r.lengths
    B, T = r.stacked.shape[0], r.stacked.shape[1]
    # XXX should not pad in sequence reward
    rewards_B_T = r.stacked
    qvals_zfilled_B_T = util.discount(rewards_B_T, gamma)
    # Convert data type to follow theano config
    qvals_zfilled_B_T = qvals_zfilled_B_T.astype(theano.config.floatX)
    assert qvals_zfilled_B_T.shape == (B, T)
#    cumlengths = [0] + list(np.cumsum(r.lengths))
#    indices = [(int(cumlengths[i]), int(cumlengths[i+1])) for i in range(len(cumlengths)-1)]
#    return RaggedArray([qvals_zfilled_B_T[l1:l2,:] for (l1, l2) in indices]), rewards_B_T
    # s == 1, np.split(B_T) => [B*(1_T)]
    return RaggedArray(qvals_zfilled_B_T, lengths=trajlengths), rewards_B_T


# add to compute advantage of sequential model
# XXX use new simulation of trajectory sequence. i.e. B_s_T == B_1_T => B_T, s == 1
# XXX use the original value function (not sequential)
def compute_sequence_advantage(r, obsfeat, time, value_func, gamma, lam):
    assert isinstance(r, RaggedArray) and isinstance(obsfeat, RaggedArray) and isinstance(time, RaggedArray)
    trajlengths = r.lengths
    assert np.array_equal(obsfeat.lengths, trajlengths) and np.array_equal(time.lengths, trajlengths)
#    print("Debug: compute_sequence_advantage: ")
#    print("Debug: r stacked shape:", r.stacked)
#    print("Debug: obsfeat stacked shape:", obsfeat.stacked)
#    print("Debug: time stacked shape:", time.stacked)
#    L, B, maxT = len(r.lengths), r.stacked.shape[0], r.stacked.shape[1]
    B, maxT = r.stacked.shape[0], r.stacked.shape[1]
#    print("Debug: B, maxT:", B, maxT)
#    print("Debug: trajlengths:", trajlengths)
#    cumlengths = [0] + list(np.cumsum(r.lengths))
#    indices = [(int(cumlengths[i]), int(cumlengths[i+1])) for i in range(len(cumlengths)-1)]

    # Compute Q values
    q, rewards_B_T = compute_sequence_qvals(r, gamma)
    q_B_T = q.stacked; assert q_B_T.shape == (B, maxT)
#    q_B_T = q.padded(fill=np.nan); assert q_B_T.shape == (B, maxT) # q values, padded with nans at the end

    # Time-dependent baseline that cheats on the current batch
    simplev_B_T = np.tile(np.nanmean(q_B_T, axis=0, keepdims=True), (B, 1)); assert simplev_B_T.shape == (B, maxT)
    # s == 1, np.split(B_T) => [B*(1_T)] 
    simplev = RaggedArray(simplev_B_T, lengths=trajlengths)

    # State-dependent baseline (value function)
    v_stacked = value_func.evaluate(obsfeat.stacked, time.stacked); assert v_stacked.ndim == 2
    # TODO check shape 
    v = RaggedArray(v_stacked, lengths=trajlengths)
#    print("Debug: v_stacked shape:", v_stacked.shape)

    # Compare squared loss of value function to that of the time-dependent value function
    constfunc_prediction_loss = np.var(q.stacked)
    simplev_prediction_loss = np.var(q.stacked-simplev.stacked) #((q.stacked-simplev.stacked)**2).mean()
    simplev_r2 = 1. - simplev_prediction_loss/(constfunc_prediction_loss + 1e-8)
    vfunc_prediction_loss = np.var(q.stacked-v_stacked) #((q.stacked-v_stacked)**2).mean()
    vfunc_r2 = 1. - vfunc_prediction_loss/(constfunc_prediction_loss + 1e-8)
    # Convert data type to follow theano config
    simplev_r2 = simplev_r2.astype(theano.config.floatX)
    vfunc_r2 = vfunc_r2.astype(theano.config.floatX)

    # Compute advantage -- GAE(gamma, lam) estimator
    # XXX should not pad in sequential reward
    v_B_T = v.stacked
#    print("Debug: v_B_T shape:", v_B_T.shape)
    # append 0 to the right
    # XXX check the orginal code for compute_advantage    
    v_B_Tp1 = np.concatenate([v_B_T, np.zeros((B,1), dtype=theano.config.floatX)], axis=1); assert v_B_Tp1.shape == (B, maxT+1)
    delta_B_T = rewards_B_T + gamma*v_B_Tp1[:,1:] - v_B_Tp1[:,:-1]
    adv_B_T = util.discount(delta_B_T, gamma*lam); assert adv_B_T.shape == (B, maxT)
    # Convert data type to follow theano config
    adv_B_T = adv_B_T.astype(theano.config.floatX)
    # s == 1, np.split(B_T) => [B*(1_T)]
    adv = RaggedArray(adv_B_T, lengths=trajlengths)
    assert np.allclose(adv.stacked, adv_B_T)

    return adv, q, vfunc_r2, simplev_r2



class SamplingPolicyOptimizer(object):
    def __init__(self, mdp, discount, lam, policy, sim_cfg, step_func, value_func, obsfeat_fn):
        self.mdp, self.discount, self.lam, self.policy = mdp, discount, lam, policy
        self.sim_cfg = sim_cfg
        self.step_func = step_func
        self.value_func = value_func
        self.obsfeat_fn = obsfeat_fn

        self.total_num_sa = 0
        self.total_time = 0.
        self.curr_iter = 0

    def step(self):
        with util.Timer() as t_all:

            # Sample trajectories using current policy
            with util.Timer() as t_sample:
                # At the first iter, sample an extra batch to initialize standardization parameters
                if self.curr_iter == 0:
                    trajbatch0 = self.mdp.sim_mp(
                        policy_fn=lambda obsfeat_B_Df: self.policy.sample_actions(obsfeat_B_Df),
                        obsfeat_fn=self.obsfeat_fn,
                        cfg=self.sim_cfg)
                    self.policy.update_obsnorm(trajbatch0.obsfeat.stacked)
                    self.value_func.update_obsnorm(trajbatch0.obsfeat.stacked)

                trajbatch = self.mdp.sim_mp(
                    policy_fn=lambda obsfeat_B_Df: self.policy.sample_actions(obsfeat_B_Df),
                    obsfeat_fn=self.obsfeat_fn,
                    cfg=self.sim_cfg)
                # TODO: normalize rewards

            # Compute baseline / advantages
            with util.Timer() as t_adv:
                advantages, qvals, vfunc_r2, simplev_r2 = compute_advantage(
                    trajbatch.r, trajbatch.obsfeat, trajbatch.time,
                    self.value_func, self.discount, self.lam)

            # Take a step
            with util.Timer() as t_step:
                params0_P = self.policy.get_params()
                extra_print_fields = self.step_func(
                    self.policy, params0_P,
                    trajbatch.obsfeat.stacked, trajbatch.a.stacked, trajbatch.adist.stacked,
                    advantages.stacked)
                self.policy.update_obsnorm(trajbatch.obsfeat.stacked)

            # Fit value function for next iteration
            with util.Timer() as t_vf_fit:
                if self.value_func is not None:
                    extra_print_fields += self.value_func.fit(
                        trajbatch.obsfeat.stacked, trajbatch.time.stacked, qvals.stacked)

        # Log
        self.total_num_sa += sum(len(traj) for traj in trajbatch)
        self.total_time += t_all.dt
        fields = [
            ('iter', self.curr_iter, int),
            ('ret', trajbatch.r.padded(fill=0.).sum(axis=1).mean(), float), # average return for this batch of trajectories
            # ('discret', np.mean([q[0] for q in qvals]), float),
            # ('ravg', trajbatch.r.stacked.mean(), float), # average reward encountered
            ('avglen', int(np.mean([len(traj) for traj in trajbatch])), int), # average traj length
            ('nsa', self.total_num_sa, int), # total number of state-action pairs sampled over the course of training
            ('ent', self.policy._compute_actiondist_entropy(trajbatch.adist.stacked).mean(), float), # entropy of action distributions
            ('vf_r2', vfunc_r2, float),
            ('tdvf_r2', simplev_r2, float),
            ('dx', util.maxnorm(params0_P - self.policy.get_params()), float), # max parameter difference from last iteration
        ] + extra_print_fields + [
            ('tsamp', t_sample.dt, float), # time for sampling
            ('tadv', t_adv.dt + t_vf_fit.dt, float), # time for advantage computation
            ('tstep', t_step.dt, float), # time for step computation
            ('ttotal', self.total_time, float), # total time
        ]
        self.curr_iter += 1
        return fields


# Policy gradient update rules
# XXX Add step for sequential model
# XXX Use original TRPO for policy and value function
def TRPO(max_kl, damping, subsample_hvp_frac=.1, grad_stop_tol=1e-6, sequential_model=False):

    def trpo_step(policy, params0_P, obsfeat, a, adist, adv):
#        print 'Debug: TRPO trpo_step'
#        print 'Debug: obsfeat shape:', obsfeat.shape
#        print 'Debug: a shape:', a.shape
#        print 'Debug: adist shape:', adist.shape
#        print 'Debug: adv shape:', adv.shape
#        print 'Debug: params0_P shape:', params0_P.shape
#        print 'Debug: a:', a
#        print 'Debug: adist:', adist
        feed = (obsfeat, a, adist, util.standardized(adv))
        stdadv = util.standardized(adv)
        stepinfo = policy._ngstep(feed, max_kl=max_kl, damping=damping, subsample_hvp_frac=subsample_hvp_frac, grad_stop_tol=grad_stop_tol)
        return [
            ('dl', stepinfo.obj1 - stepinfo.obj0, float), # improvement of penalized objective
            ('kl', stepinfo.kl1, float), # kl cost of solution
            ('gnorm', stepinfo.gnorm, float), # gradient norm
            ('bt', stepinfo.bt, int), # number of backtracking steps
        ]

#    def trpo_seq_step(policy, params0_P, obsfeat, preva, h, c, a, adist, adv):
    def trpo_seq_step(policy, params0_P, obsfeat, mask, a, adist, adv):
        feed = (obsfeat, mask, policy.hidden, policy.cell, a, adist, util.standardized(adv))
        stdadv = util.standardized(adv)
        stepinfo = policy._ngstep(feed, max_kl=max_kl, damping=damping, subsample_hvp_frac=subsample_hvp_frac, grad_stop_tol=grad_stop_tol, sequential_model=True)
        return [
            ('dl', stepinfo.obj1 - stepinfo.obj0, float), # improvement of penalized objective
            ('kl', stepinfo.kl1, float), # kl cost of solution
            ('gnorm', stepinfo.gnorm, float), # gradient norm
            ('bt', stepinfo.bt, int), # number of backtracking steps
        ]

    if not sequential_model:
        return trpo_step
    else:
        return trpo_seq_step
#    return trop_step


import scipy.linalg
class LinearValueFunc(object):
    def __init__(self, l2reg=1e-5):
        self.w_Df = None
        self.l2reg = l2reg

    def _feat(self, obs_B_Do, t_B):
        assert obs_B_Do.ndim == 2 and t_B.ndim == 1
        B = obs_B_Do.shape[0]
        return np.concatenate([
                obs_B_Do,
                t_B[:,None]/100.,
                (t_B[:,None]/100.)**2,
                np.ones((B,1))
            ], axis=1)

    def evaluate(self, obs_B_Do, t_B):
        feat_Df = self._feat(obs_B_Do, t_B)
        if self.w_Df is None:
            self.w_Df = np.zeros(feat_Df.shape[1], dtype=obs_B_Do.dtype)
        return feat_Df.dot(self.w_Df)

    def fit(self, obs_B_Do, t_B, y_B):
        assert y_B.shape == (obs_B_Do.shape[0],)
        feat_B_Df = self._feat(obs_B_Do, t_B)
        self.w_Df = scipy.linalg.solve(
            feat_B_Df.T.dot(feat_B_Df) + self.l2reg*np.eye(feat_B_Df.shape[1]),
            feat_B_Df.T.dot(y_B),
            sym_pos=True)


class ValueFunc(nn.Model):
    def __init__(self, hidden_spec, obsfeat_space, enable_obsnorm, enable_vnorm, varscope_name, max_kl, damping, time_scale):
        self.hidden_spec = hidden_spec
        self.obsfeat_space = obsfeat_space
        self.enable_obsnorm = enable_obsnorm
        self.enable_vnorm = enable_vnorm
        self.max_kl = max_kl
        self.damping = damping
        self.time_scale = time_scale
        
        with nn.variable_scope(varscope_name) as self.__varscope:
            # Standardizers. Used outside, not in the computation graph.
            with nn.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim)
            with nn.variable_scope('vnorm'):
                self.vnorm = (nn.Standardizer if enable_vnorm else nn.NoOpStandardizer)(1)

            # Input observations
            obsfeat_B_Df = tensor.matrix(name='obsfeat_B_Df')
            t_B = tensor.vector(name='t_B')
            scaled_t_B = t_B * self.time_scale
            net_input = tensor.concatenate([obsfeat_B_Df, scaled_t_B[:,None]], axis=1)
            
            # Compute (normalized) value of states using a feedforward network
            with nn.variable_scope('hidden'):
                net = nn.FeedforwardNet(net_input, (self.obsfeat_space.dim + 1,), self.hidden_spec)
            with nn.variable_scope('out'):
                out_layer = nn.AffineLayer(net.output, net.output_shape, (1,), initializer=np.zeros((net.output_shape[0], 1)))
                assert out_layer.output_shape == (1,)
            val_B = out_layer.output[:,0]

        # Only code above this line is allowed to make trainable variables.
        param_vars = self.get_trainable_variables()

        self._evaluate_raw = thutil.function([obsfeat_B_Df, t_B], val_B)

        # Squared loss for fitting the value function
        target_val_B = tensor.vector(name='target_val_B')
        obj = -tensor.square(val_B - target_val_B).mean()
        objgrad_P = thutil.flatgrad(obj, param_vars)
        # KL divergence (as Gaussian) and its gradient
        old_val_B = tensor.vector(name='old_val_B')
        kl = tensor.square(old_val_B - val_B).mean()
        compute_obj_kl = thutil.function([obsfeat_B_Df, t_B, target_val_B, old_val_B], [obj, kl])
        compute_obj_kl_with_grad = thutil.function([obsfeat_B_Df, t_B, target_val_B, old_val_B], [obj, kl, objgrad_P])
        # KL Hessian-vector product
        x_P = tensor.vector(name='x')
        klgrad_P = thutil.flatgrad(kl, param_vars)
        hvp = thutil.function([obsfeat_B_Df, t_B, old_val_B, x_P], thutil.flatgrad((klgrad_P*x_P).sum(), param_vars))
        compute_kl_hvp = lambda _obsfeat_B_Df, _t_B, _target_val_B, _old_val_B, _x_P: hvp(_obsfeat_B_Df, _t_B, _old_val_B, _x_P)
        self._ngstep = optim.make_ngstep_func(self, compute_obj_kl, compute_obj_kl_with_grad, compute_kl_hvp)

    @property
    def varscope(self): return self.__varscope

    def evaluate(self, obs_B_Do, t_B):
        # ignores the time
        assert obs_B_Do.shape[0] == t_B.shape[0]
        return self.vnorm.unstandardize(self._evaluate_raw(self.obsnorm.standardize(obs_B_Do), t_B)[:,None])[:,0]

    def fit(self, obs_B_Do, t_B, y_B):
        # ignores the time
        assert obs_B_Do.shape[0] == t_B.shape[0] == y_B.shape[0]

        # Update normalization
        self.obsnorm.update(obs_B_Do)
        self.vnorm.update(y_B[:,None])

        # Take step
        sobs_B_Do = self.obsnorm.standardize(obs_B_Do)
        feed = (sobs_B_Do, t_B, self.vnorm.standardize(y_B[:,None])[:,0], self._evaluate_raw(sobs_B_Do, t_B))
        stepinfo = self._ngstep(feed, max_kl=self.max_kl, damping=self.damping)
        return [
            ('vf_dl', stepinfo.obj1 - stepinfo.obj0, float), # improvement of penalized objective
            ('vf_kl', stepinfo.kl1, float), # kl cost of solution
            ('vf_gnorm', stepinfo.gnorm, float), # gradient norm
            ('vf_bt', stepinfo.bt, int), # number of backtracking steps
        ]

    def update_obsnorm(self, obs_B_Do):
        self.obsnorm.update(obs_B_Do)



# XXX use original value function with memory
class SequentialValueFunc(nn.Model):
    def __init__(self, hidden_spec, obsfeat_space, time_step, enable_obsnorm, enable_vnorm, varscope_name, max_kl, damping, time_scale):
        self.hidden_spec = hidden_spec
        self.obsfeat_space = obsfeat_space
        self.time_step = time_step
        self.enable_obsnorm = enable_obsnorm
        self.enable_vnorm = enable_vnorm
        self.max_kl = max_kl
        self.damping = damping
        self.time_scale = time_scale

        self.hidden = None
        self.cell = None
        
        with nn.variable_scope(varscope_name) as self.__varscope:
            # Standardizers. Used outside, not in the computation graph.
            with nn.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim)
            with nn.variable_scope('vnorm'):
                self.vnorm = (nn.Standardizer if enable_vnorm else nn.NoOpStandardizer)(1)

            # Input observations
            obsfeat_BT_Df = tensor.matrix(name='obsfeat_BT_Df')
#            hidden_B_Dh = tensor.matrix(name='hidden_B_Dh')
#            cell_B_Dh = tensor.matrix(name='cell_B_Dh')
            t_BT = tensor.vector(name='t_BT')

            # get shape
            B, T, Df = obsfeat_BT_Df.shape[0], obsfeat_BT_Df.shape[1], obsfeat_BT_Df.shape[2]

            scaled_t_BT = t_BT * self.time_scale
            net_input = tensor.concatenate([obsfeat_BT_Df, scaled_t_BT[:,None]], axis=1)

            
            # Compute (normalized) value of states using a feedforward network
            with nn.variable_scope('hidden'):
                self.nn_net = nn.FeedforwardNet(net_input, (self.obsfeat_space.dim + 1,), self.hidden_spec)
            with nn.variable_scope('out'):
                self.out_layer = nn.AffineLayer(self.nn_net.output, self.nn_net.output_shape, (1,), initializer=np.zeros((self.nn_net.output_shape[0], 1)))
                assert self.out_layer.output_shape == (1,)
            val_BT = self.out_layer.output[:,0]

        # Only code above this line is allowed to make trainable variables.
        param_vars = self.get_trainable_variables()

        self._evaluate_raw = thutil.function([obsfeat_BT_Df, t_BT], val_BT)

        # Squared loss for fitting the value function
        target_val_BT = tensor.vector(name='target_val_BT')
        obj = -tensor.square(val_BT - target_val_BT).mean()
        objgrad_P = thutil.flatgrad(obj, param_vars)
        # KL divergence (as Gaussian) and its gradient
        old_val_BT = tensor.vector(name='old_val_BT')
        kl = tensor.square(old_val_BT - val_BT).mean()
        compute_obj_kl = thutil.function([obsfeat_BT_Df, t_BT, target_val_BT, old_val_BT], [obj, kl])
        compute_obj_kl_with_grad = thutil.function([obsfeat_BT_Df, t_BT, target_val_BT, old_val_BT], [obj, kl, objgrad_P])
        # KL Hessian-vector product
        x_P = tensor.vector(name='x')
        klgrad_P = thutil.flatgrad(kl, param_vars)
        hvp = thutil.function([obsfeat_BT_Df, t_BT, old_val_BT, x_P], thutil.flatgrad((klgrad_P*x_P).sum(), param_vars))
        compute_kl_hvp = lambda _obsfeat_BT_Df, _t_BT, _target_val_BT, _old_val_BT, _x_P: hvp(_obsfeat_BT_Df, _t_BT, _old_val_BT, _x_P)
        self._ngstep = optim.make_ngstep_func(self, compute_obj_kl, compute_obj_kl_with_grad, compute_kl_hvp)

    @property
    def varscope(self): return self.__varscope

    def evaluate(self, obs_BT_Do, t_BT):
        # ignores the time
        assert obs_BT_Do.shape[0] == t_BT.shape[0]
        return self.vnorm.unstandardize(self._evaluate_raw(self.obsnorm.standardize(obs_BT_Do), t_BT)[:,None])[:,0]

#    # XXX handle restart
#    def restart_hidden_cell(self, n_samples=None):
#        # only handle 1 lstm layer
#        self.lstm_net.layers[0].restart_hidden_cell(n_samples=n_samples)
#        self.hidden = self.lstm_net.layers[0].sample_hidden
#        self.cell = self.lstm_net.layers[0].sample_cell

#    def evaluate_restart(self, obs_B_T_Do, mask_B_T, t_B_T, restart=False, n_samples=None):
#        if restart:
#            self.restart_hidden_cell(n_samples=n_samples)
#        return self.evaluate(obs_B_T_Do, mask_B_T, t_B_T)
    

    def fit(self, obs_BT_Do, t_BT, y_BT):
        # ignores the time
        assert obs_BT_Do.shape[0] == t_BT.shape[0] == y_BT.shape[0]

        # Update normalization
        self.obsnorm.update(obs_BT_Do)
        self.vnorm.update(y_BT[:,None])

        # Take step
        sobs_BT_Do = self.obsnorm.standardize(obs_BT_Do)
#        #XXX restart lstm hidden cell
#        self.restart_hidden_cell(n_samples=obs_B_T_Do.shape[0])
        old_val_BT = self._evaluate_raw(sobs_BT_Do, t_BT)

#        #XXX restart lstm hidden cell again to clear memory
#        self.restart_hidden_cell()

        feed = (sobs_BT_Do, t_BT, self.vnorm.standardize(y_BT[:,None])[:,0], old_val_BT)

        stepinfo = self._ngstep(feed, max_kl=self.max_kl, damping=self.damping)
        return [
            ('vf_dl', stepinfo.obj1 - stepinfo.obj0, float), # improvement of penalized objective
            ('vf_kl', stepinfo.kl1, float), # kl cost of solution
            ('vf_gnorm', stepinfo.gnorm, float), # gradient norm
            ('vf_bt', stepinfo.bt, int), # number of backtracking steps
        ]

    def update_obsnorm(self, obs_BT_Do):
        self.obsnorm.update(obs_BT_Do)




class ConstantValueFunc(object):
    def __init__(self, max_timesteps):
        self.max_timesteps = max_timesteps
        self.v_T = np.zeros(max_timesteps)

    def evaluate(self, obs_B_Do, t_B):
        int_t_B = t_B.astype(int, copy=False)
        assert np.all(int_t_B == t_B) and np.all(0 <= t_B) and np.all(t_B < self.max_timesteps)
        return self.v_T[int_t_B].copy()

    def fit(self, obs_B_Do, t_B, y_B):
        int_t_B = t_B.astype(int, copy=False)
        assert np.all(int_t_B == t_B) and np.all(0 <= t_B) and np.all(t_B < self.max_timesteps)
        # Add up y values at various timesteps
        sum_T = np.zeros(self.max_timesteps)
        np.add.at(sum_T, int_t_B, y_B) # like sum_T[t_B] += y_B, but accumulates over duplicated time indices
        # Count number of values added at each timestep
        counts_T = np.zeros(self.max_timesteps)
        np.add.at(counts_T, int_t_B, 1)
        counts_T[counts_T < 1] = 1
        # Divide to get average
        self.v_T = sum_T / counts_T
        return []

    def update_obsnorm(self, obs_B_Do):
        pass
