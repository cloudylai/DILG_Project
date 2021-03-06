from . import nn, rl, util, RaggedArray, ContinuousSpace, FiniteSpace, optim, thutil
import math
import time
import numpy as np
from contextlib import contextmanager
import theano; from theano import tensor

from scipy.optimize import fmin_l_bfgs_b


class BehavioralCloningOptimizer(object):
    def __init__(self, mdp, policy, lr, batch_size, obsfeat_fn, ex_obs, ex_a, eval_sim_cfg, eval_freq, train_frac):
        self.mdp, self.policy, self.lr, self.batch_size, self.obsfeat_fn = mdp, policy, lr, batch_size, obsfeat_fn

        # Randomly split data into train/val
        assert ex_obs.shape[0] == ex_a.shape[0]
        num_examples = ex_obs.shape[0]
        num_train = int(train_frac * num_examples)
        shuffled_inds = np.random.permutation(num_examples)
        train_inds, val_inds = shuffled_inds[:num_train], shuffled_inds[num_train:]
        assert len(train_inds) >= 1 and len(val_inds) >= 1
        print ('{} training examples and {} validation examples'.format(len(train_inds), len(val_inds)))
        self.train_ex_obsfeat, self.train_ex_a = self.obsfeat_fn(ex_obs[train_inds]), ex_a[train_inds]
        self.val_ex_obsfeat, self.val_ex_a = self.obsfeat_fn(ex_obs[val_inds]), ex_a[val_inds]

        self.eval_sim_cfg = eval_sim_cfg
        self.eval_freq = eval_freq

        self.total_time = 0.
        self.curr_iter = 0

    def step(self):
        with util.Timer() as t_all:
            # Subsample expert transitions for SGD
            inds = np.random.choice(self.train_ex_obsfeat.shape[0], size=self.batch_size)
            batch_obsfeat_B_Do = self.train_ex_obsfeat[inds,:]
            batch_a_B_Da = self.train_ex_a[inds,:]
            # Take step
            loss = self.policy.step_bclone(batch_obsfeat_B_Do, batch_a_B_Da, self.lr)

            # Roll out trajectories when it's time to evaluate our policy
            val_loss = val_acc = trueret = avgr = ent = np.nan
            avglen = -1
            if self.eval_freq != 0 and self.curr_iter % self.eval_freq == 0:
                val_loss = self.policy.compute_bclone_loss(self.val_ex_obsfeat, self.val_ex_a)
                # Evaluate validation accuracy (independent of standard deviation)
                if isinstance(self.mdp.action_space, ContinuousSpace):
                    val_acc = -np.square(self.policy.compute_actiondist_mean(self.val_ex_obsfeat) - self.val_ex_a).sum(axis=1).mean()
                else:
                    assert self.val_ex_a.shape[1] == 1
                    # val_acc = (self.policy.sample_actions(self.val_ex_obsfeat)[1].argmax(axis=1) == self.val_ex_a[1]).mean()
                    val_acc = -val_loss # val accuracy doesn't seem too meaningful so just use this


        # Log
        self.total_time += t_all.dt
        fields = [
            ('iter', self.curr_iter, int),
            ('bcloss', loss, float), # supervised learning loss
            ('valloss', val_loss, float), # loss on validation set
            ('valacc', val_acc, float), # loss on validation set
            ('trueret', trueret, float), # true average return for this batch of trajectories
            ('avgr', avgr, float), # average reward encountered
            ('avglen', avglen, int), # average traj length
            ('ent', ent, float), # entropy of action distributions
            ('ttotal', self.total_time, float), # total time
        ]
        self.curr_iter += 1
        return fields


class TransitionClassifier(nn.Model):
    '''Reward/adversary for generative-adversarial training'''

    def __init__(self, obsfeat_space, action_space, hidden_spec, max_kl, adam_lr, adam_steps, ent_reg_weight, enable_inputnorm, include_time, time_scale, favor_zero_expert_reward, varscope_name):
        self.obsfeat_space, self.action_space = obsfeat_space, action_space
        self.hidden_spec = hidden_spec
        self.max_kl = max_kl
        self.adam_steps = adam_steps
        self.ent_reg_weight = ent_reg_weight; assert ent_reg_weight >= 0
        self.include_time = include_time
        self.time_scale = time_scale
        self.favor_zero_expert_reward = favor_zero_expert_reward

        with nn.variable_scope(varscope_name) as self.__varscope:
            # Map (s,a) pairs to classifier scores (log probabilities of classes)
            obsfeat_B_Df = tensor.matrix(name='obsfeat_B_Df')
            a_B_Da = tensor.matrix(name='a_B_Da', dtype=theano.config.floatX if self.action_space.storage_type == float else 'int64')
            t_B = tensor.vector(name='t_B')

            scaled_t_B = self.time_scale * t_B

            if isinstance(self.action_space, ContinuousSpace):
                # For a continuous action space, map observation-action pairs to a real number (reward)
                trans_B_Doa = tensor.concatenate([obsfeat_B_Df, a_B_Da], axis=1)
                trans_dim = self.obsfeat_space.dim + self.action_space.dim
                # Normalize
                with nn.variable_scope('inputnorm'):
                    self.inputnorm = (nn.Standardizer if enable_inputnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim + self.action_space.dim)
                normedtrans_B_Doa = self.inputnorm.standardize_expr(trans_B_Doa)
                if self.include_time:
                    net_input = tensor.concatenate([normedtrans_B_Doa, scaled_t_B[:,None]], axis=1)
                    net_input_dim = trans_dim + 1
                else:
                    net_input = normedtrans_B_Doa
                    net_input_dim = trans_dim
                # Compute scores
                with nn.variable_scope('hidden'):
                    net = nn.FeedforwardNet(net_input, (net_input_dim,), self.hidden_spec)
                with nn.variable_scope('out'):
                    out_layer = nn.AffineLayer(net.output, net.output_shape, (1,), initializer=np.zeros((net.output_shape[0], 1)))
                scores_B = out_layer.output[:,0]

            else:
                # For a finite action space, map observation observations to a vector of rewards

                # Normalize observations
                with nn.variable_scope('inputnorm'):
                    self.inputnorm = (nn.Standardizer if enable_inputnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim)
                normedobs_B_Df = self.inputnorm.standardize_expr(obsfeat_B_Df)
                if self.include_time:
                    net_input = tensor.concatenate([normedobs_B_Df, scaled_t_B[:,None]], axis=1)
                    net_input_dim = self.obsfeat_space.dim + 1
                else:
                    net_input = normedobs_B_Df
                    net_input_dim = self.obsfeat_space.dim
                # Compute scores
                with nn.variable_scope('hidden'):
                    net = nn.FeedforwardNet(net_input, (net_input_dim,), self.hidden_spec)
                with nn.variable_scope('out'):
                    out_layer = nn.AffineLayer(
                        net.output, net.output_shape, (self.action_space.size,),
                        initializer=np.zeros((net.output_shape[0], self.action_space.size)))
                scores_B = out_layer.output[tensor.arange(normedobs_B_Df.shape[0]), a_B_Da[:,0]]


        if self.include_time:
            self._compute_scores = thutil.function([obsfeat_B_Df, a_B_Da, t_B], scores_B) # scores define the conditional distribution p(label | (state,action))
        else:
            compute_scores_without_time = thutil.function([obsfeat_B_Df, a_B_Da], scores_B)
            self._compute_scores = lambda _obsfeat_B_Df, _a_B_Da, _t_B: compute_scores_without_time(_obsfeat_B_Df, _a_B_Da)

        if self.favor_zero_expert_reward:
            # 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            rewards_B = thutil.logsigmoid(scores_B)
        else:
            # 0 for non-expert-like states, goes to +inf for expert-like states
            # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
            # e.g. walking simulations that get cut off when the robot falls over
            rewards_B = -tensor.log(1.-tensor.nnet.sigmoid(scores_B))
        if self.include_time:
            self._compute_reward = thutil.function([obsfeat_B_Df, a_B_Da, t_B], rewards_B)
        else:
            compute_reward_without_time = thutil.function([obsfeat_B_Df, a_B_Da], rewards_B)
            self._compute_reward = lambda _obsfeat_B_Df, _a_B_Da, _t_B: compute_reward_without_time(_obsfeat_B_Df, _a_B_Da)

        param_vars = self.get_trainable_variables()

        # Logistic regression loss, regularized by negative entropy
        labels_B = tensor.vector(name='labels_B')
        weights_B = tensor.vector(name='weights_B')
        losses_B = thutil.sigmoid_cross_entropy_with_logits(scores_B, labels_B)
        ent_B = thutil.logit_bernoulli_entropy(scores_B)
        loss = ((losses_B - self.ent_reg_weight*ent_B)*weights_B).sum(axis=0)
        lossgrad_P = thutil.flatgrad(loss, param_vars)

        if self.include_time:
            self._adamstep = thutil.function(
                [obsfeat_B_Df, a_B_Da, t_B, labels_B, weights_B], loss,
                updates=thutil.adam(loss, param_vars, lr=adam_lr))
        else:
            adamstep_without_time = thutil.function(
                [obsfeat_B_Df, a_B_Da, labels_B, weights_B], loss,
                updates=thutil.adam(loss, param_vars, lr=adam_lr))
            self._adamstep = lambda _obsfeat_B_Df, _a_B_Da, _t_B, _labels_B, _weights_B: adamstep_without_time(_obsfeat_B_Df, _a_B_Da, _labels_B, _weights_B)

    @property
    def varscope(self): return self.__varscope

    def compute_reward(self, obsfeat_B_Df, a_B_Da, t_B):
        return self._compute_reward(obsfeat_B_Df, a_B_Da, t_B)

    def fit(self, obsfeat_B_Df, a_B_Da, t_B, exobs_Bex_Do, exa_Bex_Da, ext_Bex):
        # Transitions from the current policy go first, then transitions from the expert
        obsfeat_Ball_Df = np.concatenate([obsfeat_B_Df, exobs_Bex_Do])
        a_Ball_Da = np.concatenate([a_B_Da, exa_Bex_Da])
        t_Ball = np.concatenate([t_B, ext_Bex])

        # Update normalization
        self.update_inputnorm(obsfeat_Ball_Df, a_Ball_Da)

        B = obsfeat_B_Df.shape[0] # number of examples from the current policy
        Ball = obsfeat_Ball_Df.shape[0] # Ball - b = num examples from expert

        # Label expert as 1, current policy as 0
        labels_Ball = np.zeros(Ball, dtype=theano.config.floatX)
        labels_Ball[B:] = 1.

        # Evenly weight the loss terms for the expert and the current policy
        weights_Ball = np.zeros(Ball, dtype=theano.config.floatX)
        weights_Ball[:B] = 1./B
        weights_Ball[B:] = 1./(Ball - B); assert len(weights_Ball[B:]) == Ball-B

        # Optimize
        for _ in range(self.adam_steps):
            loss, kl, num_bt_steps = self._adamstep(obsfeat_Ball_Df, a_Ball_Da, t_Ball, labels_Ball, weights_Ball), None, 0
        
        # Evaluate
        scores_Ball = self._compute_scores(obsfeat_Ball_Df, a_Ball_Da, t_Ball); assert scores_Ball.shape == (Ball,)
        accuracy = .5 * (weights_Ball * ((scores_Ball < 0) == (labels_Ball == 0))).sum()
        accuracy = accuracy.astype(theano.config.floatX)
        accuracy_for_currpolicy = (scores_Ball[:B] <= 0).mean()
        accuracy_for_currpolicy = accuracy_for_currpolicy.astype(theano.config.floatX)
        accuracy_for_expert = (scores_Ball[B:] > 0).mean()
        accuracy_for_expert = accuracy_for_expert.astype(theano.config.floatX)
        assert np.allclose(accuracy, .5*(accuracy_for_currpolicy + accuracy_for_expert))

        return [
            ('rloss', loss, float), # reward function fitting loss
            ('racc', accuracy, float), # reward function accuracy
            ('raccpi', accuracy_for_currpolicy, float), # reward function accuracy
            ('raccex', accuracy_for_expert, float), # reward function accuracy
            ('rkl', kl, float),
            ('rbt', num_bt_steps, int),
            # ('rpnorm', util.maxnorm(self.get_params()), float),
            # ('snorm', util.maxnorm(scores_Ball), float),
        ]

    def update_inputnorm(self, obs_B_Do, a_B_Da):
        if isinstance(self.action_space, ContinuousSpace):
            self.inputnorm.update(np.concatenate([obs_B_Do, a_B_Da], axis=1))
        else:
            self.inputnorm.update(obs_B_Do)

    def plot(self, ax, idx1, idx2, range1, range2, n=100):
        assert len(range1) == len(range2) == 2 and idx1 != idx2
#        print 'Debug: TransitionClassifier:plot'
        x, y = np.mgrid[range1[0]:range1[1]:(n+0j), range2[0]:range2[1]:(n+0j)]
        # convert dtype to follow theano config
        x = x.astype(theano.config.floatX)
        y = y.astype(theano.config.floatX)
#        print 'Debug: x dtype:', x.dtype
#        print 'Debug: y dtype:', y.dtype
        if isinstance(self.action_space, ContinuousSpace):
            points_B_Doa = np.zeros((n*n, self.obsfeat_space.storage_size + self.action_space.storage_size), dtype=theano.config.floatX)
            points_B_Doa[:,idx1] = x.ravel()
            points_B_Doa[:,idx2] = y.ravel()
#            print 'Debug: points_b_doa dtype:', points_b_doa.dtype
            obsfeat_B_Df, a_B_Da = points_B_Doa[:,:self.obsfeat_space.storage_size], points_B_Doa[:,self.obsfeat_space.storage_size:]
            assert a_B_Da.shape[1] == self.action_space.storage_size
            #t_B = np.zeros(a_B_Da.shape[0]) # XXX make customizable
            t_B = np.zeros(a_B_Da.shape[0], dtype=theano.config.floatX) # XXX make customizable
            z = self.compute_reward(obsfeat_B_Df, a_B_Da, t_B).reshape(x.shape)
#            print 'Debug: t_B dtype:', t_B.dtype
#            print 'Debug: z dtype:', z.dtype
        else:
            obsfeat_B_Df = np.zeros((n*n, self.obsfeat_space.storage_size), dtype=theano.config.floatX)
            obsfeat_B_Df[:,idx1] = x.ravel()
            obsfeat_B_Df[:,idx2] = y.ravel()
#            print 'Debug: obsfeat_B_Df dtype:', obsfeat_B_Df.dtype
            # 
            #a_B_Da = np.zeros((obsfeat_B_Df.shape[0], 1), dtype=np.int32) # XXX make customizable
            a_B_Da = np.zeros((obsfeat_B_Df.shape[0], 1), dtype=np.int64) # XXX make customizable
            #t_B = np.zeros(a_B_Da.shape[0]) # XXX make customizable
            t_B = np.zeros(a_B_Da.shape[0], dtype=theano.config.floatX) # XXX make customizable
            z = self.compute_reward(obsfeat_B_Df, a_B_Da, t_B).reshape(x.shape)

#            print 'Debug: a_B_Da dtype:', a_B_Da.dtype
#            print 'Debug: t_B dtype:', t_B.dtype
#            print 'Debug: z dtype:', z.dtype
#        ax.pcolormesh(x, y, z, cmap='viridis')
#        ax.contour(x, y, z, levels=np.log(np.linspace(2., 3., 10)))
        # ax.contourf(x, y, z, levels=[np.log(2.), np.log(2.)+.5], alpha=.5) # high-reward region is highlighted
        return x, y, z






class SequentialTransitionClassifier(nn.Model):
    '''Reward/adversary for sequential generative-adversarial training'''

    def __init__(self, obsfeat_space, action_space, hidden_spec, max_kl, adam_lr, adam_steps, ent_reg_weight, time_step, enable_inputnorm, include_time, time_scale, favor_zero_expert_reward, varscope_name):
        self.obsfeat_space, self.action_space = obsfeat_space, action_space
        self.hidden_spec = hidden_spec
        self.max_kl = max_kl
        self.adam_steps = adam_steps
        self.ent_reg_weight = ent_reg_weight; assert ent_reg_weight >= 0
        self.time_step = time_step
        self.include_time = include_time
        self.time_scale = time_scale
        self.favor_zero_expert_reward = favor_zero_expert_reward

        self.hidden = None
        self.cell = None

        with nn.variable_scope(varscope_name) as self.__varscope:
            # Map (s,a) pairs to classifier scores (log probabilities of classes)
            obsfeat_B_T_Df = tensor.tensor3(name='obsfeat_B_T_Df')
            a_B_T_Da = tensor.tensor3(name='a_B_T_Da', dtype=theano.config.floatX if self.action_space.storage_type == float else 'int64')
            t_B_T = tensor.matrix(name='t_B_T')
            mask_B_T = tensor.matrix(name='mask_B_T', dtype='int64')
            hidden_B_Dh = tensor.matrix(name='hidden_B_Dh', dtype=theano.config.floatX)
            cell_B_Dh = tensor.matrix(name='cell_B_Dh', dtype=theano.config.floatX)

            scaled_t_B_T = self.time_scale * t_B_T

            # get shape
            B, T, Da = a_B_T_Da.shape[0], a_B_T_Da.shape[1], a_B_T_Da.shape[2]
            if isinstance(self.action_space, ContinuousSpace):
                # For a continuous action space, map observation-action pairs to a real number (reward)
                trans_B_T_Doa = tensor.concatenate([obsfeat_B_T_Df, a_B_T_Da], axis=2)
                trans_dim = self.obsfeat_space.dim + self.action_space.dim
                # Normalize
                with nn.variable_scope('inputnorm'):
#                    print("Debug: inputnorm")
#                    print("obsfeat space, action space, time step:", self.obsfeat_space.dim, self.action_space.dim, time_step)
                    self.inputnorm = (nn.SeqStandardizer if enable_inputnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim + self.action_space.dim)
                normedtrans_B_T_Doa = self.inputnorm.standardize_expr(trans_B_T_Doa)
                if self.include_time:
                    net_input = tensor.concatenate([normedtrans_B_T_Doa, scaled_t_B_T[:,None]], axis=2)
                    net_input_dim = trans_dim + 1
                else:
                    net_input = normedtrans_B_Doa
                    net_input_dim = trans_dim
                # Compute scores
                with nn.variable_scope('hidden'):
#                    net = nn.FeedforwardNet(net_input, (net_input_dim,), self.hidden_spec)
                    self.lstm_net = nn.LSTMNet(net_input, mask_B_T, hidden_B_Dh, cell_B_Dh, (net_input_dim,), self.hidden_spec)
                out_B_T_Dh = self.lstm_net.output
                # reshape: (B, T, ...) => (B*T, ...)
                out_BT_Dh = tensor.reshape(out_B_T_Dh, (B*T, -1), ndim=2)
                with nn.variable_scope('out'):
                    self.out_layer = nn.AffineLayer(out_BT_Dh, self.lstm_net.output_shape, (1,), initializer=np.zeros((self.lstm_net.output_shape[0], 1)))
                # XXX use scores of all timesteps as scores
                last_scores_BT = self.out_layer.output

            else:
                # For a finite action space, map observation-action pairs to a real number (reward)
                # Normalize observations
                with nn.variable_scope('inputnorm'):
#                    print("Debug: inputnorm")
#                    print("obsfeat_sapce, time_step:", self.obsfeat_space.dim, self.time_step)
                    self.inputnorm = (nn.SeqStandardizer if enable_inputnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim)
                normedobs_B_T_Df = self.inputnorm.standardize_expr(obsfeat_B_T_Df)
                if self.include_time:
                    net_input = tensor.concatenate([normedobs_B_T_Df, scaled_t_B_T[:,None]], axis=2)
                    net_input_dim = self.obsfeat_space.dim + 1
                else:
                    net_input = normedobs_B_T_Df
                    net_input_dim = self.obsfeat_space.dim
                # Compute scores
                with nn.variable_scope('hidden'):
#                    net = nn.FeedforwardNet(net_input, (net_input_dim,), self.hidden_spec)
                    self.lstm_net = nn.LSTMNet(net_input, mask_B_T, hidden_B_Dh, cell_B_Dh, (net_input_dim,), self.hidden_spec)
                out_B_T_Dh = self.lstm_net.output
                # reshape: (B,T, ...) => (B*T, ...)
                out_BT_Dh = tensor.reshape(out_B_T_Dh, (B*T, -1), ndim=2)
                a_BT_Da = tensor.reshape(a_B_T_Da, (B*T, -1), ndim=2)
                with nn.variable_scope('out'):
                    self.out_layer = nn.AffineLayer(
                        out_BT_Dh, self.lstm_net.output_shape, (self.action_space.size,),
                        initializer=np.zeros((self.lstm_net.output_shape[0], self.action_space.size)))
#                scores_B = self.out_layer.output[tensor.arange(normedobs_B_T_Df.shape[0]), a_B_T_Da[:,:,0]]
                # XXX use the scores of all timesteps as scores
                last_scores_BT = self.out_layer.output[tensor.arange(out_BT_Dh.shape[0]),  a_BT_Da[:,0]]

        # TODO use monto-carlo sampling to compute the score of each time step
        # by now, just copy the scores
        scores_BT = last_scores_BT
        # reshape: (B*T) => (B, T)
        scores_B_T = tensor.reshape(scores_BT, (B, T), ndim=2)

        if self.include_time:
            self._compute_scores = thutil.function([obsfeat_B_T_Df, mask_B_T, hidden_B_Dh, cell_B_Dh, a_B_T_Da, t_B_T], scores_B_T) # scores define the conditional distribution p(label | (state,action))
        else:
            compute_scores_without_time = thutil.function([obsfeat_B_T_Df, mask_B_T, hidden_B_Dh, cell_B_Dh, a_B_T_Da], scores_B_T)
            self._compute_scores = lambda _obsfeat_B_T_Df, _mask_B_T, _hidden_B_Dh, _cell_B_Dh, _a_B_T_Da, _t_B_T: compute_scores_without_time(_obsfeat_B_T_Df, _mask_B_T, _hidden_B_Dh, _cell_B_Dh, _a_B_T_Da)

        if self.favor_zero_expert_reward:
            # 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            rewards_B_T = thutil.logsigmoid(scores_B_T)
        else:
            # 0 for non-expert-like states, goes to +inf for expert-like states
            # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
            # e.g. walking simulations that get cut off when the robot falls over
            rewards_B_T = -tensor.log(1.-tensor.nnet.sigmoid(scores_B_T))
        if self.include_time:
            self._compute_reward = thutil.function([obsfeat_B_T_Df, mask_B_T, hidden_B_Dh, cell_B_Dh, a_B_T_Da, t_B_T], rewards_B_T)
        else:
            compute_reward_without_time = thutil.function([obsfeat_B_T_Df, mask_B_T, hidden_B_Dh, cell_B_Dh, a_B_T_Da], rewards_B_T)
            self._compute_reward = lambda _obsfeat_B_T_Df, _mask_B_T, _hidden_B_Dh, _cell_B_Dh, _a_B_T_Da, _t_B_T: compute_reward_without_time(_obsfeat_B_T_Df, _mask_B_T, _hidden_B_Dh, _cell_B_Dh, _a_B_T_Da)

        param_vars = self.get_trainable_variables()

        # TODO use sequential model inputs 
        # Logistic regression loss, regularized by negative entropy
        labels_BT = tensor.vector(name='labels_BT')
        weights_BT = tensor.vector(name='weights_BT')
        losses_BT = thutil.sigmoid_cross_entropy_with_logits(scores_BT, labels_BT)
        ent_BT = thutil.logit_bernoulli_entropy(scores_BT)
        loss = ((losses_BT - self.ent_reg_weight*ent_BT)*weights_BT).sum(axis=0)
        lossgrad_P = thutil.flatgrad(loss, param_vars)

        if self.include_time:
            self._adamstep = thutil.function(
                [obsfeat_B_T_Df, mask_B_T, hidden_B_Dh, cell_B_Dh, a_B_T_Da, t_B_T, labels_BT, weights_BT], loss,
                updates=thutil.adam(loss, param_vars, lr=adam_lr))
        else:
            adamstep_without_time = thutil.function(
                [obsfeat_B_T_Df, mask_B_T, hidden_B_Dh, cell_B_Dh, a_B_T_Da, labels_BT, weights_BT], loss,
                updates=thutil.adam(loss, param_vars, lr=adam_lr))
            self._adamstep = lambda _obsfeat_B_T_Df, _mask_B_T, _hidden_B_Dh, _cell_B_Dh, _a_B_T_Da, _t_B_T, _labels_BT, _weights_BT: adamstep_without_time(_obsfeat_B_T_Df, _mask_B_T, _hidden_B_Dh, _cell_B_Dh, _a_B_T_Da, _labels_BT, _weights_BT)

    @property
    def varscope(self): return self.__varscope

    def compute_reward(self, obsfeat_B_T_Df, mask_B_T, hidden_B_Dh, cell_B_Dh, a_B_T_Da, t_B_T):
        return self._compute_reward(obsfeat_B_T_Df, mask_B_T, hidden_B_Dh, cell_B_Dh, a_B_T_Da, t_B_T)

    # XXX handle restart
    def restart_hidden_cell(self, n_samples=None):
        # only handle 1 lstm layer
        self.lstm_net.layers[0].restart_hidden_cell(n_samples=n_samples)
        self.hidden = self.lstm_net.layers[0].sample_hidden
        self.cell = self.lstm_net.layers[0].sample_cell



    # XXX handle restart
    def compute_reward_restart(self, obsfeat_B_T_Df, mask_B_T, a_B_T_Da, t_B_T, restart, n_samples=None):
        if restart:
            self.restart_hidden_cell(n_samples=n_samples)
        return self.compute_reward(obsfeat_B_T_Df, mask_B_T, self.hidden, self.cell, a_B_T_Da, t_B_T)


    def fit(self, obsfeat_B_T_Df, mask_B_T, a_B_T_Da, t_B_T, exobs_Bex_T_Do, exa_Bex_T_Da, ext_Bex_T):
        # Transitions from the current policy go first, then transitions from the expert
#        print 'Debug: TransitionClassifier fit'
#        print 'Debug: obsfeat_B_Df dtype:', obsfeat_B_Df.dtype    # float32
#        print 'Debug: a_B_Da dtype:', a_B_Da.dtype    # int64
#        print 'Debug: t_B dtype:', t_B.dtype    # float32
#        print 'Debug: exobs_Bex_Do dtype:', exobs_Bex_Do.dtype    # float32
#        print 'Debug: exa_Bex_Da dtype:', exa_Bex_Da.dtype    # int64
#        print 'Debug: ext_Bex dtype:', ext_Bex.dtype    # float32

#        print("Debug: self time_step:", self.time_step)
#        print("Debug: obsfeat_B_T_Df shape:", obsfeat_B_T_Df.shape)
#        print("Debug: mask_B_T shape:", mask_B_T.shape)
#        print("Debug: a_B_T_Da shape:", a_B_T_Da.shape)
#        print("Debug: t_B_T shape:", t_B_T.shape)
#        print("Debug: exobs_Bex_T_Do shape:", exobs_Bex_T_Do.shape)
#        print("Debug: exobs_Bex_T_Da shape:", exa_Bex_T_Da.shape)
#        print("Debug: ext_Bex_T shape:", ext_Bex_T.shape)
        
        obsfeat_Ball_T_Df = np.concatenate([obsfeat_B_T_Df, exobs_Bex_T_Do])
        
        B = obsfeat_B_T_Df.shape[0] # number of examples from the current policy
        Ball = obsfeat_Ball_T_Df.shape[0] # Ball - b = num examples from expert
        T = obsfeat_Ball_T_Df.shape[1]

        mask_Ball_T = np.concatenate([mask_B_T, np.ones((Ball - B, T), dtype='int64')])
        a_Ball_T_Da = np.concatenate([a_B_T_Da, exa_Bex_T_Da])
        t_Ball_T = np.concatenate([t_B_T, ext_Bex_T])

#        print 'Debug: obsfeat_Ball_Df dtype:', obsfeat_Ball_Df.dtype    # float32
#        print 'Debug: a_Ball_Da dtype:', a_Ball_Da.dtype    # int64
#        print 'Debug: t_Ball dtype:', t_Ball.dtype    # float32

        # Update normalization
        self.update_inputnorm(obsfeat_Ball_T_Df, a_Ball_T_Da)

#        # reshape: (B, T, ...) => (B*T, ...)
#        obsfeat_BallT_Df = np.reshape(obsfeat_Ball_T_Df, (Ball*T, -1))
#        a_BallT_Da = np.reshape(a_Ball_T_Da, (Ball*T, -1))
#        t_BallT = np.reshape(t_Ball_T, (Ball*T,))
        
        # Label expert as 1, current policy as 0
        labels_BallT = np.zeros(Ball*T, dtype=theano.config.floatX)
#        print("Debug: labels_BallT shape:", labels_BallT.shape)
        labels_BallT[B*T:] = 1.

#        print 'Debug: labels_BallT dtype:', labels_BallT.dtype
        # Evenly weight the loss terms for the expert and the current policy
        weights_BallT = np.zeros(Ball*T, dtype=theano.config.floatX)
#        print("Debug: weights_BallT shape:", weights_BallT.shape)
        weights_BallT[:B*T] = 1./(B*T)
        weights_BallT[B*T:] = 1./((Ball - B)*T); assert len(weights_BallT[B*T:]) == (Ball-B)*T

#        print 'Debug: weights_Ball dtype:', weights_Ball.dtype
        # Optimize

        # XXX restart lstm hidden cell
        self.restart_hidden_cell(n_samples=Ball)

        for _ in range(self.adam_steps):
            loss, kl, num_bt_steps = self._adamstep(obsfeat_Ball_T_Df, mask_Ball_T, self.hidden, self.cell, a_Ball_T_Da, t_Ball_T, labels_BallT, weights_BallT), None, 0
        
#        print 'Debug: loss type:', type(loss)
#        print 'Debug: kl type:', type(kl)
#        print 'Debug: num_bt_steps type:', type(num_bt_steps)
#        print 'Debug: labels_BallT dtype:', labels_BallT.dtype
#        print 'Debug: weights_BallT dtype:', weights_BallT.dtype
        # Evaluate
        # XXX restart lstm hidden cell
        self.restart_hidden_cell()
        scores_Ball_T = self._compute_scores(obsfeat_Ball_T_Df, mask_Ball_T, self.hidden, self.cell, a_Ball_T_Da, t_Ball_T); assert scores_Ball_T.shape == (Ball,T)
        scores_BallT = np.reshape(scores_Ball_T, (Ball*T,))
        accuracy = .5 * (weights_BallT * ((scores_BallT < 0) == (labels_BallT == 0))).sum()
        accuracy = accuracy.astype(theano.config.floatX)
        accuracy_for_currpolicy = (scores_BallT[:B*T] <= 0).mean()
        accuracy_for_currpolicy = accuracy_for_currpolicy.astype(theano.config.floatX)
        accuracy_for_expert = (scores_BallT[B*T:] > 0).mean()
        accuracy_for_expert = accuracy_for_expert.astype(theano.config.floatX)
#        print("Debug: accuracy:", accuracy)
#        print("Debug: accuracy_for_currpolicy:", accuracy_for_currpolicy)
#        print("Debug: accuracy_for_expert:", accuracy_for_expert)
        assert np.allclose(accuracy, .5*(accuracy_for_currpolicy + accuracy_for_expert))

#        print 'Debug: scores_Ball dtype:', scores_Ball.dtype
#        print 'Debug: accuracy type:', type(accuracy)
#        print 'Debug: accuracy_for_currpolicy type:', type(accuracy_for_currpolicy)
#        print 'Debug: accuracy_for_expert type:', type(accuracy_for_expert)
        return [
            ('rloss', loss, float), # reward function fitting loss
            ('racc', accuracy, float), # reward function accuracy
            ('raccpi', accuracy_for_currpolicy, float), # reward function accuracy
            ('raccex', accuracy_for_expert, float), # reward function accuracy
            ('rkl', kl, float),
            ('rbt', num_bt_steps, int),
            # ('rpnorm', util.maxnorm(self.get_params()), float),
            # ('snorm', util.maxnorm(scores_Ball), float),
        ]

    def update_inputnorm(self, obs_B_T_Do, a_B_T_Da):
        if isinstance(self.action_space, ContinuousSpace):
            self.inputnorm.update(np.concatenate([obs_B_T_Do, a_B_T_Da], axis=2))
        else:
            self.inputnorm.update(obs_B_T_Do)


    # TODO check the propose of this code  
    def plot(self, ax, idx1, idx2, range1, range2, traj_length, n=10):
        assert len(range1) == len(range2) == 2 and idx1 != idx2
        # set a small value by now
        T = 10
#        print 'Debug: TransitionClassifier:plot'
        x, y = np.mgrid[range1[0]:range1[1]:(n*T+0j), range2[0]:range2[1]:(n*T+0j)]
        # convert dtype to follow theano config
        x = x.astype(theano.config.floatX)
        y = y.astype(theano.config.floatX)
#        print 'Debug: x dtype:', x.dtype
#        print 'Debug: y dtype:', y.dtype
        if isinstance(self.action_space, ContinuousSpace):
            points_BTT_Doa = np.zeros((n*T*n*T, self.obsfeat_space.storage_size + self.action_space.storage_size), dtype=theano.config.floatX)
            points_BTT_Doa[:,idx1] = x.ravel()
            points_BTT_Doa[:,idx2] = y.ravel()
#            print 'Debug: points_b_doa dtype:', points_b_doa.dtype
            obsfeat_BTT_Df, a_BTT_Da = points_BTT_Doa[:,:self.obsfeat_space.storage_size], points_BTT_Doa[:,self.obsfeat_space.storage_size:]
            assert a_BTT_Da.shape[1] == self.action_space.storage_size
            #t_B = np.zeros(a_B_Da.shape[0]) # XXX make customizable
            t_BTT = np.zeros(a_BTT_Da.shape[0], dtype=theano.config.floatX) # XXX make customizable
            mask_BTT = np.ones(a_BTT_Da.shape[0], dtype='int64') # XXX  make customizable
            # reshape: (B*T*T, ...) => (B*T, T, ...)
            obsfeat_BT_T_Df = np.reshape(obsfeat_BTT_Df, (n*n*T, T, -1))
            a_BT_T_Da = np.reshape(a_BTT_Da, (n*n*T, T, -1))
            t_BT_T = np.reshape(t_BTT, (n*n*T, T))
            mask_BT_T = np.reshape(mask_BTT, (n*n*T, T))

            # XXX restart lstm hidden and cell
            self.restart_hidden_cell(n_samples=n*n*T)

            z = self.compute_reward(obsfeat_BT_T_Df, mask_BT_T, self.hidden, self.cell, a_BT_T_Da, t_BT_T).reshape(x.shape)
#            print 'Debug: t_B dtype:', t_B.dtype
#            print 'Debug: z dtype:', z.dtype
        else:
            obsfeat_BTT_Df = np.zeros((n*T*n*T, self.obsfeat_space.storage_size), dtype=theano.config.floatX)
            obsfeat_BTT_Df[:,idx1] = x.ravel()
            obsfeat_BTT_Df[:,idx2] = y.ravel()
#            print 'Debug: obsfeat_B_Df dtype:', obsfeat_B_Df.dtype
            #a_B_Da = np.zeros((obsfeat_B_Df.shape[0], 1), dtype=np.int32) # XXX make customizable
            a_BTT_Da = np.zeros((obsfeat_BTT_Df.shape[0], 1), dtype=np.int64) # XXX make customizable
            #t_B = np.zeros(a_B_Da.shape[0]) # XXX make customizable
            t_BTT = np.zeros(a_BTT_Da.shape[0], dtype=theano.config.floatX) # XXX make customizable
            mask_BTT = np.ones(a_BTT_Da.shape[0], dtype='int64')
            # reshape: (B*T*T, ...) => (B*T, T, ...)
            obsfeat_BT_T_Df = np.reshape(obsfeat_BTT_Df, (n*n*T, T, -1))
            a_BT_T_Da = np.reshape(a_BTT_Da, (n*n*T, T, -1))
            t_BT_T = np.reshape(t_BTT, (n*n*T, T))
            mask_BT_T = np.reshape(mask_BTT, (n*n*T, T))

            # XXX restart lstm hidden and cell
            self.restart_hidden_cell(n_samples=n*n*T)

            z = self.compute_reward(obsfeat_BT_T_Df, mask_BT_T, self.hidden, self.cell, a_BT_T_Da, t_BT_T).reshape(x.shape)

#            print 'Debug: a_B_Da dtype:', a_B_Da.dtype
#            print 'Debug: t_B dtype:', t_B.dtype
#            print 'Debug: z dtype:', z.dtype
#        ax.pcolormesh(x, y, z, cmap='viridis')
#        ax.contour(x, y, z, levels=np.log(np.linspace(2., 3., 10)))
        # ax.contourf(x, y, z, levels=[np.log(2.), np.log(2.)+.5], alpha=.5) # high-reward region is highlighted
        return x, y, z







class LinearReward(object):
    # things to keep in mind
    # - continuous vs discrete actions
    # - simplex or l2 ball
    # - input norm
    # - shifting so that 0 == expert or 0 == non-expert

    def __init__(self,
            obsfeat_space, action_space,
            mode, enable_inputnorm, favor_zero_expert_reward,
            include_time,
            time_scale,
            exobs_Bex_Do, exa_Bex_Da, ext_Bex,
            sqscale=.01,
            quadratic_features=False):

        self.obsfeat_space, self.action_space = obsfeat_space, action_space
        assert mode in ['l2ball', 'simplex']
        print ('Linear reward function type: {}'.format(mode))
        self.simplex = mode == 'simplex'
        self.favor_zero_expert_reward = favor_zero_expert_reward
        self.include_time = include_time
        self.time_scale = time_scale
        self.sqscale = sqscale
        self.quadratic_features = quadratic_features
        self.exobs_Bex_Do, self.exa_Bex_Da, self.ext_Bex = exobs_Bex_Do, exa_Bex_Da, ext_Bex
        with nn.variable_scope('inputnorm'):
            # Standardize both observations and actions if actions are continuous
            # otherwise standardize observations only.
            self.inputnorm = (nn.Standardizer if enable_inputnorm else nn.NoOpStandardizer)(
                (obsfeat_space.dim + action_space.dim) if isinstance(action_space, ContinuousSpace)
                    else obsfeat_space.dim)
            self.inputnorm_updated = False
        self.update_inputnorm(self.exobs_Bex_Do, self.exa_Bex_Da) # pre-standardize with expert data

        # Expert feature expectations
        self.expert_feat_Df = self._compute_featexp(self.exobs_Bex_Do, self.exa_Bex_Da, self.ext_Bex)
        # The current reward function
        feat_dim = self.expert_feat_Df.shape[0]
        print ('Linear reward: {} features'.format(feat_dim))
        if self.simplex:
            # widx is the index of the most discriminative reward function
            self.widx = np.random.randint(feat_dim)
        else:
            # w is a weight vector
            self.w = np.random.randn(feat_dim)
            self.w /= np.linalg.norm(self.w) + 1e-8

        self.reward_bound = 0.

    def _featurize(self, obsfeat_B_Do, a_B_Da, t_B):
        assert self.inputnorm_updated
        assert obsfeat_B_Do.shape[0] == a_B_Da.shape[0] == t_B.shape[0]
        B = obsfeat_B_Do.shape[0]

        # Standardize observations and actions
        if isinstance(self.action_space, ContinuousSpace):
            trans_B_Doa = self.inputnorm.standardize(np.concatenate([obsfeat_B_Do, a_B_Da], axis=1))
            obsfeat_B_Do, a_B_Da = trans_B_Doa[:,:obsfeat_B_Do.shape[1]], trans_B_Doa[:,obsfeat_B_Do.shape[1]:]
            assert obsfeat_B_Do.shape[1] == self.obsfeat_space.dim and a_B_Da.shape[1] == self.action_space.dim
        else:
            assert a_B_Da.shape[1] == 1 and np.allclose(a_B_Da, a_B_Da.astype(int)), 'actions must all be ints'
            obsfeat_B_Do = self.inputnorm.standardize(obsfeat_B_Do)

        # Concatenate with other stuff to get final features
        scaledt_B_1 = t_B[:,None]*self.time_scale
        if isinstance(self.action_space, ContinuousSpace):
            if self.quadratic_features:
                feat_cols = [obsfeat_B_Do, a_B_Da]
                if self.include_time:
                    feat_cols.extend([scaledt_B_1])
                feat = np.concatenate(feat_cols, axis=1)
                quadfeat = (feat[:,:,None] * feat[:,None,:]).reshape((B,-1))
                feat_B_Df = np.concatenate([feat,quadfeat,np.ones((B,1))], axis=1)
            else:
                feat_cols = [obsfeat_B_Do, a_B_Da, (self.sqscale*obsfeat_B_Do)**2, (self.sqscale*a_B_Da)**2]
                if self.include_time:
                    feat_cols.extend([scaledt_B_1, scaledt_B_1**2, scaledt_B_1**3])
                feat_cols.append(np.ones((B,1)))
                feat_B_Df = np.concatenate(feat_cols, axis=1)

        else:
            assert not self.quadratic_features
            # Observation-only features
            obsonly_feat_cols = [obsfeat_B_Do, (.01*obsfeat_B_Do)**2]
            if self.include_time:
                obsonly_feat_cols.extend([scaledt_B_1, scaledt_B_1**2, scaledt_B_1**3])
            obsonly_feat_B_f = np.concatenate(obsonly_feat_cols, axis=1)

            # To get features that include actions, we'll have blocks of obs-only features,
            # one block for each action.
            assert a_B_Da.shape[1] == 1
            action_inds = [np.flatnonzero(a_B_Da[:,0] == a) for a in xrange(self.action_space.size)]
            assert sum(len(inds) for inds in action_inds) == B
            action_block_size = obsonly_feat_B_f.shape[1]
            # Place obs features into their appropriate blocks
            blocked_feat_B_Dfm1 = np.zeros((obsonly_feat_B_f.shape[0], action_block_size*self.action_space.size))
            for a in range(self.action_space.size):
                blocked_feat_B_Dfm1[action_inds[a],a*action_block_size:(a+1)*action_block_size] = obsonly_feat_B_f[action_inds[a],:]
            assert np.isfinite(blocked_feat_B_Dfm1).all()
            feat_B_Df = np.concatenate([blocked_feat_B_Dfm1, np.ones((B,1))], axis=1)

        if self.simplex:
            feat_B_Df = np.concatenate([feat_B_Df, -feat_B_Df], axis=1)

        assert feat_B_Df.ndim == 2 and feat_B_Df.shape[0] == B
        return feat_B_Df


    def _compute_featexp(self, obsfeat_B_Do, a_B_Da, t_B):
        return self._featurize(obsfeat_B_Do, a_B_Da, t_B).mean(axis=0)


    def fit(self, obsfeat_B_Do, a_B_Da, t_B, _unused_exobs_Bex_Do, _unused_exa_Bex_Da, _unused_ext_Bex):
        # Ignore expert data inputs here, we'll use the one provided in the constructor.

        # Current feature expectations
        curr_feat_Df = self._compute_featexp(obsfeat_B_Do, a_B_Da, t_B)

        # Compute adversary reward
        if self.simplex:
            v = curr_feat_Df - self.expert_feat_Df
            self.widx = np.argmin(v)
            return [('vmin', v.min(), float)]
        else:
            self.w = self.expert_feat_Df - curr_feat_Df
            l2 = np.linalg.norm(self.w)
            self.w /= l2 + 1e-8
            return [('l2', l2, float)]


    def compute_reward(self, obsfeat_B_Do, a_B_Da, t_B):
        feat_B_Df = self._featurize(obsfeat_B_Do, a_B_Da, t_B)
        r_B = (feat_B_Df[:,self.widx] if self.simplex else feat_B_Df.dot(self.w)) / float(feat_B_Df.shape[1])
        assert r_B.shape == (obsfeat_B_Do.shape[0],)

        if self.favor_zero_expert_reward:
            self.reward_bound = max(self.reward_bound, r_B.max())
        else:
            self.reward_bound = min(self.reward_bound, r_B.min())
        shifted_r_B = r_B - self.reward_bound
        if self.favor_zero_expert_reward:
            assert (shifted_r_B <= 0).all()
        else:
            assert (shifted_r_B >= 0).all()

        return shifted_r_B

    def update_inputnorm(self, obs_B_Do, a_B_Da):
        if isinstance(self.action_space, ContinuousSpace):
            self.inputnorm.update(np.concatenate([obs_B_Do, a_B_Da], axis=1))
        else:
            self.inputnorm.update(obs_B_Do)
        self.inputnorm_updated = True


class ImitationOptimizer(object):
    def __init__(self, mdp, discount, lam, policy, sim_cfg, step_func, reward_func, value_func, policy_obsfeat_fn, reward_obsfeat_fn, policy_ent_reg, ex_obs, ex_a, ex_t):
        self.mdp, self.discount, self.lam, self.policy = mdp, discount, lam, policy
        self.sim_cfg = sim_cfg
        self.step_func = step_func
        self.reward_func = reward_func
        self.value_func = value_func
        # assert value_func is not None, 'not tested'
        self.policy_obsfeat_fn = policy_obsfeat_fn
        self.reward_obsfeat_fn = reward_obsfeat_fn
        self.policy_ent_reg = policy_ent_reg
        util.header('Policy entropy regularization: {}'.format(self.policy_ent_reg))

        assert ex_obs.ndim == ex_a.ndim == 2 and ex_t.ndim == 1 and ex_obs.shape[0] == ex_a.shape[0] == ex_t.shape[0]
        self.ex_pobsfeat, self.ex_robsfeat, self.ex_a, self.ex_t = policy_obsfeat_fn(ex_obs), reward_obsfeat_fn(ex_obs), ex_a, ex_t

        self.total_num_trajs = 0
        self.total_num_sa = 0
        self.total_time = 0.
        self.curr_iter = 0
        self.last_sampbatch = None # for outside access for debugging

    def step(self):
        with util.Timer() as t_all:

            # Sample trajectories using current policy
            # print 'Sampling'
            with util.Timer() as t_sample:
                sampbatch = self.mdp.sim_mp(
                    policy_fn=lambda obsfeat_B_Df: self.policy.sample_actions(obsfeat_B_Df),
                    obsfeat_fn=self.policy_obsfeat_fn,
                    cfg=self.sim_cfg)
                samp_pobsfeat = sampbatch.obsfeat
                self.last_sampbatch = sampbatch

            # Compute baseline / advantages
            # print 'Computing advantages'
            with util.Timer() as t_adv:
                # Compute observation features for reward input
                samp_robsfeat_stacked = self.reward_obsfeat_fn(sampbatch.obs.stacked)
                # Reward is computed wrt current reward function
                # TODO: normalize rewards
                rcurr_stacked = self.reward_func.compute_reward(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked)
                assert rcurr_stacked.shape == (samp_robsfeat_stacked.shape[0],)

                # If we're regularizing the policy, add negative log probabilities to the rewards
                # Intuitively, the policy gets a bonus for being less certain of its actions
                orig_rcurr_stacked = rcurr_stacked.copy()
                if self.policy_ent_reg is not None and self.policy_ent_reg != 0:
                    assert self.policy_ent_reg > 0
                    # XXX probably faster to compute this from sampbatch.adist instead
                    actionlogprobs_B = self.policy.compute_action_logprobs(samp_pobsfeat.stacked, sampbatch.a.stacked)
                    policyentbonus_B = -self.policy_ent_reg * actionlogprobs_B
                    policyentbonus_B = policyentbonus_B.astype(theano.config.floatX)
                    rcurr_stacked += policyentbonus_B
                else:
                    policyentbonus_B = np.zeros_like(rcurr_stacked)

                rcurr = RaggedArray(rcurr_stacked, lengths=sampbatch.r.lengths)

                # Compute advantages using these rewards
                advantages, qvals, vfunc_r2, simplev_r2 = rl.compute_advantage(
                    rcurr, samp_pobsfeat, sampbatch.time, self.value_func, self.discount, self.lam)

            # Take a step
            # print 'Fitting policy'
            with util.Timer() as t_step:
                params0_P = self.policy.get_params()
                step_print = self.step_func(
                    self.policy, params0_P,
                    samp_pobsfeat.stacked, sampbatch.a.stacked, sampbatch.adist.stacked,
                    advantages.stacked)
                self.policy.update_obsnorm(samp_pobsfeat.stacked)


            # Fit reward function
            # print 'Fitting reward'
            with util.Timer() as t_r_fit:
                if True:#self.curr_iter % 20 == 0:
                    # Subsample expert transitions to the same sample count for the policy
                    inds = np.random.choice(self.ex_robsfeat.shape[0], size=samp_pobsfeat.stacked.shape[0])
                    exbatch_robsfeat = self.ex_robsfeat[inds,:]
                    exbatch_pobsfeat = self.ex_pobsfeat[inds,:] # only used for logging
                    exbatch_a = self.ex_a[inds,:]
                    exbatch_t = self.ex_t[inds]
                    rfit_print = self.reward_func.fit(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked, exbatch_robsfeat, exbatch_a, exbatch_t)
                else:
                    rfit_print = []

            # Fit value function for next iteration
            # print 'Fitting value function'
            with util.Timer() as t_vf_fit:
                if self.value_func is not None:
                    # Recompute q vals # XXX: this is only necessary if fitting reward after policy
                    # qnew = qvals

                    # TODO: this should be a byproduct of reward fitting
                    rnew = RaggedArray(
                        self.reward_func.compute_reward(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked),
                        lengths=sampbatch.r.lengths)
                    qnew, _ = rl.compute_qvals(rnew, self.discount)
                    vfit_print = self.value_func.fit(samp_pobsfeat.stacked, sampbatch.time.stacked, qnew.stacked)
                else:
                    vfit_print = []

        # Log
        self.total_num_trajs += len(sampbatch)
        self.total_num_sa += sum(len(traj) for traj in sampbatch)
        self.total_time += t_all.dt
        fields = [
            ('iter', self.curr_iter, int),
            ('trueret', sampbatch.r.padded(fill=0.).sum(axis=1).mean(), float), # average return for this batch of trajectories
            ('iret', rcurr.padded(fill=0.).sum(axis=1).mean(), float), # average return on imitation reward
            ('avglen', int(np.mean([len(traj) for traj in sampbatch])), int), # average traj length
            ('ntrajs', self.total_num_trajs, int), # total number of trajs sampled over the course of training
            ('nsa', self.total_num_sa, int), # total number of state-action pairs sampled over the course of training
            ('ent', self.policy._compute_actiondist_entropy(sampbatch.adist.stacked).mean(), float), # entropy of action distributions
            ('vf_r2', vfunc_r2, float),
            ('tdvf_r2', simplev_r2, float),
            ('dx', util.maxnorm(params0_P - self.policy.get_params()), float), # max parameter difference from last iteration
        ] + step_print + vfit_print + rfit_print + [
            ('avgr', rcurr_stacked.mean(), float), # average regularized reward encountered
            ('avgunregr', orig_rcurr_stacked.mean(), float), # average unregularized reward
            ('avgpreg', policyentbonus_B.mean(), float), # average policy regularization
            # ('bcloss', -self.policy.compute_action_logprobs(exbatch_pobsfeat, exbatch_a).mean(), float), # negative log likelihood of expert actions
            # ('bcloss', np.square(self.policy.compute_actiondist_mean(exbatch_pobsfeat) - exbatch_a).sum(axis=1).mean(axis=0), float),
            ('tsamp', t_sample.dt, float), # time for sampling
            ('tadv', t_adv.dt + t_vf_fit.dt, float), # time for advantage computation
            ('tstep', t_step.dt, float), # time for step computation
            ('ttotal', self.total_time, float), # total time
        ]
        self.curr_iter += 1
        return fields





# XXX implement as original imitation except a sequential classifier
class SequentialImitationOptimizer(object):
    def __init__(self, mdp, discount, lam, policy, sim_cfg, step_func, reward_func, value_func, policy_obsfeat_fn, reward_obsfeat_fn, policy_ent_reg, ex_obs, ex_a, ex_t):
        self.mdp, self.discount, self.lam, self.policy = mdp, discount, lam, policy
        self.sim_cfg = sim_cfg
        self.step_func = step_func
        self.reward_func = reward_func
        self.value_func = value_func
        # assert value_func is not None, 'not tested'
        self.policy_obsfeat_fn = policy_obsfeat_fn
        self.reward_obsfeat_fn = reward_obsfeat_fn
        self.policy_ent_reg = policy_ent_reg
        util.header('Policy entropy regularization: {}'.format(self.policy_ent_reg))

        assert ex_obs.ndim == ex_a.ndim == 2 and ex_t.ndim == 1 and ex_obs.shape[0] == ex_a.shape[0] == ex_t.shape[0]
        assert sim_cfg.time_step > 0
        self.ex_pobsfeat, self.ex_robsfeat, self.ex_a, self.ex_t = policy_obsfeat_fn(ex_obs), reward_obsfeat_fn(ex_obs), ex_a, ex_t

        self.total_num_trajs = 0
        self.total_num_sa = 0
        self.total_time = 0.
        self.curr_iter = 0
        self.last_sampbatch = None # for outside access for debugging

    def step(self):
        with util.Timer() as t_all:

            # Sample trajectories using current policy
            # print 'Sampling'
#            print('Debug: Sampling')
            with util.Timer() as t_sample:
                st = time.time()
#                # XXX restart memory                
#                self.policy.restart_hidden_cell(n_samples=1)
                # XXX use original simulation
                sampbatch = self.mdp.sim_mp(
                    policy_fn=lambda obsfeat_B_Df: self.policy.sample_actions(obsfeat_B_Df),
                    obsfeat_fn=self.policy_obsfeat_fn,
                    cfg=self.sim_cfg)
#                print 'Debug: sampbatch.trajs type:', type(sampbatch.trajs)    # list
#                print 'Debug: sampbatch.obs type:', type(sampbatch.obs)    # RaggedArray
#                print 'Debug: sampbatch.obsfeat type:', type(sampbatch.obsfeat)    # RaggedArray
#                print 'Debug: sampbatch.adist type:', type(sampbatch.adist)    # RaggedArray
#                print 'Debug: sampbatch.a type:', type(sampbatch.a)    # RaggedArray
#                print 'Debug: sampbatch.r type:', type(sampbatch.r)    # RaggedArray
#                print 'Debug: sampbatch.time type:', type(sampbatch.time)    # RaggedArray

#                print 'Debug: sampbatch.obs.array[0] dtype:', sampbatch.obs.arrays[0].dtype
#                print 'Debug: sampbatch.obsfeat.array[0] dtype:', sampbatch.obsfeat.arrays[0].dtype
#                print 'Debug: sampbatch.adist.array[0] dtype:', sampbatch.adist.arrays[0].dtype
#                print 'Debug: sampbatch.a.array[0] dtype:', sampbatch.a.arrays[0].dtype
#                print 'Debug: sampbatch.r.array[0] dtype:', sampbatch.r.arrays[0].dtype
#                print 'Debug: sampbatch.time.array[0] dtype:', sampbatch.time.arrays[0].dtype

                samp_pobsfeat = sampbatch.obsfeat
                self.last_sampbatch = sampbatch

                # XXX add a function to convert sampbatch to seqsampbatches
                seqbatch = sampbatch.to_seq_batch_traj(self.sim_cfg.time_step)
                et = time.time()
#                print("cost time:", et - st)

            ### TODO Reward Function (Discriminator) ###
            # Compute baseline / advantages
            # print 'Computing advantages'
#            print('Debug: Computing advantages')
            with util.Timer() as t_adv:
                st = time.time()
                # Compute observation features for reward input
                seq_robsfeat_stacked = self.reward_obsfeat_fn(seqbatch.obs.stacked)
                # Reward is computed wrt current reward function
                # TODO: normalize rewards
                seq_rcurr_stacked = self.reward_func.compute_reward_restart(seq_robsfeat_stacked, seqbatch.m.stacked, seqbatch.a.stacked, seqbatch.time.stacked, restart=True, n_samples=seq_robsfeat_stacked.shape[0])
#                print 'Debug: seq_rcurr_stacked dtype:', seq_rcurr_stacked.dtype
                assert seq_rcurr_stacked.shape == (seq_robsfeat_stacked.shape[0], seq_robsfeat_stacked.shape[1])


                # TODO convert seq_rcurr_stacked to rcurr corresponding to the trajectories in sampbatch
                seq_rcurr_stacked_flatten = np.reshape(seq_rcurr_stacked, (seq_robsfeat_stacked.shape[0]*seq_robsfeat_stacked.shape[1],))
                rcurr_trajs = []                
                idx = 0
                t = self.sim_cfg.time_step
                for traj in sampbatch.trajs:
                    l = len(traj)
                    seql = int(math.ceil(l/float(t)))
                    rcurr_traj = seq_rcurr_stacked_flatten[idx*t:(idx+seql)*t][:l]
                    rcurr_trajs.append(rcurr_traj)
                    idx += seql
                rcurr = RaggedArray(rcurr_trajs)

#                print("Debug: rcurr lengths:", rcurr.lengths)
#                print("Debug: sampbatch r lengths:", sampbatch.r.lengths)
                assert np.all(rcurr.lengths == sampbatch.r.lengths)
#                rcurr = RaggedArray(rcurr_stacked_flatten, lengths=sampbatch.r.lengths)
#                print 'Debug: rcurr.arrays[0] dtype:', rcurr.arrays[0].dtype

                # If we're regularizing the policy, add negative log probabilities to the rewards
                # Intuitively, the policy gets a bonus for being less certain of its actions
                rcurr_stacked = rcurr.stacked
                orig_rcurr_stacked = rcurr_stacked.copy()
                if self.policy_ent_reg is not None and self.policy_ent_reg != 0:
                    assert self.policy_ent_reg > 0
                    # XXX probably faster to compute this from sampbatch.adist instead
                    # XXX use original policy (not sequential)
                    actionlogprobs_BT = self.policy.compute_action_logprobs(samp_pobsfeat.stacked, sampbatch.a.stacked)
                    policyentbonus_BT = -self.policy_ent_reg * actionlogprobs_BT
                    policyentbonus_BT = policyentbonus_BT.astype(theano.config.floatX)
                    rcurr_stacked += policyentbonus_BT

                else:
                    policyentbonus_BT = np.zeros_like(rcurr_stacked)

                # Compute sequential advantages using these rewards
#                advantages, qvals, vfunc_r2, simplev_r2 = rl.compute_advantage(
#                    rcurr, sampbatch_obsfeat, sampbatch_time, self.value_func, self.discount, self.lam)
                ### TODO Compute sequential advantages for sequential model using these rewards ###
#                print("Debug: sequential value function")
#                    print("Debug: rcurr stacked shape dtype:", rcurr.stacked.shape, rcurr.stacked.dtype)
#                print("Debug: sampbatch obsfeat stacked shape dtype:", sampbatch.obsfeat.stacked.shape, sampbatch.obsfeat.stacked.dtype)
#                print("Debug: sampbatch m stacked shape dtype:", sampbatch.m.stacked.shape, sampbatch.m.stacked.dtype)
#                    print("Debug: sampbatch time stacked shape dtype:", sampbatch.time.stacked.shape, sampbatch.time.stacked.dtype)
#                advantages, qvals, vfunc_r2, simplev_r2 = rl.compute_sequence_advantage(
#                    rcurr, samp_pobsfeat, sampbatch.time, self.value_func, self.discount, self.lam)
#                print 'Debug: advantages type:', type(advantages)    # RaggedArray
#                print 'Debug: qvals type:', type(qvals)    # RaggedArray
#                print 'Debug: vfunc_r2 type:', type(vfunc_r2)    # numpy.float64
#                print 'Debug: simplev_r2 type:', type(simplev_r2)    # numpy.float64

#                print 'Debug: advantages.arrays[0] dtype:', advantages.arrays[0].dtype
#                print 'Debug: qvals.arrays[0] dtype:', qvals.arrays[0].dtype
#                print 'Debug: vfunc_r2 dtype:', vfunc_r2.dtype
#                print 'Debug: simplev_r2 dtype:', simplev_r2.dtype
                # XXX use original compute_advantage
                # Compute advantages using these rewards
                advantages, qvals, vfunc_r2, simplev_r2 = rl.compute_advantage(
                        rcurr, samp_pobsfeat, sampbatch.time, self.value_func, self.discount, self.lam)
                et = time.time()
#                print("cost time:", et - st)

            # Take a step
            # print 'Fitting policy'
#            print('Debug: Fitting policy')
            with util.Timer() as t_step:
                st = time.time()
                params0_P = self.policy.get_params()
#                print 'Debug: params0_P dtype:', params0_P.dtype
#                # XXX restart memory
#                self.policy.restart_hidden_cell(n_samples=sampbatch.obsfeat.stacked.shape[0])
                step_print = self.step_func(
                    self.policy, params0_P,
                    samp_pobsfeat.stacked, sampbatch.a.stacked, sampbatch.adist.stacked,
                    advantages.stacked)
#                print 'Debug: step_print type:', type(step_print)
#                print 'Debug: step_print[0][1] dtype:', step_print[0][1].dtype
#                print 'Debug: step_print[1][1] dtype:', step_print[1][1].dtype
#                print 'Debug: step_print[2][1] dtype:', step_print[2][1].dtype
#                print 'Debug: step_print[3][1] dtype:', step_print[3][1].dtype
                self.policy.update_obsnorm(sampbatch.obsfeat.stacked)
                et = time.time()
#                print("cost time:", et - st)


            # Fit reward function
            # print 'Fitting reward'
#            print('Debug: Fitting reward')
            with util.Timer() as t_r_fit:
                st = time.time()
                ### TODO ### use sequence of expect trajectory
                if True:#self.curr_iter % 20 == 0:
#                    trajlen = seqbatch.r.stacked.shape[1]
#                    print("Debug: ex_robsfeat shape:", self.ex_robsfeat.shape, "ex_pobsfeat shape:", self.ex_pobsfeat.shape)
                    # Subsample expert transitions to the same sample count for the policy
#                    inds = np.random.choice(self.ex_robsfeat.shape[0], size=sampbatch.obsfeat.stacked.shape[0])
                    inds = [ind for si in np.random.choice(self.ex_robsfeat.shape[0]-self.sim_cfg.time_step, size=seqbatch.obsfeat.stacked.shape[0]) for ind in range(si,si+self.sim_cfg.time_step)]
                    exbatch_robsfeat = self.ex_robsfeat[inds,:]
                    exbatch_pobsfeat = self.ex_pobsfeat[inds,:] # only used for logging
                    exbatch_a = self.ex_a[inds,:]
                    exbatch_t = self.ex_t[inds]
                    # reshape: (B*T, ...) => (B, T, ...)
                    exbatch_robsfeat = np.reshape(exbatch_robsfeat, (seqbatch.obsfeat.stacked.shape[0], self.sim_cfg.time_step, -1))
                    exbatch_a = np.reshape(exbatch_a, (seqbatch.obsfeat.stacked.shape[0], self.sim_cfg.time_step, -1))
                    exbatch_t = np.reshape(exbatch_t, (seqbatch.obsfeat.stacked.shape[0], self.sim_cfg.time_step,))
#                    print 'Debug: samp_robsfeat_stacked type:', type(samp_robsfeat_stacked)
#                    print 'Debug: inds type:', type(inds)    # ndarray
#                    print 'Debug: exbatch_robsfeat type:', type(exbatch_robsfeat)    # ndarray
#                    print 'Debug: exbatch_pobsfeat type:', type(exbatch_pobsfeat)    # ndarray
#                    print 'Debug: exbatch_a type:', type(exbatch_a)    # ndarray
#                    print 'Debug: exbatch_t type:', type(exbatch_t)    # ndarray

#                    print 'Debug: inds type:', inds.dtype    # int64
#                    print 'Debug: exbatch_robsfeat dtype:', exbatch_robsfeat.dtype    # float32
#                    print 'Debug: exbatch_pobsfeat dtype:', exbatch_pobsfeat.dtype    # float32
#                    print 'Debug: exbatch_a dtype:', exbatch_a.dtype    # int64
#                    print 'Debug: exbatch_t dtype:', exbatch_t.dtype    # float32
#                    print 'Debug: samp_robsfeat_stacked dtype:', samp_robsfeat_stacked.dtype
                    rfit_print = self.reward_func.fit(seq_robsfeat_stacked, seqbatch.m.stacked, seqbatch.a.stacked, seqbatch.time.stacked, exbatch_robsfeat, exbatch_a, exbatch_t)
                else:
                    rfit_print = []
                et = time.time()
#                print("cost time:", et - st)

            # Fit value function for next iteration
            # print 'Fitting value function'
#            print('Debug: Fitting value function')
            with util.Timer() as t_vf_fit:
                st = time.time()
                if self.value_func is not None:
                    # Recompute q vals # XXX: this is only necessary if fitting reward after policy
                    # qnew = qvals

                    # TODO: this should be a byproduct of reward fitting
#                    print("Debug: samp_robsfeat_stacked shape:", samp_robsfeat_stacked.shape)
#                    print("Debug: sampbatch m stacked shape:", sampbatch.m.stacked.shape)
#                    print("Debug: sampbatch a stacked shape:", sampbatch.a.stacked.shape)
#                    print("Debug: sampbatch time stacked shape:", sampbatch.time.stacked.shape)
                    seq_rnew_stacked = self.reward_func.compute_reward_restart(seq_robsfeat_stacked, seqbatch.m.stacked, seqbatch.a.stacked, seqbatch.time.stacked, restart=True, n_samples=seq_robsfeat_stacked.shape[0])
                    # TODO convert seq_rnew_stacked to rnew corresponding to the trajectories in sampbatch
                    seq_rnew_stacked_flatten = np.reshape(seq_rnew_stacked, (seq_robsfeat_stacked.shape[0]*seq_robsfeat_stacked.shape[1],))
                    rnew_trajs = []
                    idx = 0                    
                    t = self.sim_cfg.time_step
                    for traj in sampbatch.trajs:
                        l = len(traj)
                        seql = int(math.ceil(l/float(t)))
                        rnew_traj = seq_rnew_stacked_flatten[idx*t:(idx+seql)*t][:l]
                        rnew_trajs.append(rnew_traj)
                        idx += seql
                    rnew = RaggedArray(rnew_trajs)
                    
                    assert np.all(rnew.lengths == sampbatch.r.lengths)
#                    rnew = RaggedArray(
#                        self.reward_func.compute_reward_restart(samp_robsfeat_stacked, sampbatch.m.stacked, sampbatch.a.stacked, sampbatch.time.stacked, restart=True, n_samples=samp_robsfeat_stacked.shape[0]),
#                        lengths=sampbatch.r.lengths)
#                        print("Debug: rnew stacked shape:", rnew.stacked.shape)
                    qnew, _ = rl.compute_qvals(rnew, self.discount)
#                    print("Debug: qnew stacked shape:", qnew.stacked.shape)
                    vfit_print = self.value_func.fit(samp_pobsfeat.stacked, sampbatch.time.stacked, qnew.stacked)
                else:
                    vfit_print = []
                et = time.time()
#                print("cost time:", et - st)

        # Log
        self.total_num_trajs += len(sampbatch)
        self.total_num_sa += sum(len(traj) for traj in sampbatch)
        self.total_time += t_all.dt
#        print("Debug: Log")
#        print("sampbatch adist stacked shape:", sampbatch.adist.stacked.shape, "dtype:", sampbatch.adist.stacked.dtype)
        fields = [
            ('iter', self.curr_iter, int),
            ('trueret', sampbatch.r.padded(fill=0.).sum(axis=1).mean(), float), # average return for this batch of trajectories
            ('iret', rcurr.padded(fill=0.).sum(axis=1).mean(), float), # average return on imitation reward
            ('avglen', int(np.mean([len(traj) for traj in sampbatch])), int), # average traj length
            ('ntrajs', self.total_num_trajs, int), # total number of trajs sampled over the course of training
            ('nsa', self.total_num_sa, int), # total number of state-action pairs sampled over the course of training
            ('ent', self.policy._compute_actiondist_entropy(sampbatch.adist.stacked).mean(), float), # entropy of action distributions
            ('vf_r2', vfunc_r2, float),
            ('tdvf_r2', simplev_r2, float),
            ('dx', util.maxnorm(params0_P - self.policy.get_params()), float), # max parameter difference from last iteration
        ] + step_print + vfit_print + rfit_print + [
            ('avgr', rcurr_stacked.mean(), float), # average regularized reward encountered
            ('avgunregr', orig_rcurr_stacked.mean(), float), # average unregularized reward
            ('avgpreg', policyentbonus_BT.mean(), float), # average policy regularization
            # ('bcloss', -self.policy.compute_action_logprobs(exbatch_pobsfeat, exbatch_a).mean(), float), # negative log likelihood of expert actions
            # ('bcloss', np.square(self.policy.compute_actiondist_mean(exbatch_pobsfeat) - exbatch_a).sum(axis=1).mean(axis=0), float),
            ('tsamp', t_sample.dt, float), # time for sampling
            ('tadv', t_adv.dt + t_vf_fit.dt, float), # time for advantage computation
            ('tstep', t_step.dt, float), # time for step computation
            ('ttotal', self.total_time, float), # total time
        ]
        self.curr_iter += 1
#        print("Debug: fields:")
#        for i in range(len(fields)):
#            print(fields[i][0], len(fields[i]))
        return fields
