import argparse, h5py, json
import os, os.path
import sys, inspect
import numpy as np
import theano
from theano import tensor
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

curdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
pardir = os.path.dirname(curdir)
sys.path.insert(0,pardir)

from environments import rlgymenv
import policyopt
from policyopt import imitation, nn, rl, util

# Add sequential modes
#MODES = ('bclone', 'ga')
MODES = ('bclone', 'ga', 'sga')
OBSNORM_MODES = ('none', 'expertdata', 'online')
# Add sequential architectures
# XXX use sequential architecture only in discriminator
TINY_ARCHITECTURE = '[{"type": "fc", "n": 64}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 64}, {"type": "nonlin", "func": "tanh"}]'
SIMPLE_ARCHITECTURE = '[{"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}]'
SEQ_TINY_ARCHITECTURE = '[{"type": "lstm", "n": 64}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 64}, {"type": "nonlin", "func": "tanh"}]'
SEQ_SIMPLE_ARCHITECTURE = '[{"type": "lstm", "n": 100}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}]'

def load_dataset(filename, limit_trajs, data_subsamp_freq):
    # Load expert data
    with h5py.File(filename, 'r') as f:
        # Read data as written by vis_mj.py
        full_dset_size = f['obs_B_T_Do'].shape[0] # full dataset size
        dset_size = min(full_dset_size, limit_trajs) if limit_trajs is not None else full_dset_size

        exobs_B_T_Do = f['obs_B_T_Do'][:dset_size,...][...]
        exa_B_T_Da = f['a_B_T_Da'][:dset_size,...][...]
        exr_B_T = f['r_B_T'][:dset_size,...][...]
        exlen_B = f['len_B'][:dset_size,...][...]

    print ('Expert dataset size: {} transitions ({} trajectories)'.format(exlen_B.sum(), len(exlen_B)))
    print ('Expert average return:', exr_B_T.sum(axis=1).mean())

    # Stack everything together
    start_times_B = np.random.RandomState(0).randint(0, data_subsamp_freq, size=exlen_B.shape[0])
    print ('start times')
    print (start_times_B)
    exobs_Bstacked_Do = np.concatenate(
        [exobs_B_T_Do[i,start_times_B[i]:l:data_subsamp_freq,:] for i, l in enumerate(exlen_B)],
        axis=0)
    exa_Bstacked_Da = np.concatenate(
        [exa_B_T_Da[i,start_times_B[i]:l:data_subsamp_freq,:] for i, l in enumerate(exlen_B)],
        axis=0)
    ext_Bstacked = np.concatenate(
        [np.arange(start_times_B[i], l, step=data_subsamp_freq) for i, l in enumerate(exlen_B)]).astype(float)

    assert exobs_Bstacked_Do.shape[0] == exa_Bstacked_Da.shape[0] == ext_Bstacked.shape[0]# == np.ceil(exlen_B.astype(float)/data_subsamp_freq).astype(int).sum() > 0

    print ('Subsampled data every {} timestep(s)'.format(data_subsamp_freq))
    print ('Final dataset size: {} transitions (average {} per traj)'.format(exobs_Bstacked_Do.shape[0], float(exobs_Bstacked_Do.shape[0])/dset_size))

    # Convert data type to follow theano config
    exobs_Bstacked_Do = exobs_Bstacked_Do.astype(theano.config.floatX)
    ext_Bstacked = ext_Bstacked.astype(theano.config.floatX)

    return exobs_Bstacked_Do, exa_Bstacked_Da, ext_Bstacked


def main():
    np.set_printoptions(suppress=True, precision=5, linewidth=1000)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES, required=True)
    # Expert dataset
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--limit_trajs', type=int, required=True)
    parser.add_argument('--data_subsamp_freq', type=int, required=True)
    # MDP options
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--max_traj_len', type=int, default=None)
    # Policy architecture
    parser.add_argument('--policy_hidden_spec', type=str, default=SIMPLE_ARCHITECTURE)
    parser.add_argument('--tiny_policy', action='store_true')
    parser.add_argument('--obsnorm_mode', choices=OBSNORM_MODES, default='expertdata')

    # add a spec for transition classifier
    parser.add_argument('--clf_hidden_spec', type=str, default=SIMPLE_ARCHITECTURE)

    # Behavioral cloning optimizer
    parser.add_argument('--bclone_lr', type=float, default=1e-3)
    parser.add_argument('--bclone_batch_size', type=int, default=128)
    # parser.add_argument('--bclone_eval_nsa', type=int, default=128*100)
    parser.add_argument('--bclone_eval_ntrajs', type=int, default=20)
    parser.add_argument('--bclone_eval_freq', type=int, default=1000)
    parser.add_argument('--bclone_train_frac', type=float, default=.7)
    # Imitation optimizer
    parser.add_argument('--discount', type=float, default=.995)
    parser.add_argument('--lam', type=float, default=.97)
    parser.add_argument('--max_iter', type=int, default=1000000)
    parser.add_argument('--policy_max_kl', type=float, default=.01)
    parser.add_argument('--policy_cg_damping', type=float, default=.1)
    parser.add_argument('--no_vf', type=int, default=0)
    parser.add_argument('--vf_max_kl', type=float, default=.01)
    parser.add_argument('--vf_cg_damping', type=float, default=.1)
    parser.add_argument('--policy_ent_reg', type=float, default=0.)
    parser.add_argument('--reward_type', type=str, default='nn')
    # parser.add_argument('--linear_reward_bin_features', type=int, default=0)
    parser.add_argument('--reward_max_kl', type=float, default=.01)
    parser.add_argument('--reward_lr', type=float, default=.01)
    parser.add_argument('--reward_steps', type=int, default=1)
    parser.add_argument('--reward_ent_reg_weight', type=float, default=.001)
    parser.add_argument('--reward_include_time', type=int, default=0)
    parser.add_argument('--sim_batch_size', type=int, default=None)
    parser.add_argument('--min_total_sa', type=int, default=50000)
    parser.add_argument('--favor_zero_expert_reward', type=int, default=0)
    # Saving stuff
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--plot_freq', type=int, default=100)
    parser.add_argument('--log', type=str, required=False)

    # Sequential model
    parser.add_argument('--seq_model', type=int, default=0)
    parser.add_argument('--time_step', type=int, default=10)

    args = parser.parse_args()

    # Initialize the MDP
    if not args.seq_model:
        if args.tiny_policy:
            assert args.policy_hidden_spec == SIMPLE_ARCHITECTURE, 'policy_hidden_spec must remain unspecified if --tiny_policy is set'
            args.policy_hidden_spec = TINY_ARCHITECTURE
        argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
        print(argstr)
    # Add sequential model
    else:
        if args.tiny_policy:
            assert args.policy_hidden_spec == SEQ_SIMPLE_ARCHITECTURE, 'policy_hidden_spec must remain unspecified if --tiny_policy is set'
            args.policy_hidden_spec = SEQ_TINY_ARCHITECTURE
#        # change the default architecture to fit sequential model
#        if args.policy_hidden_spec == SIMPLE_ARCHITECTURE:
#            args.policy_hidden_spec = SEQ_SIMPLE_ARCHITECTURE
        if args.clf_hidden_spec == SIMPLE_ARCHITECTURE:
            args.clf_hidden_spec = SEQ_SIMPLE_ARCHITECTURE
        argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)

    mdp = rlgymenv.RLGymMDP(args.env_name)
    util.header('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

    # Initialize the policy
    enable_obsnorm = args.obsnorm_mode != 'none'
    
    if not args.seq_model:
        if isinstance(mdp.action_space, policyopt.ContinuousSpace):
            policy_cfg = rl.GaussianPolicyConfig(
                hidden_spec=args.policy_hidden_spec,
                min_stdev=0.,
                init_logstdev=0.,
                enable_obsnorm=enable_obsnorm)
            policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy')
        else:
            policy_cfg = rl.GibbsPolicyConfig(
                hidden_spec=args.policy_hidden_spec,
                enable_obsnorm=enable_obsnorm)
            policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GibbsPolicy')
    # Add squential model
    else:
        if isinstance(mdp.action_space, policyopt.ContinuousSpace):
            policy_cfg = rl.SeqGaussianPolicyConfig(
                hidden_spec=args.policy_hidden_spec,
                time_step=args.time_step,    # add time step
                min_stdev=0.,
                init_logstdev=0.,
                enable_obsnorm=enable_obsnorm,
                enable_actnorm=False)    # XXX not implement actnorm yet
            policy = rl.SeqGaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'SeqGaussianPolicy')
        else:
            policy_cfg = rl.SeqGibbsPolicyConfig(
                hidden_spec=args.policy_hidden_spec,
                time_step=args.time_step,    # add time step
                enable_obsnorm=enable_obsnorm,
                enable_actnorm=False)    # XXX not implement actnorm yet
            policy = rl.SeqGibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'SeqGibbsPolicy')        

    util.header('Policy architecture')
    for v in policy.get_trainable_variables():
        util.header('- %s (%d parameters)' % (v.name, v.get_value().size))
    util.header('Total: %d parameters' % (policy.get_num_params(),))

    # Load expert data
    exobs_Bstacked_Do, exa_Bstacked_Da, ext_Bstacked = load_dataset(
        args.data, args.limit_trajs, args.data_subsamp_freq)
    assert exobs_Bstacked_Do.shape[1] == mdp.obs_space.storage_size
    assert exa_Bstacked_Da.shape[1] == mdp.action_space.storage_size
    assert ext_Bstacked.ndim == 1

#    print 'Debug: exobs_Bstacked_Do dtype:', exobs_Bstacked_Do.dtype
#    print 'Debug: exa_Bstacked_Da dtype:', exa_Bstacked_Da.dtype
#    print 'Debug: ext_Bstacked dtype:', ext_Bstacked.dtype

#    assert 1 == 0

    # Start optimization
    max_traj_len = args.max_traj_len if args.max_traj_len is not None else mdp.env_spec.timestep_limit
    print ('Max traj len:', max_traj_len)

    if args.mode == 'bclone':
        # For behavioral cloning, only print output when evaluating
#        args.print_freq = args.bclone_eval_freq
#        args.save_freq = args.bclone_eval_freq

        reward, vf = None, None
        opt = imitation.BehavioralCloningOptimizer(
            mdp, policy,
            lr=args.bclone_lr,
            batch_size=args.bclone_batch_size,
            obsfeat_fn=lambda o:o,
            ex_obs=exobs_Bstacked_Do, ex_a=exa_Bstacked_Da,
            eval_sim_cfg=policyopt.SimConfig(
                min_num_trajs=args.bclone_eval_ntrajs, min_total_sa=-1,
                batch_size=args.sim_batch_size, max_traj_len=max_traj_len,
                smp_traj_len=-1),
            eval_freq=args.bclone_eval_freq,  # XXX set a value when using bclone
            train_frac=args.bclone_train_frac)

    elif args.mode == 'ga':
        if args.reward_type == 'nn':
            reward = imitation.TransitionClassifier(
                hidden_spec=args.policy_hidden_spec,
                obsfeat_space=mdp.obs_space,
                action_space=mdp.action_space,
                max_kl=args.reward_max_kl,
                adam_lr=args.reward_lr,
                adam_steps=args.reward_steps,
                ent_reg_weight=args.reward_ent_reg_weight,
                enable_inputnorm=True,
                include_time=bool(args.reward_include_time),
                time_scale=1./mdp.env_spec.timestep_limit,
                favor_zero_expert_reward=bool(args.favor_zero_expert_reward),
                varscope_name='TransitionClassifier')
        elif args.reward_type in ['l2ball', 'simplex']:
            reward = imitation.LinearReward(
                obsfeat_space=mdp.obs_space,
                action_space=mdp.action_space,
                mode=args.reward_type,
                enable_inputnorm=True,
                favor_zero_expert_reward=bool(args.favor_zero_expert_reward),
                include_time=bool(args.reward_include_time),
                time_scale=1./mdp.env_spec.timestep_limit,
                exobs_Bex_Do=exobs_Bstacked_Do,
                exa_Bex_Da=exa_Bstacked_Da,
                ext_Bex=ext_Bstacked)
        else:
            raise NotImplementedError(args.reward_type)

        vf = None if bool(args.no_vf) else rl.ValueFunc(
            hidden_spec=args.policy_hidden_spec,
            obsfeat_space=mdp.obs_space,
            enable_obsnorm=args.obsnorm_mode != 'none',
            enable_vnorm=True,
            max_kl=args.vf_max_kl,
            damping=args.vf_cg_damping,
            time_scale=1./mdp.env_spec.timestep_limit,
            varscope_name='ValueFunc')

        opt = imitation.ImitationOptimizer(
            mdp=mdp,
            discount=args.discount,
            lam=args.lam,
            policy=policy,
            sim_cfg=policyopt.SimConfig(
                min_num_trajs=-1, min_total_sa=args.min_total_sa,
                batch_size=args.sim_batch_size, max_traj_len=max_traj_len,
                smp_traj_len=-1),
            step_func=rl.TRPO(max_kl=args.policy_max_kl, damping=args.policy_cg_damping, sequential_model=False),  # add sequential model
            reward_func=reward,
            value_func=vf,
            policy_obsfeat_fn=lambda obs: obs,
            reward_obsfeat_fn=lambda obs: obs,
            policy_ent_reg=args.policy_ent_reg,
            ex_obs=exobs_Bstacked_Do,
            ex_a=exa_Bstacked_Da,
            ex_t=ext_Bstacked)


    # Add Sequential Model
    elif args.mode == 'sga':
        if args.reward_type == 'nn':
            reward = imitation.SequentialTransitionClassifier(
                hidden_spec=args.clf_hidden_spec,
                obsfeat_space=mdp.obs_space,
                action_space=mdp.action_space,
                max_kl=args.reward_max_kl,
                adam_lr=args.reward_lr,
                adam_steps=args.reward_steps,
                ent_reg_weight=args.reward_ent_reg_weight,
                time_step=args.time_step,    # add time step
                enable_inputnorm=True,
                include_time=bool(args.reward_include_time),
                time_scale=1./mdp.env_spec.timestep_limit,
                favor_zero_expert_reward=bool(args.favor_zero_expert_reward),
                varscope_name='SequentialTransitionClassifier')
#        elif args.reward_type in ['l2ball', 'simplex']:
#            reward = imitation.LinearReward(
#                obsfeat_space=mdp.obs_space,
#                action_space=mdp.action_space,
#                mode=args.reward_type,
#                enable_inputnorm=True,
#                favor_zero_expert_reward=bool(args.favor_zero_expert_reward),
#                include_time=bool(args.reward_include_time),
#                time_scale=1./mdp.env_spec.timestep_limit,
#                exobs_Bex_Do=exobs_Bstacked_Do,
#                exa_Bex_Da=exa_Bstacked_Da,
#                ext_Bex=ext_Bstacked)
        else:
            raise NotImplementedError(args.reward_type)

        vf = None if bool(args.no_vf) else rl.SequentialValueFunc(
            hidden_spec=args.policy_hidden_spec,
            obsfeat_space=mdp.obs_space,
            time_step=args.time_step,    # add time step
            enable_obsnorm=args.obsnorm_mode != 'none',
            enable_vnorm=True,
            max_kl=args.vf_max_kl,
            damping=args.vf_cg_damping,
            time_scale=1./mdp.env_spec.timestep_limit,
            varscope_name='SequentialValueFunc')

        opt = imitation.SequentialImitationOptimizer(
            mdp=mdp,
            discount=args.discount,
            lam=args.lam,
            policy=policy,
            sim_cfg=policyopt.SeqSimConfig(
                min_num_trajs=-1, min_total_sa=args.min_total_sa,
                batch_size=args.sim_batch_size, max_traj_len=max_traj_len,
                time_step=args.time_step),    # add time step
            step_func=rl.TRPO(max_kl=args.policy_max_kl, 
                damping=args.policy_cg_damping,
                sequential_model=False),    # XXX not use sequential trpo
            reward_func=reward,
            value_func=vf,
            policy_obsfeat_fn=lambda obs: obs,
            reward_obsfeat_fn=lambda obs: obs,
            policy_ent_reg=args.policy_ent_reg,
            ex_obs=exobs_Bstacked_Do,
            ex_a=exa_Bstacked_Da,
            ex_t=ext_Bstacked)
    
    

    # Set observation normalization
    if args.obsnorm_mode == 'expertdata':
        if not args.seq_model:
            policy.update_obsnorm(exobs_Bstacked_Do)
            if reward is not None: reward.update_inputnorm(opt.reward_obsfeat_fn(exobs_Bstacked_Do), exa_Bstacked_Da)
            if vf is not None: vf.update_obsnorm(opt.policy_obsfeat_fn(exobs_Bstacked_Do))
        # Add sequential model
        else:
            Bstacked, Do, T = exobs_Bstacked_Do.shape[0], exobs_Bstacked_Do.shape[1], args.time_step
            exobs_BT_Do = exobs_Bstacked_Do[:T*(Bstacked//T),:]
            exa_BT_Da = exa_Bstacked_Da[:T*(Bstacked//T),:]
            # reshape:(B*T, ...) => (B, T, ...)
            exobs_B_T_Do = np.reshape(exobs_BT_Do, (Bstacked//T, T, exobs_Bstacked_Do.shape[1]))
            exa_B_T_Da = np.reshape(exa_BT_Da, (Bstacked//T, T, exa_Bstacked_Da.shape[1]))
            print("Debug: exobs_Bstacked_Do:", exobs_Bstacked_Do.shape[0], exobs_Bstacked_Do.shape[1])            
            print("Debug: exobs_B_T_Do:", exobs_B_T_Do.shape[0], exobs_B_T_Do.shape[1], exobs_B_T_Do.shape[2])            
            # XXX use original policy (not sequential)
            policy.update_obsnorm(exobs_Bstacked_Do)
            if reward is not None: reward.update_inputnorm(opt.reward_obsfeat_fn(exobs_B_T_Do), exa_B_T_Da)
            if vf is not None: vf.update_obsnorm(opt.policy_obsfeat_fn(exobs_Bstacked_Do))
            

    # Run optimizer

#    log = nn.TrainingLog(args.log, [('args', argstr)])
    log = nn.BasicTrainingLog(args.log, [('args', argstr)])
    for i in xrange(args.max_iter):
        iter_info = opt.step()
#        log.write(iter_info, print_header=i % (20*args.print_freq) == 0, display=i % args.print_freq == 0)
        log.add_log(iter_info, print_header=i % (20*args.print_freq) == 0, display= i % args.print_freq == 0)
        if args.save_freq != 0 and i % args.save_freq == 0 and args.log is not None:
            print ('%i/%i iters is done. Save snapshot.' %(i, args.max_iter))
#            log.write_snapshot(policy, i)
            log.write_snapshot(policy, i)

        if args.mode == 'ga' and args.plot_freq != 0 and i % args.plot_freq == 0:
            print ('%i/%i iters is done. Save plot.' %(i, args.max_iter))
            exdata_N_Doa = np.concatenate([exobs_Bstacked_Do, exa_Bstacked_Da], axis=1)
            pdata_M_Doa = np.concatenate([opt.last_sampbatch.obs.stacked, opt.last_sampbatch.a.stacked], axis=1)
            # convert dtype to follow theano config
            exdata_N_Doa = exdata_N_Doa.astype(theano.config.floatX)
            pdata_M_Doa = pdata_M_Doa.astype(theano.config.floatX)
#            print 'Debug: exobs_Bstacked_Do dtype:', exobs_Bstacked_Do.dtype    # float32
#            print 'Debug: exa_Bstacked_Da dtype:', exa_Bstacked_Da.dtype    # int64
#            print 'Debug: opt.last_sampbatch.obs.stacked dtype:', opt.last_sampbatch.obs.stacked.dtype    # float32
#            print 'Debug: opt.last_sampbatch.a.stacked dtype:', opt.last_sampbatch.a.stacked.dtype    # int64
#            print 'Debug: exdata_N_Doa dtype:', exdata_N_Doa.dtype    # float32
#            print 'Debug: pdata_M_Doa dtype:', pdata_M_Doa.dtype    # float32

            # Plot reward
#            import matplotlib
#            matplotlib.use('Agg')
#            import matplotlib.pyplot as plt
            _, ax = plt.subplots()
            idx1, idx2 = 0,1
            range1 = (min(exdata_N_Doa[:,idx1].min(), pdata_M_Doa[:,idx1].min()), max(exdata_N_Doa[:,idx1].max(), pdata_M_Doa[:,idx1].max()))
            range2 = (min(exdata_N_Doa[:,idx2].min(), pdata_M_Doa[:,idx2].min()), max(exdata_N_Doa[:,idx2].max(), pdata_M_Doa[:,idx2].max()))

#            print 'Debug: range1 types:', type(range1[0]), type(range1[1])    # float32, float32
#            print 'Debug: range2 types:', type(range2[0]), type(range2[1])    # float32, float32

            x, y, z = reward.plot(ax, idx1, idx2, range1, range2, n=100)
            plot = [x, y, z, exdata_N_Doa[:,idx1], exdata_N_Doa[:,idx2], pdata_M_Doa[:,idx1], pdata_M_Doa[:,idx2]]
            log.write_plot(plot, i)

            # Plot expert data
#            ax.scatter(exdata_N_Doa[:,idx1], exdata_N_Doa[:,idx2], color='blue', s=1, label='expert')

            # Plot policy samples
#            ax.scatter(pdata_M_Doa[:,idx1], pdata_M_Doa[:,idx2], color='red', s=1, label='apprentice')

#            ax.legend()
#            plt.show()
#            plt.savefig()
#            plot = [x, y, z, exdata_N_Doa[:,idx1], exdata_N_Doa[:,idx2], pdata_M_Doa[:,idx1], pdata_M_Doa[:,idx2]]
#            log.write_plot(plot, i)



#        if args.mode == 'sga' and args.plot_freq != 0 and i % args.plot_freq == 0:
#            print ('%i/%i iters is done. Save plot.' %(i, args.max_iter))
#            exdata_N_Doa = np.concatenate([exobs_Bstacked_Do, exa_Bstacked_Da], axis=1)
#            # reshape: (B, T, ...) => (B*T, ...)
##            B, T, Df = opt.last_sampbatch.obs.stacked.shape
##            obs_flatten = np.reshape(opt.last_sampbatch.obs.stacked, (B*T, opt.last_sampbatch.obs.stacked.shape[2]))
##            a_flatten = np.reshape(opt.last_sampbatch.a.stacked, (B*T, opt.last_sampbatch.a.stacked.shape[2]))
###            pdata_M_Doa = np.concatenate([opt.last_sampbatch.obs.stacked, opt.last_sampbatch.a.stacked], axis=1)
#            pdata_M_Doa = np.concatenate([opt.last_sampbatch.obs.stacked, opt.last_sampbatch.a.stacked], axis=1)
#            # convert dtype to follow theano config
#            exdata_N_Doa = exdata_N_Doa.astype(theano.config.floatX)
#            pdata_M_Doa = pdata_M_Doa.astype(theano.config.floatX)
##            print 'Debug: exobs_Bstacked_Do dtype:', exobs_Bstacked_Do.dtype    # float32
##            print 'Debug: exa_Bstacked_Da dtype:', exa_Bstacked_Da.dtype    # int64
##            print 'Debug: opt.last_sampbatch.obs.stacked dtype:', opt.last_sampbatch.obs.stacked.dtype    # float32
##            print 'Debug: opt.last_sampbatch.a.stacked dtype:', opt.last_sampbatch.a.stacked.dtype    # int64
##            print 'Debug: exdata_N_Doa dtype:', exdata_N_Doa.dtype    # float32
##            print 'Debug: pdata_M_Doa dtype:', pdata_M_Doa.dtype    # float32

#            # Plot reward
##            import matplotlib
##            matplotlib.use('Agg')
##            import matplotlib.pyplot as plt
#            _, ax = plt.subplots()
#            idx1, idx2 = 0,1
#            range1 = (min(exdata_N_Doa[:,idx1].min(), pdata_M_Doa[:,idx1].min()), max(exdata_N_Doa[:,idx1].max(), pdata_M_Doa[:,idx1].max()))
#            range2 = (min(exdata_N_Doa[:,idx2].min(), pdata_M_Doa[:,idx2].min()), max(exdata_N_Doa[:,idx2].max(), pdata_M_Doa[:,idx2].max()))

##            print 'Debug: range1 types:', type(range1[0]), type(range1[1])    # float32, float32
##            print 'Debug: range2 types:', type(range2[0]), type(range2[1])    # float32, float32

 #           # for sequential model, input the length of sequence
 #           # XXX take care of the usage of memory !!
 #           x, y, z = reward.plot(ax, idx1, idx2, range1, range2, args.time_step, n=100)
 #           plot = [x, y, z, exdata_N_Doa[:,idx1], exdata_N_Doa[:,idx2], pdata_M_Doa[:,idx1], pdata_M_Doa[:,idx2]]
 #           log.write_plot(plot, i)

#            # Plot expert data
##            ax.scatter(exdata_N_Doa[:,idx1], exdata_N_Doa[:,idx2], color='blue', s=1, label='expert')

#            # Plot policy samples
##            ax.scatter(pdata_M_Doa[:,idx1], pdata_M_Doa[:,idx2], color='red', s=1, label='apprentice')

##            ax.legend()
##            plt.show()
##            plt.savefig()
##            plot = [x, y, z, exdata_N_Doa[:,idx1], exdata_N_Doa[:,idx2], pdata_M_Doa[:,idx1], pdata_M_Doa[:,idx2]]
##            log.write_plot(plot, i)




    # write log
    print ('Training is done. Save log.')
    log.write_log()
    log.close()


if __name__ == '__main__':
    main()
