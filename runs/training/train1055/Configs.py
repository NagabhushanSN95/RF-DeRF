config = {
    'device': 'cuda:0',

    # Run first for 1 step with data_downsample=4 to generate weights for ray importance sampling
    'data_downsample': 2,
    'data_dirpath': '../../../../../databases/InterDigital/data/all',
    'camera_suffix': '_undistorted',
    # 'set_num': 8,
    # 'scene_names': ['Birthday'],
    'set_num': 5,
    'num_frames': 300,
    'num_render_frames': 300,
    'contract': False,
    'ndc': True,
    'ndc_far': 2.6,
    'near_scaling': 0.95,
    'isg': False,
    'isg_step': -1,
    'ist_step': 50000,
    'keyframes': False,
    'scene_bbox': [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],

    # Optimization settings
    'num_steps': 30001,
    'batch_size': 4096,
    'scheduler_type': 'warmup_cosine',
    'optim_type': 'adam',
    'lr': 0.01,

    # Sparse flow configs
    'num_sparse_flow_pixels': 2048,
    'sparse_flow_dirnames': ['FEL001_FE05'],

    # Dense flow configs
    'dense_flow_dirnames': ['FEL003_FE05'],
    # 'dense_flow_mask_dirnames': ['FEL003_FV05'],
    'dense_flow_cache_size': 180,  # 180 files for every 1000 iters, so 1800 (3 input viewpoints) files for 10000 iters
    'dense_flow_reload_iters': 1000,

    # # Sparse depth configs
    # 'num_sparse_depth_pixels': 2048,
    # 'sparse_depth_dirnames': ['DEL001_DE05'],

    # Regularization
    'distortion_loss_weight': 0.001,
    'histogram_loss_weight': 1.0,
    'l1_time_planes': 0.0001,
    'l1_time_planes_proposal_net': 0.0001,
    'plane_tv_weight': 0.0002,
    'plane_tv_weight_proposal_net': 0.0002,
    'time_smoothness_weight': 0.001,
    'time_smoothness_weight_proposal_net': 1e-05,
    'sparse_flow_loss_weight': 1,
    'sparse_flow_loss_threshold': 0,
    'sparse_flow_loss_average_point': True,
    'sparse_flow_loss_weighted_error': False,
    'sparse_flow_loss_stop_gradient_weights': False,
    'dense_flow_loss_weight': 0.1,
    'dense_flow_loss_threshold': 0,
    'dense_flow_loss_average_point': True,
    'dense_flow_loss_weighted_error': False,
    'dense_flow_loss_stop_gradient_weights': False,
    # 'sparse_depth_loss_weight': 0.01,

    # Training settings
    'save_every': 30000,
    'valid_every': 30000,
    'save_outputs': True,
    'save_true_depth': False,
    'train_fp16': True,

    # Raymarching settings
    'single_jitter': False,
    'num_samples': 48,
    'num_proposal_samples': [256, 128],
    'num_proposal_iterations': 2,
    'use_same_proposal_network': False,
    'use_proposal_weight_anneal': True,
    'proposal_net_args_list': [
        {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [128, 128, 128, 150]},
        {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [256, 256, 256, 150]}
    ],

    # Model settings
    'concat_features_across_scales': True,
    'density_activation': 'trunc_exp',
    'flow_activation': 'tanh',
    'linear_decoder': False,
    'multiscale_res': [1, 2, 4, 8],
    'canonical_time': -2,
    'time_dependent_color': True,
    'grid_config': {
        'model_3d': {
            'grid_dimensions': 2,
            'input_coordinate_dim': 3,
            'output_coordinate_dim': 8,
            'resolution': [64, 64, 64]
        },
        'model_bf': {                        # model_backward_flow
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 8,
            'resolution': [64, 64, 64, 150]
        },
    },
}
