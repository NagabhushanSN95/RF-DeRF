config = {
    'expname': 'cook_spinach',
    'logdir': '../runs/training/train0006',
    'data_dirs': [
        '../../../../../databases/N3DV/data/all/database_data/cook_spinach'
    ],
    'flow_dirpath': '../../../../../databases/N3DV/data/all/estimated_flows',
    'depth_dirpath': '../../../../../databases/N3DV/data/all/estimated_depths',
    'device': 'cuda:0',
    'data_downsample': 2,
    'set_num': 4,
    'contract': False,
    'ndc': True,
    'ndc_far': 2.6,
    'near_scaling': 0.95,
    'isg': False,
    'isg_step': -1,
    'ist_step': 50000,
    'keyframes': False,
    'scene_bbox': [
        [
            -3.0,
            -1.8,
            -1.2
        ],
        [
            3.0,
            1.8,
            1.2
        ]
    ],
    'num_steps': 30001,
    'batch_size': 4096,
    'scheduler_type': 'warmup_cosine',
    'optim_type': 'adam',
    'lr': 0.01,
    'num_sparse_flow_pixels': 2048,
    'sparse_flow_dirnames': [
        'FEL001_FE04'
    ],
    'dense_flow_dirnames': [
        'FEL003_FE04'
    ],
    'dense_flow_cache_size': 180,
    'dense_flow_reload_iters': 1000,
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
    'dense_flow_loss_weight': 1,
    'dense_flow_loss_threshold': 0,
    'dense_flow_loss_average_point': True,
    'dense_flow_loss_weighted_error': False,
    'dense_flow_loss_stop_gradient_weights': False,
    'save_every': 10000,
    'valid_every': 30000,
    'save_outputs': True,
    'save_true_depth': False,
    'train_fp16': True,
    'single_jitter': False,
    'num_samples': 48,
    'num_proposal_samples': [
        256,
        128
    ],
    'num_proposal_iterations': 2,
    'use_same_proposal_network': False,
    'use_proposal_weight_anneal': True,
    'proposal_net_args_list': [
        {
            'num_input_coords': 4,
            'num_output_coords': 8,
            'resolution': [
                128,
                128,
                128,
                150
            ]
        },
        {
            'num_input_coords': 4,
            'num_output_coords': 8,
            'resolution': [
                256,
                256,
                256,
                150
            ]
        }
    ],
    'concat_features_across_scales': True,
    'density_activation': 'trunc_exp',
    'flow_activation': 'tanh',
    'linear_decoder': False,
    'multiscale_res': [
        1,
        2,
        4,
        8
    ],
    'canonical_time': -2,
    'time_dependent_color': True,
    'grid_config': {
        'model_3d': {
            'grid_dimensions': 2,
            'input_coordinate_dim': 3,
            'output_coordinate_dim': 8,
            'resolution': [
                64,
                64,
                64
            ]
        },
        'model_bf': {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 8,
            'resolution': [
                64,
                64,
                64,
                150
            ]
        }
    }
}