"""Resolved, deliberately non-tunable scientific settings for the accepted v2 runs."""
from copy import deepcopy

from .subcell_manifest import PROTOCOL


def model_config(family):
    if family not in ('mae', 'vit'):
        raise ValueError('Expected mae or vit')
    backbone = dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
                    hidden_act='gelu', hidden_dropout_prob=0., attention_probs_dropout_prob=0.,
                    initializer_range=.02, layer_norm_eps=1e-12, image_size=448, patch_size=16,
                    num_channels=4, qkv_bias=True)
    contrastive = {'name': 'ContrastiveLoss', 'args': {
        'projector': {'name': 'ProjectionHead', 'args': {'in_channels': 1536, 'mlp_layers': [8192, 8192, 512],
                                                       'add_bn': True, 'avg_pool': False, 'normalize': False}},
        'temperature': .1}}
    config = {'supcon_model': deepcopy(contrastive),
              'pool_model': {'name': 'GatedAttentionPooler', 'args': {
                  'dim': 768, 'int_dim': 512, 'num_heads': 2, 'dropout': .2}}, 'pl_args': {}}
    if family == 'mae':
        backbone.update(decoder_num_attention_heads=16, decoder_hidden_size=512, decoder_num_hidden_layers=8,
                        decoder_intermediate_size=2048, mask_ratio=.25, norm_pix_loss=True, object_mask_ratio=0.)
        config['mae_model'] = {'name': 'ViTMAEForPreTraining', 'args': backbone}
        config['ssl_model'] = deepcopy(contrastive)
    else:
        config['vit_model'] = {'name': 'ViTModel', 'args': backbone}
    return config


def resolved_protocol(family):
    return {'protocol': PROTOCOL, 'model': model_config(family),
            'channels': ['agp', 'mito', 'dna', 'gfp'], 'geometry': [128, 955, 253, 448],
            'normalization': 'joint_per_cell_minmax_eps_1e-6',
            'global_alleles': 16, 'cells_per_allele': 8, 'global_cells': 128, 'views': 2,
            'mask_prob': 0, 'return_cell_mask': False, 'object_mask_ratio': 0,
            'mae_view1_mask_ratio': .25 if family == 'mae' else 0,
            'view2_validation_extraction_mask_ratio': 0,
            'mae_zero_mask_order': {'training': 'upstream_random_shuffle', 'evaluation': 'identity'} if family == 'mae' else None,
            'intensity_views': [2] if family == 'mae' else [1, 2],
            'augmentations': {
                'horizontal_flip_p': .5, 'vertical_flip_p': .5, 'geometry_choice': [.5, .5],
                'affine': {'degrees': 90, 'translate': [.2, .2], 'scale': [.8, 1.2]},
                'perspective': {'distortion': .25, 'p': .5}, 'interpolation': 'bilinear', 'fill': 0,
                'intensity_order': ['remove_channel', 'rescale_gfp', 'jitter', 'blur_or_sharpness', 'noise', 'erase'],
                'remove_non_gfp_p': .25, 'rescale_gfp_p': .25,
                'rescale_formula': 'GFP * (2 * numpy_uniform / (current_image_max + 1e-6))',
                'jitter': {'brightness': .5, 'contrast': .5, 'per_channel_p': 1.},
                'blur_or_sharpness_choice': [.5, .5],
                'blur': {'kernel': 7, 'sigma': [.1, 2.], 'per_channel_p': .5},
                'sharpness': {'factor': 2, 'outer_per_channel_p': .5, 'inner_p': .5},
                'noise': {'p': .5, 'per_channel_sigma': [.01, .05]},
                'erase': {'area': [.02, .1], 'aspect': [.3, 3.3], 'outer_per_channel_p': .5, 'inner_p': .5},
                'post_pipeline_clip_or_normalization': False},
            'metric_statistic_dtype': 'float64', 'model_and_probability_dtype': 'float32',
            'loss_weights': {'reconstruction': 1 if family == 'mae' else 0,
                             'cell_contrastive': 1 if family == 'mae' else 0,
                             'allele_supcon': .1 if family == 'mae' else 1, 'detached_ce': 1},
            'optimizer': 'AdamW', 'betas': [.9, .95], 'eps': 1e-8, 'base_lr': 1e-4, 'peak_lr': 5e-5,
            'encoder_decoder_decay': [.05, 0], 'auxiliary_decay': .01,
            'max_passes': 100, 'warmup_passes': 5, 'lr_floor_factor': .001,
            'accumulate_grad_batches': 1, 'clip_global_norm': 1., 'precision': '32-true',
            'matmul_precision': 'high', 'validation_every': 10, 'validation_seed': 2026,
            'validation_cells_per_allele': 32, 'selector': 'macro_allele_average_precision',
            'early_stopping_patience_checks': 5, 'early_stopping_min_delta': .001}


def validate_config(config):
    keys = {'protocol', 'family', 'seed', 'sampling_seed', 'preflight', 'pretrained_weights',
            'pretrained_sha256', 'output', 'workers', 'devices', 'mask_prob', 'return_cell_mask', 'object_mask_ratio'}
    if set(config) != keys:
        raise ValueError(f'Unsupported/missing v2 config fields: {set(config) ^ keys}; historical YAMLs are rejected')
    if config['protocol'] != PROTOCOL or config['family'] not in ('mae', 'vit') or config['seed'] not in (42, 43, 44):
        raise ValueError('Expected protocol v2, mae/vit, and a fresh seed 42/43/44')
    if config['mask_prob'] != 0 or config['return_cell_mask'] is not False or config['object_mask_ratio'] != 0:
        raise ValueError('Cell/background/object masks are forbidden')
    if not isinstance(config['devices'], int) or config['devices'] < 1 or 16 % config['devices']:
        raise ValueError('Device count must divide 16')
    if not isinstance(config['sampling_seed'], int) or config['sampling_seed'] < 0 or config['workers'] not in (0, 1, 2):
        raise ValueError('Expected nonnegative sampling seed and bounded worker count (0/1/2)')
