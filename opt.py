import argparse
from pathlib import Path

def get_opts():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'data_path', type=Path,
        help=('Path to the datasets. This should point to the location with '
              '`megadepth` and `imw2020-val` directories.'),
    )
    parser.add_argument(
        '--num-gpus', type=int, default=1,
        help='Number of GPUs per node',
    )
    parser.add_argument(
        '--num-nodes', type=int, default=1,
        help='Number of node',
    )
    parser.add_argument(
        '--window', type=int, default=8,
        help='Windows size',
    )
    parser.add_argument(
        '--precision', type=str, default="32-true",
        help='"16-mixed" for automatic mixed precision ',
    )
    parser.add_argument(
        '--exp-name', type=str,
        help='Experiment name prefix',
    )
    parser.add_argument(
        '--freeze', type=str, default="none",
        help='Model parts to freeze',
    )
    parser.add_argument(
        '--reward', choices=['epipolar', 'depth'], default='depth',
        help='Reward criterion to use'
    )
    parser.add_argument(
        '--backbone', choices=['unet', 'dust3r', 'mickey'], default='unet',
        help='Backbone architecture to use'
    )
    parser.add_argument(
        '--optimizer', choices=['adam', 'adamw'], default='adam',
        help='Optimizer'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--save-dir', type=Path, default='artifacts',
        help=('Path for saving artifacts (checkpoints and tensorboard logs). Will '
              'be created if doesn\'t exist')
    )
    parser.add_argument(
        '--batch-size', type=int, default=2,
        help='The size of the batch',
    )
    parser.add_argument(
        '--chunk-size', type=int, default=5000,
        help=('The number of batches in the (pseudo) epoch. We run validation and '
              'save a checkpoint once per epoch, as well as use them for scheduling'
              ' the reward annealing'),
    )
    parser.add_argument(
        '--substep', type=int, default=1,
        help=('Number of batches to accumulate gradients over. Can be increased to'
              ' compensate for smaller batches on GPUs with less VRAM'),
    )
    parser.add_argument(
        '--warmup', type=int, default=250,
        help=('The first (pseudo) epoch can be much shorter, this avoids wasting '
              'time.'),
    )
    parser.add_argument(
        '--height', type=int, default=768,
        help='We train on images resized to (height, width)',
    )
    parser.add_argument(
        '--width', type=int, default=768,
        help='We train on images resized to (height, width)',
    )
    parser.add_argument(
        '--train-scene-limit', type=int, default=1000,
        help=('Different scenes in the dataset have a different amount of '
              'covisible image triplets. We (randomly) subselect '
              '--train-scene-limit of them for training, to avoid introducing '
              'a data bias towards those scenes.')
    )
    parser.add_argument(
        '--test-scene-limit', type=int, default=250,
        help=('Different scenes in the dataset have a different amount of '
              'covisible image triplets. We (randomly) subselect '
              '--test-scene-limit of them for validation to avoid '
              'to avoid introducing a bias towards those scenes.')
    )
    parser.add_argument(
        '--num-epochs', type=int, default=50,
        help='Number of (pseudo) epochs to train for',
    )
    parser.add_argument(
        '--desc-dim', type=int, default=128,
        help='Dimensionality of descriptors to produce. 128 by default',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to a checkpoint to resume training from',
    )
    parser.add_argument(
        '--pretrained-weight', type=str, default=None,
        help='Path to pretrained model weight to load',
    )
    parser.add_argument(
        '--penalty', type=float, default=1e-1,
        help='Penalty value to ramp up from',
    )
    parser.add_argument(
        '--debug', action=argparse.BooleanOptionalAction,
        help='Debug mode',
    )
    parser.add_argument(
        '--tf32', action=argparse.BooleanOptionalAction,
        help='Tensor core',
    )
    parser.add_argument(
        '--compile', action=argparse.BooleanOptionalAction,
        help='Torch compile model',
    )
    args = parser.parse_args()
    return args