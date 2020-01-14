defaults = {
    # Architecture related
    'resnet': True,
    'img_size': 256,
    'depth': 3,
    'n_classes': 3,

    # Compilation related
    'lr': 0.01,

    # Training related
    'epochs': 100,
    'validation_freq': 2,
    'verbose': 0,

    # Data-preprocessing
    'bit_depth': 16,
    'batch_size': 32,
    'border_size': 2,
    'convert_to_rgb': False,
    'scaling': True,
    'cropping': True,
    'flipping': True,
    'padding': True,
    'rotation': True,
    'brightness': True,
    'contrast': True,
}
