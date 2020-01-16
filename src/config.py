defaults = {
    # Architecture related
    'resnet': False,
    'img_size': 256,
    'depth': 4,
    'width': 16,
    'classes': True,

    # Compilation related
    'lr': 0.01,

    # Training related
    'epochs': 150,
    'validation_freq': 1,
    'batch_size': 16,

    # Data-augmentation
    # Border (if n_classes == 1 not considered)
    'border': True,
    'border_size': 2,
    'touching_only': True,
    # 'convert_to_rgb': False,
    'bit_depth': 16,

    # 'scaling': True,
    # 'cropping': True,
    # 'flipping': True,
    # 'padding': True,
    # 'rotation': True,
    # 'brightness': True,
    # 'contrast': True,
}
