defaults = {
    # Architecture related
    'resnet': False,
    'img_size': 256,
    'depth': 4,
    'width': 16,
    'n_classes': 1,  # Binary: 1, Multiclass: Background + ...

    # Compilation related
    'lr': 0.01,

    # Training related
    'epochs': 1,
    'validation_freq': 1,
    'batch_size': 1,

    # Data-augmentation
    # Border (if n_classes == 1 not considered)
    # 'border_size': 2,
    # 'touching_only': True,
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
