defaults = {
    # Architecture related
    'resnet': False,
    'img_size': 256,
    'depth': 4,
    'width': 16,
    'momentum': 0.9,
    'dropout': 0.2,
    'classes': 3,

    # Compilation related
    'lr': 0.01,

    # Training related
    'epochs': 150,
    'validation_freq': 1,
    'batch_size': 16,

    # Data-augmentation
    'border': True,
    'border_size': 2,
    'add_touching': False,
    'bit_depth': 16,
}
