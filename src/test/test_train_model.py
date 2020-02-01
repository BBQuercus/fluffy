import os
import sys
import pytest
import numpy as np
import tensorflow as tf

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models.train_model import \
    train_valid_split, \
    shuffle_lists, \
    standard_unet, \
    add_borders, \
    add_augmentation, \
    random_cropping, \
    normalize_image, \
    random_sample_generator


english = ['one', 'two', 'three']
spanish = ['uno', 'dos', 'tres']
mask_plain = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0]])
mask_borders = np.array([[0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0],
                [0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0],
                [0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0],
                [0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0],
                [0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0],
                [0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0],
                [0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0],
                [0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0],
                [0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0],
                [0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0],
                [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0],
                [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]])


def test_shuffle_lists():
    assert type(shuffle_lists(english, spanish)) == tuple
    assert type(shuffle_lists(english, spanish)[0]) == list
    assert type(shuffle_lists(english, spanish)[1]) == list

    with pytest.raises(TypeError):
        shuffle_lists(english, 1)
    with pytest.raises(TypeError):
        shuffle_lists('list', spanish)


def test_train_valid_split():
    output = [(['one', 'three'], ['two'], ['uno', 'tres'], ['dos']),
              (['one', 'two'], ['three'], ['uno', 'dos'], ['tres']),
              (['three', 'one'], ['two'], ['tres', 'uno'], ['dos']),
              (['three', 'two'], ['one'], ['tres', 'dos'], ['uno']),
              (['two', 'one'], ['three'], ['dos', 'uno'], ['tres']),
              (['two', 'three'], ['one'], ['dos', 'tres'], ['uno'])]

    assert train_valid_split(english, spanish, valid_split=0.25) in output

    with pytest.raises(TypeError):
        train_valid_split(english, 1, valid_split=0.25)
    with pytest.raises(TypeError):
        train_valid_split('list', spanish, valid_split=0.25)
    with pytest.raises(TypeError):
        train_valid_split(english, spanish, valid_split=None)
    with pytest.raises(ValueError):
        train_valid_split([''], [''])
    with pytest.raises(ValueError):
        train_valid_split(english, spanish, 1.8)


def test_standard_unet():
    input_size = 128
    try:
        model = standard_unet(input_size)
        assert type(model) == tf.keras.models.Model
        assert model.layers[0].output_shape == [(None, input_size, input_size, 1)]
        assert model.layers[-1].output_shape == (None, input_size, input_size, 3)
    except Exception:
        assert False

    with pytest.raises(ValueError):
        standard_unet(3)


def test_add_borders():
    assert (add_borders(mask_plain) == mask_borders).all()
    assert (add_borders(np.zeros((20, 20)) == np.zeros((20, 20)))).all()
    assert type(add_borders(mask_plain)) == np.ndarray
    assert len(np.unique(add_borders)) <= 3

    with pytest.raises(TypeError):
        add_borders(mask_plain, 'size')
    with pytest.raises(TypeError):
        add_borders('one', 2)


def test_add_augmentation():
    assert type(add_augmentation(mask_plain, mask_borders)) == tuple
    assert type(add_augmentation(mask_plain, mask_borders)[0]) == np.ndarray
    assert (add_augmentation(np.zeros((20, 20)), mask_borders)[0] == np.zeros((20, 20))).all()

    with pytest.raises(TypeError):
        add_augmentation('image', mask_borders)
    with pytest.raises(TypeError):
        add_augmentation(mask_plain, 1)


def test_random_cropping():
    assert type(random_cropping(mask_plain, mask_borders, 10)) == tuple
    assert type(random_cropping(mask_plain, mask_borders, 10)[0]) == np.ndarray
    assert random_cropping(mask_plain, mask_borders, 10)[0].shape[:2] == (10, 10)

    with pytest.raises(TypeError):
        random_cropping(mask_plain, mask_borders, 'crop_size')
    with pytest.raises(TypeError):
        random_cropping('image', 2, 2)
    with pytest.raises(ValueError):
        random_cropping(mask_plain, mask_borders, 0)


def test_normalize_image():
    assert (normalize_image(np.zeros((10, 10))) == np.zeros((10, 10))).all()

    with pytest.raises(TypeError):
        normalize_image('image', 24)
    with pytest.raises(TypeError):
        normalize_image(mask_plain, 'one')


def test_random_sample_generator():
    import types
    assert type(random_sample_generator(english, spanish, True, 16, 16, 256)) == types.GeneratorType

    with pytest.raises(TypeError):
        next(random_sample_generator(english, spanish, True, 'size', 'depth', 'size'))
    with pytest.raises(TypeError):
        next(random_sample_generator('list', spanish, True, 16, 16, 256))
    with pytest.raises(TypeError):
        next(random_sample_generator(english, 1, True, 16, 16, 256))
    with pytest.raises(ValueError):
        next(random_sample_generator(english[:1], spanish[:2], True, 16, 16, 256))
