# dataset_generator.py
"""
Data generator setup for GoogLeNet image classification on the Intel Dataset.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

DEFAULT_IMAGE_SHAPE = (224, 224)
DEFAULT_CLASS_NUM = 6

# Default class labels for 6 classes (Intel dataset)
DEFAULT_CLASS_NAMES_DICT = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}

def get_data_generators(train_dir, test_dir, batch_size=32, num_classes=DEFAULT_CLASS_NUM, image_size=DEFAULT_IMAGE_SHAPE):
    """
    Creates train, validation, and test generators using ImageDataGenerator.

    Args:
        train_dir (str): Path to training dataset
        test_dir (str): Path to test dataset
        batch_size (int): Batch size
        num_classes (int): Number of classes
        image_size (tuple): Image size, e.g. (224, 224)

    Returns:
        train_gen: Training generator
        val_gen: Validation generator
        test_gen: Testing generator
        class_names_dict: Mapping of class index to class label
    """
    # If using default 6 classes, use default mapping, otherwise generate generic labels
    if num_classes == DEFAULT_CLASS_NUM:
        class_names_dict = DEFAULT_CLASS_NAMES_DICT
    else:
        class_names_dict = {i: f'class_{i}' for i in range(num_classes)}

    # Training and validation data augmentation
    train_aug = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Test data preprocessing
    test_aug = ImageDataGenerator(rescale=1./255)

    # Generators
    train_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    val_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=True
    )

    test_gen = test_aug.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )

    return train_gen, val_gen, test_gen, class_names_dict
