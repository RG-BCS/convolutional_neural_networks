from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_image_generators(train_dir, test_dir, img_size=(227, 227), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse'
    )

    return train_gen, val_gen, test_gen

def plot_predictions(model, X_test, y_test, class_names_dict, row=4, col=10, figsize=(6, 4)):
    prob = model.predict(X_test, verbose=0)
    pred_labels = prob.argmax(axis=1)
    correct_pred_indx = np.where(pred_labels == y_test)[0]
    wrong_pred_indx = np.where(pred_labels != y_test)[0]

    print('\t\t\t\twrong classsifications\n'.upper())
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, wrong_indx in zip(range(len(axs)), wrong_pred_indx):
        axs[i].imshow(X_test[wrong_indx], cmap='Greys')
        axs[i].set_xticks(())
        axs[i].set_yticks(())
        axs[i].set_xlabel(f'Pred| {class_names_dict[pred_labels[wrong_indx]]}')
        axs[i].set_title(f'GT| {class_names_dict[y_test[wrong_indx]]}')
    plt.show()

    print('\t\t\t\tCorrect Classifications\n'.upper())
    _, axs = plt.subplots(row, col, figsize=figsize)
    axs = axs.flatten()
    for i, correct_indx in zip(range(len(axs)), correct_pred_indx):
        axs[i].imshow(X_test[correct_indx], cmap='Greys')
        axs[i].set_xticks(())
        axs[i].set_yticks(())
        axs[i].set_xlabel(f'Pred| {class_names_dict[pred_labels[correct_indx]]}')
        axs[i].set_title(f'GT| {class_names_dict[y_test[correct_indx]]}')
    plt.show()

def plot_confusion_matrix(model, X_test, y_test, class_names_dict):
    prob = model.predict(X_test, verbose=0)
    pred_labels = prob.argmax(axis=1)
    
    cm = confusion_matrix(y_test, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_dict.values())
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
