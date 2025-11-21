# =============================================================================
# PDAN8412 - Programming for Data Analytics 2
# Part 3: Image Recognition with Convolutional Neural Networks
# CIFAR-10 Dataset - CNN Implementation
# =============================================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("=== PDAN8412 - Part 3: CNN Image Recognition ===")
print("Loading libraries and setting up environment...")

# =============================================================================
# 1. DATA LOADING AND EXPLORATORY DATA ANALYSIS
# =============================================================================

def load_and_explore_data():
    """
    Load CIFAR-10 dataset and perform initial exploration
    """
    print("\n" + "="*50)
    print("1. LOADING AND EXPLORING CIFAR-10 DATASET")
    print("="*50)
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Basic dataset information
    print(f"\nDataset Dimensions:")
    print(f"Training images: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}")
    print(f"Test labels: {y_test.shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    # Check data types and value ranges
    print(f"\nData Types and Ranges:")
    print(f"Image data type: {x_train.dtype}")
    print(f"Label data type: {y_train.dtype}")
    print(f"Image value range: [{x_train.min()}, {x_train.max()}]")
    
    return (x_train, y_train), (x_test, y_test), class_names

def perform_eda(x_train, y_train, x_test, y_test, class_names):
    """
    Perform comprehensive Exploratory Data Analysis
    """
    print(f"\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Class distribution analysis
    print(f"\nClass Distribution Analysis:")
    train_counts = np.bincount(y_train.flatten())
    test_counts = np.bincount(y_test.flatten())
    
    distribution_df = pd.DataFrame({
        'Class': class_names,
        'Train_Count': train_counts,
        'Train_Percentage': (train_counts / len(y_train) * 100).round(2),
        'Test_Count': test_counts,
        'Test_Percentage': (test_counts / len(y_test) * 100).round(2)
    })
    print(distribution_df.to_string(index=False))
    
    # Pixel statistics
    print(f"\nPixel Statistics (Training Set):")
    print(f"Mean pixel value: {x_train.mean():.4f}")
    print(f"Standard deviation: {x_train.std():.4f}")
    print(f"Minimum pixel value: {x_train.min()}")
    print(f"Maximum pixel value: {x_train.max()}")
    
    return distribution_df

def visualize_dataset(x_train, y_train, class_names):
    """
    Create visualizations for dataset understanding
    """
    print(f"\nCreating dataset visualizations...")
    
    # 1. Class distribution plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    train_counts = np.bincount(y_train.flatten())
    plt.bar(range(10), train_counts, color='skyblue', alpha=0.7)
    plt.title('Training Set Class Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(range(10), class_names, rotation=45)
    
    # Add value labels on bars
    for i, count in enumerate(train_counts):
        plt.text(i, count + 50, str(count), ha='center', va='bottom')
    
    # 2. Sample images from each class
    plt.subplot(1, 2, 2)
    # Create a grid of sample images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(10):
        # Find first image of each class
        class_idx = np.where(y_train.flatten() == i)[0][0]
        axes[i].imshow(x_train[class_idx])
        axes[i].set_title(f'{class_names[i]}', fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 3. Pixel value distribution
    plt.figure(figsize=(10, 6))
    plt.hist(x_train.flatten(), bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Pixel Values (Training Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

# =============================================================================
# 2. DATA PREPROCESSING
# =============================================================================

def preprocess_data(x_train, x_test, y_train, y_test):
    """
    Preprocess the data for CNN training
    """
    print(f"\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Normalize pixel values to [0, 1]
    print("Normalizing pixel values...")
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical (one-hot encoding)
    print("Converting labels to categorical...")
    y_train_categorical = tf.keras.utils.to_categorical(y_train, 10)
    y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)
    
    # Split training data into training and validation sets
    print("Splitting data into training and validation sets...")
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train, y_train_categorical, test_size=0.2, random_state=42, stratify=y_train_categorical
    )
    
    print(f"Training set: {x_train_split.shape}")
    print(f"Validation set: {x_val.shape}")
    print(f"Test set: {x_test.shape}")
    
    return x_train_split, x_val, x_test, y_train_split, y_val, y_test_categorical

def create_data_augmentation():
    """
    Create data augmentation pipeline
    """
    print("\nCreating data augmentation pipeline...")
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    
    print("Data augmentation layers created:")
    for layer in data_augmentation.layers:
        print(f"  - {layer.name}")
    
    return data_augmentation

# =============================================================================
# 3. CNN MODEL ARCHITECTURE
# =============================================================================

def create_cnn_model(data_augmentation, input_shape=(32, 32, 3), num_classes=10):
    """
    Create the CNN model architecture
    """
    print(f"\n" + "="*50)
    print("CREATING CNN MODEL ARCHITECTURE")
    print("="*50)
    
    model = tf.keras.Sequential()
    
    # Add data augmentation as first layer
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(data_augmentation)
    
    # First convolutional block
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    # Second convolutional block
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    # Third convolutional block
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    # Classifier head
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # Display model summary
    print("\nModel Architecture Summary:")
    model.summary()
    
    return model

def compile_model(model):
    """
    Compile the CNN model with appropriate optimizer and loss function
    """
    print(f"\nCompiling model...")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model compiled with:")
    print(f"  - Optimizer: Adam (lr=0.001)")
    print(f"  - Loss function: categorical_crossentropy")
    print(f"  - Metrics: accuracy")
    
    return model

# =============================================================================
# 4. MODEL TRAINING
# =============================================================================

def train_model(model, x_train, y_train, x_val, y_val):
    """
    Train the CNN model with callbacks
    """
    print(f"\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_cnn_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("Training configuration:")
    print(f"  - Batch size: 64")
    print(f"  - Epochs: 50")
    print(f"  - Early stopping patience: 10")
    print(f"  - Learning rate reduction patience: 5")
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model

# =============================================================================
# 5. MODEL EVALUATION
# =============================================================================

def evaluate_model(model, history, x_test, y_test, class_names):
    """
    Comprehensive model evaluation
    """
    print(f"\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # 1. Evaluate on test set
    print("\n1. Test Set Evaluation:")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 2. Predictions for detailed analysis
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # 3. Classification report
    print(f"\n2. Detailed Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=class_names, digits=4))
    
    # 4. Confusion matrix
    print(f"\n3. Confusion Matrix Analysis:")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # 5. Training history visualization
    print(f"\n4. Training History Visualization:")
    plot_training_history(history)
    
    return test_accuracy, test_loss, y_pred_classes, y_true_classes

def plot_training_history(history):
    """
    Plot training history for accuracy and loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy During Training', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. MODEL INTERPRETATION AND VISUALIZATION
# =============================================================================

def model_interpretation(model, x_test, y_true_classes, y_pred_classes, class_names):
    """
    Provide model interpretation and error analysis
    """
    print(f"\n" + "="*50)
    print("MODEL INTERPRETATION AND ERROR ANALYSIS")
    print("="*50)
    
    # Calculate per-class accuracy
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Create accuracy by class dataframe
    accuracy_df = pd.DataFrame({
        'Class': class_names,
        'Accuracy': per_class_accuracy,
        'Percentage': (per_class_accuracy * 100).round(2)
    }).sort_values('Accuracy', ascending=False)
    
    print("\nAccuracy by Class (Sorted):")
    print(accuracy_df.to_string(index=False))
    
    # Visualize per-class performance
    plt.figure(figsize=(12, 6))
    bars = plt.bar(accuracy_df['Class'], accuracy_df['Percentage'], 
                   color=['green' if x > 75 else 'orange' if x > 65 else 'red' for x in accuracy_df['Percentage']])
    plt.title('Model Accuracy by Class', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracy_df['Percentage']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{accuracy}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Display some misclassified examples
    display_misclassified_examples(x_test, y_true_classes, y_pred_classes, class_names)

def display_misclassified_examples(x_test, y_true, y_pred, class_names, num_examples=10):
    """
    Display examples of misclassified images
    """
    print(f"\nDisplaying misclassified examples...")
    
    # Find misclassified indices
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    if len(misclassified_idx) > 0:
        # Select random misclassified examples
        selected_idx = np.random.choice(misclassified_idx, 
                                      size=min(num_examples, len(misclassified_idx)), 
                                      replace=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(selected_idx):
            axes[i].imshow(x_test[idx])
            axes[i].set_title(f'True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}', 
                            fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('Examples of Misclassified Images', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print(f"Total misclassified images: {len(misclassified_idx)}/{len(y_true)} "
              f"({len(misclassified_idx)/len(y_true)*100:.2f}%)")
    else:
        print("No misclassified images found!")

# =============================================================================
# 7. MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main function to execute the entire CNN pipeline
    """
    print("Starting CNN Image Recognition Pipeline...")
    
    try:
        # 1. Load and explore data
        (x_train, y_train), (x_test, y_test), class_names = load_and_explore_data()
        
        # 2. Perform EDA
        distribution_df = perform_eda(x_train, y_train, x_test, y_test, class_names)
        visualize_dataset(x_train, y_train, class_names)
        
        # 3. Preprocess data
        x_train_split, x_val, x_test_processed, y_train_split, y_val, y_test_categorical = preprocess_data(
            x_train, x_test, y_train, y_test
        )
        
        # 4. Create data augmentation
        data_augmentation = create_data_augmentation()
        
        # 5. Create and compile model
        model = create_cnn_model(data_augmentation)
        model = compile_model(model)
        
        # 6. Train model
        history, trained_model = train_model(model, x_train_split, y_train_split, x_val, y_val)
        
        # 7. Evaluate model
        test_accuracy, test_loss, y_pred_classes, y_true_classes = evaluate_model(
            trained_model, history, x_test_processed, y_test_categorical, class_names
        )
        
        # 8. Model interpretation
        model_interpretation(trained_model, x_test, y_true_classes, y_pred_classes, class_names)
        
        # 9. Final summary
        print(f"\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        print(f"🎯 Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"📉 Final Test Loss: {test_loss:.4f}")
        print(f"🏁 Model training completed successfully!")
        
        # Save the final model
        trained_model.save('final_cnn_model.h5')
        print(f"💾 Model saved as 'final_cnn_model.h5'")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise

# =============================================================================
# 8. ENHANCED MODEL (OPTIONAL - FOR IMPROVEMENTS)
# =============================================================================

def create_enhanced_cnn_model():
    """
    Create an enhanced CNN model with improvements
    Use this for retraining if initial model performance is unsatisfactory
    """
    print(f"\n" + "="*50)
    print("CREATING ENHANCED CNN MODEL")
    print("="*50)
    
    # Enhanced data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ])
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))
    model.add(data_augmentation)
    
    # Enhanced architecture with more filters
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    
    # Enhanced classifier
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Enhanced model created with:")
    print("  - More filters in convolutional layers")
    print("  - Enhanced data augmentation")
    print("  - Lower learning rate (0.0005)")
    print("  - Increased dropout rates")
    
    return model

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Execute main pipeline
    main()
    
    # Optional: Uncomment to run enhanced model
    # print("\n" + "="*70)
    # print("OPTIONAL: RUNNING ENHANCED MODEL FOR POTENTIAL IMPROVEMENT")
    # print("="*70)
    # enhanced_model = create_enhanced_cnn_model()