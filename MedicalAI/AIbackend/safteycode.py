import os
import sys
import gc
import json
import warnings
import psutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_auc_score, roc_curve, precision_recall_curve,
                            accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, mixed_precision
from tensorflow.keras.applications import (EfficientNetB0, DenseNet121, ResNet50,
                                          MobileNetV2, Xception, InceptionV3)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Try to import tqdm for progress bars
try:
    from tqdm.keras import TqdmCallback
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠ tqdm not available, using standard progress")

try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("✓ Mixed precision enabled")
except:
    print("⚠ Mixed precision not available")

# Check environment
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
IS_COLAB = 'COLAB_GPU' in os.environ
IS_LOCAL = not (IS_KAGGLE or IS_COLAB)

KAGGLE_WORKING = '/kaggle/working' if IS_KAGGLE else './outputs'
KAGGLE_INPUT = '/kaggle/input' if IS_KAGGLE else './data'

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'Local'}")
print(f"Output directory: {KAGGLE_WORKING}")

# ==================== SAFETY LAYERS ====================

class MedicalImageValidator:
    """Validate medical images before processing"""
    
    @staticmethod
    def validate_image(image: np.ndarray, expected_type: str = None) -> Dict[str, Any]:
        """Validate if image is suitable for medical analysis"""
        
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        if image is None or image.size == 0:
            results['valid'] = False
            results['errors'].append("Empty or invalid image")
            return results
        
        # 1. Basic image quality checks
        mean_brightness = np.mean(image)
        if mean_brightness < 0.1:
            results['warnings'].append("Image very dark (mean brightness < 0.1)")
            results['suggestions'].append("Adjust brightness/contrast")
        elif mean_brightness > 0.9:
            results['warnings'].append("Image very bright (mean brightness > 0.9)")
            results['suggestions'].append("Check for overexposure")
        
        # 2. Edge density check (medical images have structures)
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < 0.005:  # Too few edges
            results['warnings'].append("Image lacks detail (low edge density)")
            results['suggestions'].append("May not be a medical image or is blurry")
        
        # 3. Expected anatomy checks
        if expected_type == 'brain':
            # Brain MRI should be roughly circular/cranial shape
            height, width = image.shape[:2]
            circularity = min(height, width) / max(height, width)
            if circularity < 0.7:
                results['warnings'].append(f"Image aspect ratio {circularity:.2f} doesn't match typical brain MRI")
        
        elif expected_type == 'chest':
            # Chest X-ray should have distinct lung areas
            if len(image.shape) == 3:
                chest_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                chest_gray = (image * 255).astype(np.uint8)
            
            # Check for rib-like structures
            sobelx = cv2.Sobel(chest_gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(chest_gray, cv2.CV_64F, 0, 1, ksize=5)
            gradient_mag = np.sqrt(sobelx**2 + sobely**2)
            if np.mean(gradient_mag) < 10:
                results['warnings'].append("Image doesn't show typical chest anatomy")
        
        return results
    
    @staticmethod
    def detect_image_type(image: np.ndarray) -> Dict[str, float]:
        """Try to detect what type of medical image this is"""
        
        features = {}
        
        if len(image.shape) == 2 or image.shape[2] == 1:
            features['is_grayscale'] = 1.0
        else:
            features['is_grayscale'] = 0.0
        
        # Simple shape-based detection
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Brain MRI: often square-ish
        if 0.8 <= aspect_ratio <= 1.2:
            features['brain_probability'] = 0.6
        else:
            features['brain_probability'] = 0.2
        
        # Chest X-ray: often portrait orientation
        if aspect_ratio < 0.8:
            features['chest_probability'] = 0.7
        else:
            features['chest_probability'] = 0.3
        
        return features

class MedicalSafetyLayer:
    """Prevents wrong diagnoses by validating inputs"""
    
    def __init__(self):
        self.validator = MedicalImageValidator()
        self.confidence_threshold = 0.3  # Minimum confidence to give diagnosis
        self.quality_threshold = 0.5     # Minimum quality score
    
    def validate_before_diagnosis(self, image: np.ndarray, 
                                 requested_model: str) -> Dict[str, Any]:
        """Run all safety checks before diagnosis"""
        
        results = {
            'can_proceed': False,
            'error': None,
            'warning': None,
            'confidence': 0.0,
            'suggested_action': None
        }
        
        # 1. Basic image validation
        validation = self.validator.validate_image(image, requested_model)
        
        if not validation['valid']:
            results['error'] = "Invalid image: " + ", ".join(validation['errors'])
            return results
        
        # 2. Type detection
        detected_type = self.validator.detect_image_type(image)
        
        # Check if detected type matches requested model
        type_mismatch = False
        if requested_model == 'brain_tumor':
            if detected_type.get('brain_probability', 0) < 0.4:
                type_mismatch = True
                results['warning'] = "Image doesn't look like a brain MRI"
                results['suggested_action'] = "Check if you uploaded the correct scan type"
        
        elif requested_model == 'chest_xray':
            if detected_type.get('chest_probability', 0) < 0.4:
                type_mismatch = True
                results['warning'] = "Image doesn't look like a chest X-ray"
                results['suggested_action'] = "Check if you uploaded the correct scan type"
        
        # 3. Quality assessment
        quality_score = 1.0
        if validation['warnings']:
            quality_score = 0.7
            results['warning'] = "Image quality issues: " + ", ".join(validation['warnings'])
            results['suggested_action'] = validation['suggestions'][0] if validation['suggestions'] else None
        
        # 4. Decision
        if type_mismatch and quality_score < 0.6:
            results['error'] = "Wrong image type and poor quality"
            results['suggested_action'] = "Upload a proper medical image of the correct type"
        elif type_mismatch:
            results['warning'] = "Possible wrong image type - proceeding with caution"
            results['can_proceed'] = True
            results['confidence'] = 0.5
        elif quality_score < 0.5:
            results['error'] = "Image quality too poor for reliable diagnosis"
            results['suggested_action'] = "Retake or adjust the image"
        else:
            results['can_proceed'] = True
            results['confidence'] = 0.9
        
        return results

class OutOfDistributionDetector:
    """Detect if input is too different from training data"""
    
    def __init__(self, model_path=None):
        self.threshold = 0.3  # Probability threshold for OOD detection
        
    def is_out_of_distribution(self, probabilities: np.ndarray) -> bool:
        """Check if predictions indicate OOD sample"""
        
        if probabilities is None or len(probabilities) == 0:
            return True
        
        max_prob = np.max(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        # Low max probability = uncertain = likely OOD
        if max_prob < self.threshold:
            return True
        
        # High entropy = uniform distribution = likely OOD
        max_entropy = np.log(len(probabilities))
        if entropy > 0.8 * max_entropy:
            return True
        
        return False

# ==================== ENHANCED TRAINER ====================

class SafeMedicalTrainer:
    """Medical trainer with safety checks"""
    
    def __init__(self, dataset_name, data_dir=None, backbone='efficientnet',
                 auto_detect=True, enable_safety=True):
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.backbone = backbone
        self.enable_safety = enable_safety
        
        if enable_safety:
            self.safety_layer = MedicalSafetyLayer()
            self.ood_detector = OutOfDistributionDetector()
            print("✓ Safety layers enabled")
        
        if auto_detect and data_dir is None and IS_KAGGLE:
            data_dir = auto_detect_dataset_path(dataset_name)
        
        self.loader = MedicalDataLoader(dataset_name, data_dir)
        self.model = None
        self.history = None
        self.output_dir = Path(KAGGLE_WORKING) / f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
    def safe_predict(self, image_path: Union[str, Path], 
                    confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """Safe prediction with validation"""
        
        results = {
            'success': False,
            'diagnosis': None,
            'confidence': 0.0,
            'warnings': [],
            'errors': [],
            'is_ood': False
        }
        
        try:
            # Load image
            if self.config['color_mode'] == 'grayscale':
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    results['errors'].append("Failed to load image")
                    return results
                img = np.expand_dims(img, axis=-1)
            else:
                img = cv2.imread(str(image_path))
                if img is None:
                    results['errors'].append("Failed to load image")
                    return results
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = cv2.resize(img, (self.config['img_size'], self.config['img_size']))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Safety validation
            if self.enable_safety:
                safety_check = self.safety_layer.validate_before_diagnosis(img[0], self.dataset_name)
                
                if safety_check['error']:
                    results['errors'].append(safety_check['error'])
                    if safety_check['suggested_action']:
                        results['warnings'].append(safety_check['suggested_action'])
                    return results
                
                if safety_check['warning']:
                    results['warnings'].append(safety_check['warning'])
                
                if not safety_check['can_proceed']:
                    results['errors'].append("Safety check failed")
                    return results
            
            # Get prediction
            predictions = self.model.predict(img, verbose=0)
            probabilities = predictions[0]
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            # OOD detection
            if self.enable_safety:
                is_ood = self.ood_detector.is_out_of_distribution(probabilities)
                if is_ood:
                    results['is_ood'] = True
                    results['warnings'].append("Input appears different from training data")
                    results['warnings'].append("Low confidence in diagnosis")
            
            # Confidence threshold
            if confidence < confidence_threshold:
                results['warnings'].append(f"Low confidence ({confidence:.2f}) - consider human review")
            
            if confidence < 0.1:
                results['errors'].append("Confidence too low for reliable diagnosis")
                return results
            
            # Prepare results
            results['success'] = True
            results['diagnosis'] = {
                'class': self.config['classes'][predicted_class],
                'class_index': int(predicted_class),
                'confidence': confidence,
                'probabilities': {self.config['classes'][i]: float(prob) 
                                 for i, prob in enumerate(probabilities)}
            }
            results['confidence'] = confidence
            
        except Exception as e:
            results['errors'].append(f"Prediction error: {str(e)}")
        
        return results
    
    def test_wrong_inputs(self):
        """Test how model handles wrong inputs"""
        print(f"\n{'='*70}")
        print("Testing Wrong Input Handling")
        print(f"{'='*70}")
        
        test_cases = []
        
        # Test 1: Non-medical image
        print("\n1. Testing with non-medical image...")
        # Create a random image (not medical)
        random_img = np.random.rand(224, 224, 3).astype(np.float32)
        result = self.safe_predict_from_array(random_img)
        print(f"   Result: {'Success' if result['success'] else 'Failed'}")
        print(f"   Warnings: {result['warnings']}")
        
        # Test 2: Wrong modality
        print("\n2. Testing with wrong modality...")
        # Create an image that looks like wrong type
        if self.dataset_name == 'brain_tumor':
            # Create something that looks like chest X-ray (vertical lines)
            wrong_img = np.zeros((224, 224, 3), dtype=np.float32)
            for i in range(0, 224, 20):
                wrong_img[:, i:i+5, :] = 0.8
            result = self.safe_predict_from_array(wrong_img)
            print(f"   Result: {'Success' if result['success'] else 'Failed'}")
            print(f"   Warnings: {result['warnings']}")
        
        return test_cases
    
    def safe_predict_from_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Predict from numpy array with safety checks"""
        img = np.expand_dims(image_array, axis=0)
        
        results = {
            'success': False,
            'diagnosis': None,
            'confidence': 0.0,
            'warnings': [],
            'errors': []
        }
        
        # Safety validation
        if self.enable_safety:
            safety_check = self.safety_layer.validate_before_diagnosis(image_array, self.dataset_name)
            
            if safety_check['error']:
                results['errors'].append(safety_check['error'])
                return results
            
            if safety_check['warning']:
                results['warnings'].append(safety_check['warning'])
            
            if not safety_check['can_proceed']:
                results['errors'].append("Safety check failed")
                return results
        
        # Get prediction
        try:
            predictions = self.model.predict(img, verbose=0)
            probabilities = predictions[0]
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            # Confidence threshold
            if confidence < 0.3:
                results['warnings'].append(f"Low confidence ({confidence:.2f})")
            
            results['success'] = True
            results['diagnosis'] = {
                'class': self.config['classes'][predicted_class],
                'confidence': confidence
            }
            results['confidence'] = confidence
            
        except Exception as e:
            results['errors'].append(f"Prediction error: {str(e)}")
        
        return results

# ==================== ENHANCED MODEL BUILDER ====================

class SafeMedicalModelBuilder(MedicalModelBuilder):
    """Medical model builder with safety features"""
    
    @staticmethod
    def build_model(config, backbone='efficientnet', dropout_rate=0.5):
        """Build model with proper mixed precision handling"""
        
        backbone_config = BACKBONE_CONFIGS.get(backbone, BACKBONE_CONFIGS['efficientnet'])
        img_size = backbone_config['size']
        channels = 1 if config['color_mode'] == 'grayscale' else 3
        num_classes = len(config['classes'])
        
        print(f"\nBuilding {backbone} model with safety features...")
        
        inputs = layers.Input(shape=(img_size, img_size, channels), name='input')
        
        if channels == 1 and backbone != 'simple':
            x = layers.Conv2D(3, 1, padding='same', name='gray_to_rgb')(inputs)
        else:
            x = inputs
        
        if backbone != 'simple':
            base_model = backbone_config['model'](
                include_top=False,
                weights='imagenet',
                input_tensor=x,
                pooling='avg'
            )
            
            trainable_layers = backbone_config['trainable_layers']
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
            
            features = base_model.output
            
        else:
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization(dtype='float32')(x)  # Fixed for mixed precision
            x = layers.MaxPooling2D(2)(x)
            
            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization(dtype='float32')(x)
            x = layers.MaxPooling2D(2)(x)
            
            x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization(dtype='float32')(x)
            x = layers.MaxPooling2D(2)(x)
            
            x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization(dtype='float32')(x)
            
            features = layers.GlobalAveragePooling2D()(x)
        
        # Add uncertainty estimation
        x = layers.Dropout(dropout_rate)(features)
        x = layers.Dense(512, activation='relu', name='fc1')(x)
        x = layers.BatchNormalization(dtype='float32')(x)  # Fixed
        x = layers.Dropout(dropout_rate * 0.6)(x)
        x = layers.Dense(256, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization(dtype='float32')(x)  # Fixed
        x = layers.Dropout(dropout_rate * 0.4)(x)
        
        # Output with temperature scaling for better uncertainty
        outputs = layers.Dense(num_classes, activation='softmax', 
                              dtype='float32', name='output')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        return model

# ==================== MAIN EXECUTION ====================

def main_cli():
    """Command-line interface without input() calls"""
    
    parser = argparse.ArgumentParser(description='Medical AI Classifier')
    parser.add_argument('--dataset', type=str, default='brain_tumor',
                       choices=['brain_tumor', 'chest_xray', 'diabetic_retinopathy'],
                       help='Dataset to use')
    parser.add_argument('--backbone', type=str, default='efficientnet',
                       choices=list(BACKBONE_CONFIGS.keys()),
                       help='Model backbone')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset (auto-detected if not specified)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'cross_validate', 'test_safety'],
                       help='Operation mode')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per class')
    parser.add_argument('--enable_safety', action='store_true',
                       help='Enable safety layers')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    MEDICAL AI CLASSIFIER - SAFE EDITION                       ║
    ║                     With Input Validation & Safety Checks                    ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Mode: {args.mode}")
    print(f"  Safety: {'Enabled' if args.enable_safety else 'Disabled'}")
    
    MemoryMonitor.print_memory_usage("Initial")
    
    try:
        if args.enable_safety:
            trainer = SafeMedicalTrainer(args.dataset, args.data_dir, args.backbone, 
                                        enable_safety=True)
        else:
            trainer = MedicalTrainer(args.dataset, args.data_dir, args.backbone)
        
        if args.mode == 'cross_validate':
            trainer.cross_validate(n_splits=args.n_folds, epochs=args.epochs, 
                                  batch_size=args.batch_size)
        elif args.mode == 'test_safety' and hasattr(trainer, 'test_wrong_inputs'):
            trainer.train(epochs=args.epochs, batch_size=args.batch_size,
                         max_samples_per_class=args.max_samples)
            trainer.test_wrong_inputs()
        else:
            trainer.train(epochs=args.epochs, batch_size=args.batch_size,
                         max_samples_per_class=args.max_samples)
            trainer.evaluate()
        
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"\nAll outputs saved to: {trainer.output_dir}")
        
        MemoryMonitor.print_memory_usage("Final")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

# For Kaggle/Colab notebooks
def quick_start():
    """Quick start for notebooks"""
    print("Quick start: Training brain tumor classifier with safety checks")
    
    trainer = SafeMedicalTrainer(
        dataset_name='brain_tumor',
        backbone='efficientnet',
        enable_safety=True
    )
    
    trainer.train(epochs=30, batch_size=32, max_samples_per_class=200)
    results = trainer.evaluate()
    
    # Test safety features
    print("\nTesting safety features...")
    trainer.test_wrong_inputs()
    
    return trainer

if __name__ == "__main__":
    # Check if running in notebook environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            print("Running in notebook mode - use quick_start()")
            # Don't run CLI in notebook
        else:
            main_cli()
    except:
        main_cli()