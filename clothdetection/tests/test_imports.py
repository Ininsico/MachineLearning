def test_imports():
    try:
        import cv2
        import numpy as np
        from PIL import Image
        from ultralytics import YOLO
        import torch
        import sklearn
        import yaml
        print("All imports successful")
    except ImportError as e:
        print(f"Import failed: {e}")
        raise


def test_config():
    from src.utils.config import cfg
    cfg.load("config.yaml")
    assert cfg.get("project", "name") == "ClothMind AI"
    assert cfg.get("yolo", "epochs") == 50
    print("Config test passed")


def test_detector_init():
    from src.detection.person_detector import PersonDetector
    from src.detection.clothing_detector import ClothingDetector
    from src.detection.color_analyzer import ColorAnalyzer
    print("Detector imports successful")


def test_recommendation():
    from src.recommendation.engine import RecommendationEngine, UserProfile
    engine = RecommendationEngine()
    profile = UserProfile(mood="happy", weather="mild", season="summer")
    suggestions = engine.suggest(profile, n_suggestions=2)
    assert len(suggestions) == 2
    print(f"Recommendation test passed: {len(suggestions)} suggestions generated")


if __name__ == "__main__":
    test_imports()
    test_config()
    test_detector_init()
    test_recommendation()
    print("\nAll tests passed!")
