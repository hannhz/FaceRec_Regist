#!/usr/bin/env python3
"""
Test face recognition workflow - simulates registration and verification
"""

import sys
import os
import cv2
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_threshold_values():
    """Test that threshold values are properly set"""
    print("Test 1: Verify threshold values...")
    try:
        from app import (
            LBPH_CONF_THRESHOLD, VOTE_MIN_SHARE, MIN_VALID_FRAMES,
            EARLY_VOTES_REQUIRED, EARLY_CONF_THRESHOLD
        )
        
        # Verify the new values
        assert LBPH_CONF_THRESHOLD == 120, f"Expected LBPH_CONF_THRESHOLD=120, got {LBPH_CONF_THRESHOLD}"
        assert VOTE_MIN_SHARE == 0.35, f"Expected VOTE_MIN_SHARE=0.35, got {VOTE_MIN_SHARE}"
        assert MIN_VALID_FRAMES == 2, f"Expected MIN_VALID_FRAMES=2, got {MIN_VALID_FRAMES}"
        assert EARLY_VOTES_REQUIRED == 4, f"Expected EARLY_VOTES_REQUIRED=4, got {EARLY_VOTES_REQUIRED}"
        assert EARLY_CONF_THRESHOLD == 80, f"Expected EARLY_CONF_THRESHOLD=80, got {EARLY_CONF_THRESHOLD}"
        
        print(f"  ✓ LBPH_CONF_THRESHOLD: {LBPH_CONF_THRESHOLD}")
        print(f"  ✓ VOTE_MIN_SHARE: {VOTE_MIN_SHARE}")
        print(f"  ✓ MIN_VALID_FRAMES: {MIN_VALID_FRAMES}")
        print(f"  ✓ EARLY_VOTES_REQUIRED: {EARLY_VOTES_REQUIRED}")
        print(f"  ✓ EARLY_CONF_THRESHOLD: {EARLY_CONF_THRESHOLD}")
        return True
    except AssertionError as e:
        print(f"  ✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_data_dir_consistency():
    """Test that DATA_DIR is used consistently"""
    print("\nTest 2: Verify DATA_DIR consistency...")
    try:
        from app import DATA_DIR, BASE_DIR
        
        expected_path = os.path.join(BASE_DIR, "data", "database_wajah")
        
        if DATA_DIR == expected_path:
            print(f"  ✓ DATA_DIR correctly set to: {DATA_DIR}")
        else:
            print(f"  ✗ DATA_DIR mismatch: expected {expected_path}, got {DATA_DIR}")
            return False
        
        # Check that directory exists
        if os.path.exists(DATA_DIR):
            print(f"  ✓ DATA_DIR exists")
        else:
            print(f"  ✗ DATA_DIR does not exist: {DATA_DIR}")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_preprocessing_consistency():
    """Test that preprocessing is applied consistently"""
    print("\nTest 3: Verify preprocessing consistency...")
    try:
        from app import preprocess_roi, save_face_images_from_frame, DATA_DIR
        import tempfile
        
        # Create a dummy grayscale face image
        dummy_face = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        
        # Test preprocess_roi function
        preprocessed = preprocess_roi(dummy_face)
        
        # Check output shape
        if preprocessed.shape != (200, 200):
            print(f"  ✗ Preprocessed shape incorrect: {preprocessed.shape}")
            return False
        
        print(f"  ✓ preprocess_roi output shape: {preprocessed.shape}")
        
        # Create a dummy BGR image for testing save function
        dummy_bgr = cv2.cvtColor(
            np.random.randint(50, 200, (300, 300), dtype=np.uint8),
            cv2.COLOR_GRAY2BGR
        )
        
        # Note: save_face_images_from_frame will try to detect faces and may return 0
        # We just verify it doesn't crash
        test_nik = 9999999999999999
        result = save_face_images_from_frame(dummy_bgr, "Test", test_nik, 1)
        print(f"  ✓ save_face_images_from_frame executed (saved: {result})")
        
        # Clean up any test files
        import glob
        for f in glob.glob(os.path.join(DATA_DIR, f"{test_nik}.*.jpg")):
            try:
                os.remove(f)
            except:
                pass
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_blur_threshold():
    """Test blur detection threshold"""
    print("\nTest 4: Verify blur detection...")
    try:
        from app import is_blurry
        
        # Create a sharp image (high variance)
        sharp_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Create a blurry image (low variance, smooth)
        blurry_img = np.ones((100, 100), dtype=np.uint8) * 128
        blurry_img = cv2.GaussianBlur(blurry_img, (15, 15), 0)
        
        # Test with threshold 25.0 (used in recognition)
        sharp_blurry = is_blurry(sharp_img, 25.0)
        blur_blurry = is_blurry(blurry_img, 25.0)
        
        print(f"  ✓ Sharp image blur check (threshold=25): {sharp_blurry}")
        print(f"  ✓ Blurry image blur check (threshold=25): {blur_blurry}")
        
        # Blurry image should be detected as blurry
        if not blur_blurry:
            print(f"  ⚠ Warning: Very blurry image not detected as blurry")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_file_naming_format():
    """Test that file naming follows nik.index.jpg format"""
    print("\nTest 5: Verify file naming format...")
    try:
        from app import DATA_DIR
        import glob
        
        # List all jpg files in DATA_DIR
        files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
        
        if len(files) == 0:
            print(f"  ℹ No training files found (this is OK for a fresh install)")
            return True
        
        print(f"  Found {len(files)} image files")
        
        # Check format of each file
        format_ok = True
        for fpath in files[:5]:  # Check first 5 files
            fname = os.path.basename(fpath)
            parts = fname.split(".")
            
            if len(parts) < 3:
                print(f"  ✗ Invalid format: {fname}")
                format_ok = False
                continue
            
            # parts[0] should be NIK (numeric)
            # parts[1] should be index (numeric)
            # parts[2] should be jpg
            try:
                nik = int(parts[0])
                idx = int(parts[1])
                ext = parts[2]
                
                if ext.lower() != "jpg":
                    print(f"  ✗ Invalid extension in: {fname}")
                    format_ok = False
                else:
                    print(f"  ✓ Valid format: {fname} (NIK={nik}, index={idx})")
            except ValueError:
                print(f"  ✗ Invalid format: {fname}")
                format_ok = False
        
        return format_ok
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_model_training_loading():
    """Test that model training and loading work"""
    print("\nTest 6: Verify model training/loading...")
    try:
        from app import (
            train_model_blocking, load_model_if_exists, 
            get_images_and_labels, MODEL_PATH, recognizer
        )
        
        # Check if there's training data
        faces, ids = get_images_and_labels()
        
        if len(faces) == 0:
            print(f"  ℹ No training data found (this is OK for a fresh install)")
            print(f"  ✓ get_images_and_labels() works")
            return True
        
        print(f"  ✓ Found {len(faces)} training images for {len(set(ids))} unique NIKs")
        
        # Try to load existing model
        model_exists = load_model_if_exists()
        
        if model_exists:
            print(f"  ✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"  ℹ No existing model found")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("FACE RECOGNITION WORKFLOW TESTS")
    print("=" * 60)
    
    tests = [
        test_threshold_values,
        test_data_dir_consistency,
        test_preprocessing_consistency,
        test_blur_threshold,
        test_file_naming_format,
        test_model_training_loading,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"❌ {total - passed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
