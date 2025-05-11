import os
import sys
import json

def check_model_files():
    """Check if all model files exist and report their sizes"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'model')
    
    required_files = [
        'saved_model.pb',
        'variables/variables.index',
        'variables/variables.data-00000-of-00001'
    ]
    
    results = {
        "model_directory_exists": os.path.exists(model_dir),
        "model_directory_path": model_dir,
        "files": {}
    }
    
    if not results["model_directory_exists"]:
        return results
    
    # Check each required file
    for file_path in required_files:
        full_path = os.path.join(model_dir, file_path)
        file_exists = os.path.exists(full_path)
        
        file_info = {
            "exists": file_exists,
            "path": full_path
        }
        
        if file_exists:
            file_info["size_bytes"] = os.path.getsize(full_path)
            file_info["size_mb"] = round(file_info["size_bytes"] / (1024 * 1024), 2)
        
        results["files"][file_path] = file_info
    
    return results

def try_load_model():
    """Try to load the TensorFlow model and report success/failure"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'model')
        
        # Import TensorFlow
        print("Importing TensorFlow...")
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Disable GPU to save memory
        print("Configuring TensorFlow to use CPU only...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        tf.config.set_visible_devices([], 'GPU')
        
        # Try to load the model
        print(f"Loading model from {model_path}")
        saved_model = tf.saved_model.load(model_path)
        model = saved_model.signatures["serving_default"]
        
        # Get model info
        input_spec = model.structured_input_signature[1]
        input_names = list(input_spec.keys())
        
        return {
            "success": True,
            "message": "Model loaded successfully",
            "input_layer_names": input_names
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    # Check if model files exist
    print("Checking model files...")
    file_check = check_model_files()
    print(json.dumps(file_check, indent=2))
    
    # Try to load the model
    print("\nTrying to load the model...")
    load_result = try_load_model()
    print(json.dumps(load_result, indent=2))
