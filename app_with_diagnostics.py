import io
import json
import numpy as np
import os
import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Get the absolute path to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"API starting in directory: {SCRIPT_DIR}")
# Path to the SavedModel directory
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model')

# Lazy-load TensorFlow and the model
_tf = None
_model = None

def get_tf():
    global _tf
    if _tf is None:
        print("Importing TensorFlow...")
        import tensorflow as tf
        _tf = tf
        # Set TensorFlow to only use CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        _tf.config.set_visible_devices([], 'GPU')
    return _tf

def get_model():
    global _model, _tf
    if _model is None:
        tf = get_tf()
        print(f"Loading model from {MODEL_PATH}")
        saved_model = tf.saved_model.load(MODEL_PATH)
        _model = saved_model.signatures["serving_default"]
        
        # Get input names from the signature
        input_spec = _model.structured_input_signature[1]
        input_names = list(input_spec.keys())
        print(f"Model has these input layer names: {input_names}")
    return _model

# Create FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/file-check")
async def file_check():
    """Check if model files exist without attempting to load the model"""
    try:
        model_dir = MODEL_PATH
        
        required_files = [
            'saved_model.pb',
            'variables/variables.index',
            'variables/variables.data-00000-of-00001'
        ]
        
        results = {
            "model_directory_exists": os.path.exists(model_dir),
            "model_directory_path": model_dir,
            "files": {},
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cwd": os.getcwd(),
                "env_vars": {k: v for k, v in os.environ.items() if k.startswith(("TF_", "PYTHON", "PATH"))},
            }
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
        
        # Add directory listing
        results["directory_listing"] = []
        for root, dirs, files in os.walk(model_dir):
            rel_path = os.path.relpath(root, model_dir)
            if rel_path == '.':
                rel_path = ''
            
            for file in files:
                file_path = os.path.join(rel_path, file)
                size = os.path.getsize(os.path.join(root, file))
                results["directory_listing"].append({
                    "path": file_path,
                    "size": size,
                    "size_mb": round(size / (1024 * 1024), 2)
                })
        
        return results
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/tf-info")
async def tf_info():
    """Get TensorFlow info without loading the model"""
    try:
        # Import TensorFlow
        print("Importing TensorFlow...")
        import tensorflow as tf
        
        return {
            "tf_version": tf.__version__,
            "tf_path": tf.__file__,
            "devices": [device.name for device in tf.config.list_physical_devices()],
            "success": True
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False
        }

@app.get("/model-check")
async def model_check():
    """Diagnostic endpoint to check if model files exist and can be loaded"""
    # Check if model files exist
    model_files = check_model_files()
    
    # Try to load the model if all files exist
    if all(file_info.get("exists", False) for file_info in model_files["files"].values()):
        try:
            tf = get_tf()
            model = get_model()
            model_loading = {
                "success": True,
                "message": "Model loaded successfully",
                "model_input_names": list(model.structured_input_signature[1].keys())
            }
        except Exception as e:
            import traceback
            model_loading = {
                "success": False,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
    else:
        model_loading = {
            "success": False,
            "message": "Not attempted - missing model files"
        }
    
    return {
        "model_files": model_files,
        "model_loading": model_loading
    }

def check_model_files():
    """Check if all model files exist and report their sizes"""
    model_dir = MODEL_PATH
    
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

def preprocess_image(file: UploadFile):
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert('RGB')
        img = img.resize((224,224))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr
    except Exception as e:
        raise HTTPException(400, f"Can't parse image {file.filename}: {str(e)}")

@app.post("/predict")
async def predict(
    ap:  UploadFile = File(...),
    lat: UploadFile = File(...),
    ob:  UploadFile = File(...)
):
    try:
        # 1) Read & preprocess images
        def preprocess_image(file):
            try:
                img = Image.open(io.BytesIO(file.file.read())).convert('RGB')
                img = img.resize((224,224))
                arr = np.array(img, dtype=np.float32) / 255.0
                return arr
            except Exception as e:
                raise HTTPException(400, f"Can't parse image {file.filename}: {str(e)}")
            
        imgs = [preprocess_image(f) for f in (ap, lat, ob)]
        batch = [np.expand_dims(img, 0) for img in imgs]
        
        # 2) Get TensorFlow and model (lazy-loaded)
        tf = get_tf()
        model = get_model()
        
        # Get input names from the model
        input_spec = model.structured_input_signature[1]
        input_names = list(input_spec.keys())
        
        # 3) Run inference with the model
        if len(input_names) == 3:
            # Use the input names from the model
            feed_dict = {
                input_names[0]: tf.constant(batch[0], dtype=tf.float32),
                input_names[1]: tf.constant(batch[1], dtype=tf.float32),
                input_names[2]: tf.constant(batch[2], dtype=tf.float32)
            }
            result = model(**feed_dict)
        else:
            # Fallback to using standard input names
            result = model(
                input_ap=tf.constant(batch[0], dtype=tf.float32),
                input_lat=tf.constant(batch[1], dtype=tf.float32),
                input_ob=tf.constant(batch[2], dtype=tf.float32)
            )
        
        # Extract the prediction from the result dictionary
        output_key = list(result.keys())[0]
        preds = result[output_key].numpy()
        prob = float(preds[0][0])
        
        # Clean up to free memory
        del batch, imgs, result, preds
        
        return {"probability": prob}
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
