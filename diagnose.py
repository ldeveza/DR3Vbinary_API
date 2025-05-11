"""
Standalone diagnostic script to help troubleshoot model loading issues.
This script checks for model file existence and prints system information.
"""
import os
import sys
import json

def get_system_info():
    """Get basic system information"""
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "env_vars": {k: v for k, v in os.environ.items() if k.startswith(("TF_", "PYTHON", "PATH"))},
    }
    return info

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

if __name__ == "__main__":
    print("=== Model Files Diagnostic ===")
    model_check = check_model_files()
    print(json.dumps(model_check, indent=2))
    
    print("\n=== System Information ===")
    sys_info = get_system_info()
    print(json.dumps(sys_info, indent=2))
    
    print("\n=== Directory Listing ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(script_dir):
        rel_path = os.path.relpath(root, script_dir)
        if rel_path == '.':
            rel_path = ''
        else:
            print(f"DIR: {rel_path}")
        
        for file in files:
            file_path = os.path.join(rel_path, file)
            if not file_path.startswith('.git/'):
                size = os.path.getsize(os.path.join(root, file))
                print(f"  - {file_path} ({size} bytes)")
        
        # Don't go into .git directory
        if '.git' in dirs:
            dirs.remove('.git')
