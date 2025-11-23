#!/usr/bin/env python3
"""
Simple script to run the DVC pipeline stages manually for testing.
This is useful for development before setting up DVC.
"""

import subprocess
import sys
import os

def run_stage(stage_name, script_path):
    """Run a single pipeline stage"""
    print(f"\n{'='*50}")
    print(f"Running {stage_name} stage...")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        if result.returncode != 0:
            print(f"❌ {stage_name} stage failed with return code {result.returncode}")
            return False
        else:
            print(f"✅ {stage_name} stage completed successfully")
            return True
    except Exception as e:
        print(f"❌ Error running {stage_name}: {e}")
        return False

def main():
    """Run all pipeline stages"""
    print("Running DVC Pipeline Stages Manually")
    print("This is for testing before setting up DVC")
    
    stages = [
        ("generate", "project_name/scripts/generate_data.py"),
        ("preprocess", "project_name/scripts/preprocess_data.py"),
        ("train", "project_name/scripts/train_model.py"),
        ("evaluate", "project_name/scripts/evaluate_model.py"),
        ("plots", "project_name/scripts/create_plots.py")
    ]
    
    all_success = True
    
    for stage_name, script_path in stages:
        if not os.path.exists(script_path):
            print(f"❌ Script not found: {script_path}")
            all_success = False
            continue
            
        success = run_stage(stage_name, script_path)
        if not success:
            all_success = False
            print(f"\n⚠️  Pipeline failed at {stage_name} stage")
            break
    
    print(f"\n{'='*50}")
    if all_success:
        print("🎉 All pipeline stages completed successfully!")
        print("\nNext steps:")
        print("1. Install DVC: pip install dvc")
        print("2. Initialize DVC: dvc init")
        print("3. Run with DVC: dvc repro")
    else:
        print("❌ Pipeline execution failed")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()