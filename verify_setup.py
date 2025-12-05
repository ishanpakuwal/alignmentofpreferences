"""
Quick Start Verification Script
CS329H Final Project

Run this script to verify your environment is set up correctly
before running the full training pipeline.

Usage:
    python verify_setup.py
"""

import sys
import importlib

def check_python_version():
    """Check Python version is >= 3.8"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False

def check_torch_cuda():
    """Check PyTorch and CUDA availability"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  CUDA not available (CPU mode will be very slow)")
        return True
    except ImportError:
        print("❌ PyTorch: NOT INSTALLED")
        return False

def check_datasets_access():
    """Check if we can access HuggingFace datasets"""
    try:
        from datasets import load_dataset
        print("Testing dataset access...")
        # Try to load a tiny sample
        dataset = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        next(iter(dataset))  # Just get first example
        print("✅ HuggingFace datasets accessible")
        return True
    except Exception as e:
        print(f"❌ Dataset access failed: {e}")
        return False

def main():
    print("="*70)
    print(" " * 15 + "CS329H ENVIRONMENT VERIFICATION")
    print("="*70)
    print()
    
    all_checks = []
    
    # Check Python version
    all_checks.append(check_python_version())
    print()
    
    # Check core packages
    print("Checking core packages...")
    packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    
    for pkg_name, import_name in packages:
        all_checks.append(check_package(pkg_name, import_name))
    print()
    
    # Check PyTorch/CUDA
    print("Checking PyTorch and CUDA...")
    all_checks.append(check_torch_cuda())
    print()
    
    # Check optional packages
    print("Checking optional packages...")
    optional_packages = [
        ('detoxify', 'detoxify'),
        ('scikit-learn', 'sklearn'),
        ('seaborn', 'seaborn'),
    ]
    
    for pkg_name, import_name in optional_packages:
        check_package(pkg_name, import_name)
    print()
    
    # Check dataset access
    print("Checking dataset access...")
    all_checks.append(check_datasets_access())
    print()
    
    # Summary
    print("="*70)
    if all(all_checks):
        print("✅ All critical checks passed!")
        print("You're ready to run the training pipeline.")
    else:
        print("❌ Some checks failed.")
        print("Please install missing packages using:")
        print("    pip install -r requirements.txt")
    print("="*70)

if __name__ == "__main__":
    main()
