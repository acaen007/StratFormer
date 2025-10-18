#!/usr/bin/env python3
"""
Setup script to ensure Python path is correctly configured for IDE navigation.
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Python path configured. Project root: {project_root}")
print("Available modules:")
for module in ['core', 'models', 'policies', 'encoders', 'envs']:
    module_path = os.path.join(project_root, module)
    if os.path.exists(module_path):
        print(f"  ✓ {module}")
    else:
        print(f"  ✗ {module}")

# Test imports
try:
    from core.factory import load_env
    print("✓ core.factory imports successfully")
except ImportError as e:
    print(f"✗ core.factory import failed: {e}")

try:
    from models.ridge import RidgeOpponentModel
    print("✓ models.ridge imports successfully")
except ImportError as e:
    print(f"✗ models.ridge import failed: {e}")

try:
    from policies.base import TabularPolicy
    print("✓ policies.base imports successfully")
except ImportError as e:
    print(f"✗ policies.base import failed: {e}")
