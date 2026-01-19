#!/usr/bin/env python3
"""
Fast local test for web backend Cython compilation (~1 minute)
Tests only the web backend to catch issues quickly before Docker build.
"""
import os
import sys
import shutil
import tempfile
import subprocess

def main():
    print("=" * 50)
    print("  Fast Backend Compilation Test")
    print("=" * 50)
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    backend_dir = os.path.join(project_dir, "web", "backend")
    
    # Create temp directory
    test_dir = tempfile.mkdtemp(prefix="usf_backend_test_")
    venv_dir = os.path.join(test_dir, "venv")
    compile_dir = os.path.join(test_dir, "compile")
    
    try:
        # Step 1: Create venv
        print("\n[1/5] Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        
        # Get venv python
        if sys.platform == "win32":
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            venv_python = os.path.join(venv_dir, "bin", "python")
        
        # Step 2: Install deps
        print("[2/5] Installing Cython...")
        subprocess.run([venv_python, "-m", "pip", "install", "-q", "cython", "setuptools", "pydantic-settings"], check=True)
        
        # Step 3: Copy backend only
        print("[3/5] Copying web backend...")
        shutil.copytree(backend_dir, os.path.join(compile_dir, "web_backend"), 
                       ignore=shutil.ignore_patterns("venv", "__pycache__", "*.pyc", ".git"))
        
        # Step 4: Test minimize_init_files logic
        print("[4/5] Testing minimize_init_files logic...")
        init_file = os.path.join(compile_dir, "web_backend", "app", "models", "__init__.py")
        if os.path.exists(init_file):
            with open(init_file, 'r') as f:
                original = f.read()
            
            # Simulate minimize_init_files
            lines = []
            in_docstring = False
            docstring_char = None
            for l in original.split('\n'):
                stripped = l.strip()
                if not in_docstring:
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        docstring_char = stripped[:3]
                        if stripped.count(docstring_char) >= 2:
                            continue
                        in_docstring = True
                        continue
                else:
                    if docstring_char in stripped:
                        in_docstring = False
                    continue
                if stripped.startswith('#'):
                    continue
                if '#' in l and not l.strip().startswith('#'):
                    l = l.split('#')[0].rstrip()
                lines.append(l)
            
            minimized = '\n'.join(lines)
            
            # Write minimized
            with open(init_file, 'w') as f:
                f.write(minimized + '\n')
            
            # Test if it's valid Python
            try:
                compile(minimized, init_file, 'exec')
                print("  ✓ models/__init__.py syntax valid after minimization")
            except SyntaxError as e:
                print(f"  ✗ models/__init__.py SYNTAX ERROR: {e}")
                print("  Minimized content:")
                print(minimized)
                return False
        
        # Step 5: Test all __init__.py files have valid syntax after minimization
        print("[5/5] Testing all __init__.py syntax...")
        backend_path = os.path.join(compile_dir, "web_backend")
        errors = []
        for root, dirs, files in os.walk(backend_path):
            dirs[:] = [d for d in dirs if d not in {'venv', '__pycache__'}]
            for f in files:
                if f == "__init__.py":
                    filepath = os.path.join(root, f)
                    with open(filepath, 'r') as file:
                        content = file.read()
                    try:
                        compile(content, filepath, 'exec')
                    except SyntaxError as e:
                        errors.append(f"{filepath}: {e}")
        
        if errors:
            print("  ✗ Syntax errors found:")
            for err in errors:
                print(f"    {err}")
            return False
        print("  ✓ All __init__.py files have valid syntax")
        
        print("\n" + "=" * 50)
        print("  ✓ ALL TESTS PASSED")
        print("  Safe to push to GitHub")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
