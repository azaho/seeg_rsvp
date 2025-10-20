import os
from pathlib import Path

def generate_scripts():
    """
    Generate Mac-runnable shell scripts for each template directory.
    Scripts are created in the parent directory of the workspace.
    """
    # Get the current directory (workspace root)
    workspace_root = Path(__file__).parent
    
    # Templates directory
    templates_dir = workspace_root / "templates"
    
    # Parent directory where scripts will be created
    parent_dir = workspace_root.parent
    
    # Check if templates directory exists
    if not templates_dir.exists():
        print(f"Error: Templates directory not found at {templates_dir}")
        return
    
    # Get all subdirectories in templates/
    template_dirs = [d for d in templates_dir.iterdir() if d.is_dir()]
    
    if not template_dirs:
        print("No template directories found.")
        return
    
    print(f"Found {len(template_dirs)} template directory(ies).")
    
    # Generate a script for each template directory
    for template_dir in template_dirs:
        template_name = template_dir.name
        script_name = f"run_{template_name}.command"
        script_path = parent_dir / script_name
        
        # Script content
        script_content = f"""#!/bin/bash
# Auto-generated script to run {template_name}

cd seeg_rsvp/
source .venv/bin/activate
python game.py {template_name}
"""
        
        # Write the script file
        with open(script_path, 'w', newline='\n') as f:
            f.write(script_content)
        
        # Make the script executable (Unix permissions)
        try:
            os.chmod(script_path, 0o755)
            print(f"✓ Created executable: {script_path}")
        except Exception as e:
            print(f"✓ Created: {script_path} (Note: Could not set executable permission: {e})")
    
    print(f"\nAll scripts created in: {parent_dir}")
    print("\nTo run a script on Mac:")
    print("  - Double-click the .command file, or")
    print("  - Run from terminal: ./run_<template_name>.command")

if __name__ == "__main__":
    generate_scripts()

