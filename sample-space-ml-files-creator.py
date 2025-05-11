import os
import sys
import logging
import re
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('repo_creation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

repo_structure = """
sample-space-ml/
├── main.py
├── requirements.txt
├── setup.py
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .pre-commit-config.yaml
├── gui_tools/
│   ├── eda/
│   │   ├── main.py
│   │   ├── data_loader.py
│   │   ├── stats_panel.py
│   │   ├── distribution_plot.py
│   │   └── correlation_view.py
│   ├── ml/
│   │   ├── main.py
│   │   ├── model_trainer.py
│   │   ├── feature_selector.py
│   │   ├── error_analysis.py
│   │   └── results_plot.py
│   ├── transformation/
│   │   ├── main.py
│   │   ├── coord_transform_panel.py
│   │   ├── math_transform_panel.py
│   │   ├── column_ops.py
│   │   └── data_preview.py
│   ├── decision_tree/
│   │   ├── main.py
│   │   ├── trainer.py
│   │   ├── tree_visualizer.py
│   │   ├── error_tab.py
│   │   └── node_analysis.py
│   ├── model_comparison/
│   │   ├── main.py
│   │   ├── comparison_runner.py
│   │   ├── result_table.py
│   │   └── metrics_plot.py
│   └── README.md
├── samplespace_ml/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config_manager.py
│   │   ├── settings.py
│   │   └── default_config.yml
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── log_config.py
│   │   └── formatters.py
│   ├── exceptions/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── data.py
│   │   ├── model.py
│   │   └── ui.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── file_io.py
│   │   ├── validation.py
│   │   ├── math_tools.py
│   │   ├── profiler.py
│   │   └── threading.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_cleaner.py
│   │   ├── data_validator.py
│   │   └── data_exporter.py
│   ├── transformations/
│   │   ├── __init__.py
│   │   ├── feature_transformer.py
│   │   ├── coordinate_transformer.py
│   │   ├── column_manager.py
│   │   └── formula_evaluator.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistics.py
│   │   ├── correlation.py
│   │   ├── error_analyzer.py
│   │   └── outlier_detector.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── regression_models.py
│   │   ├── decision_tree.py
│   │   ├── model_comparator.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluator.py
│   │   └── model_serializer.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plot_manager.py
│   │   ├── plot_types/
│   │   │   ├── __init__.py
│   │   │   ├── distribution.py
│   │   │   ├── scatter.py
│   │   │   ├── correlation.py
│   │   │   ├── error.py
│   │   │   └── tree.py
│   │   └── themes/
│   │       ├── __init__.py
│   │       ├── default_theme.py
│   │       └── dark_theme.py
├── docs/
│   ├── user_guide/
│   ├── developer_guide/
│   ├── architecture.md
│   └── api_reference/
├── examples/
│   ├── basic/
│   ├── advanced/
│   └── notebooks/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── test_data/
└── scripts/
    ├── install_dev.sh
    ├── run_tests.sh
    ├── build_docs.sh
    └── lint.sh
"""

def extract_structure(structure_str):
    """
    Parse the tree-like structure string and extract a list of files and directories.
    """
    lines = structure_str.strip().splitlines()
    
    # Extract the root directory name
    root_dir = lines[0].strip('/ ')
    
    paths = []
    
    # Process each line
    for line in lines[1:]:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Calculate indent level
        indent = 0
        for char in line:
            if char in '│├└ ':
                indent += 1
            else:
                break
        
        # Indent level will determine directory depth
        level = indent // 4  # Each level typically has 4 spaces of indentation
        
        # Clean the item name
        item = line.strip('│├└─ \t')
        
        # Determine if it's a directory (ends with '/')
        is_dir = item.endswith('/')
        if is_dir:
            item = item[:-1]  # Remove trailing slash
            
        paths.append((level, item, is_dir))
    
    return root_dir, paths

def create_file_structure(base_dir, structure_str):
    """
    Create the file structure based on the parsed tree.
    """
    root_dir, paths = extract_structure(structure_str)
    root_path = os.path.join(base_dir, root_dir)
    
    # Check if root directory exists
    if os.path.exists(root_path):
        if os.listdir(root_path):
            choice = input(f"Directory '{root_path}' already exists and is not empty. Continue? (y/n): ")
            if choice.lower() != 'y':
                logger.info("Operation cancelled by user")
                return False
    
    # Create root directory
    try:
        os.makedirs(root_path, exist_ok=True)
        logger.info(f"Created root directory: {root_path}")
    except Exception as e:
        logger.error(f"Failed to create root directory {root_path}: {str(e)}")
        return False
    
    # Process paths
    current_path = [root_path]
    current_level = 0
    
    created_dirs = 0
    created_files = 0
    failed_items = []
    
    for level, item, is_dir in paths:
        # Adjust the current path based on the level
        while level < current_level:
            current_path.pop()
            current_level -= 1
            
        if level > current_level:
            # This shouldn't happen with a valid structure
            logger.warning(f"Unexpected indentation level increase from {current_level} to {level} for {item}")
            continue
            
        # Create the full path
        full_path = os.path.join(*current_path, item)
        
        try:
            if is_dir:
                # Create directory
                os.makedirs(full_path, exist_ok=True)
                logger.debug(f"Created directory: {full_path}")
                created_dirs += 1
                
                # Update current path for nested items
                current_path.append(item)
                current_level += 1
            else:
                # Create empty file
                with open(full_path, 'w') as f:
                    pass
                logger.debug(f"Created file: {full_path}")
                created_files += 1
                
                # Make shell scripts executable
                if item.endswith('.sh'):
                    try:
                        os.chmod(full_path, 0o755)  # rwxr-xr-x
                    except Exception as e:
                        logger.warning(f"Failed to make script executable {full_path}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to create {full_path}: {str(e)}")
            failed_items.append((full_path, str(e)))
    
    # Log summary
    logger.info(f"Created {created_dirs} directories and {created_files} files")
    if failed_items:
        logger.warning(f"Failed to create {len(failed_items)} items")
        for path, error in failed_items[:10]:  # Show only first 10 failures to avoid log flooding
            logger.warning(f" - {path}: {error}")
        if len(failed_items) > 10:
            logger.warning(f" - ... and {len(failed_items) - 10} more")
    
    return True

def main():
    """Main function to create the repository structure."""
    print("Creating Sample Space ML Repository Structure...")
    
    # Get base directory
    base_dir = input("Enter base directory (or press Enter for current directory): ").strip()
    if not base_dir:
        base_dir = "."
    
    # Create the structure
    if create_file_structure(base_dir, repo_structure):
        root_path = os.path.join(base_dir, "sample-space-ml")
        print(f"\nRepository created at: {os.path.abspath(root_path)}")
        print("Check repo_creation.log for details.")
    else:
        print("\nFailed to create repository structure. Check repo_creation.log for details.")

if __name__ == "__main__":
    main()