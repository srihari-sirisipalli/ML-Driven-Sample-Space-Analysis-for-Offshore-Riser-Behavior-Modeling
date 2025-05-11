import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

# GUI imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QTabWidget,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QComboBox,
    QSplitter, QTextEdit, QMessageBox, QProgressBar,
    QSizePolicy, QListWidget, QAbstractItemView,
    QCheckBox, QGroupBox, QRadioButton, QSpinBox,
    QDoubleSpinBox, QFormLayout, QGridLayout, QFrame,
    QListWidgetItem, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

# Matplotlib setup
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
def report_error_metrics(true_values, predicted_values):
    """
    Report basic error metrics between true and predicted/calculated values.
    """
    import numpy as np

    # Remove NaNs
    mask = ~np.isnan(true_values) & ~np.isnan(predicted_values)
    true_values = np.array(true_values)[mask]
    predicted_values = np.array(predicted_values)[mask]
    
    if len(true_values) == 0:
        return {"Error": "No valid data to compare"}

    mae = np.mean(np.abs(true_values - predicted_values))
    rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
    max_error = np.max(np.abs(true_values - predicted_values))
    r2 = 1 - np.sum((true_values - predicted_values) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)

    metrics = {
        "MAE (Mean Absolute Error)": mae,
        "RMSE (Root Mean Squared Error)": rmse,
        "Max Error": max_error,
        "R² (Coefficient of Determination)": r2
    }
    
    return metrics

class DataLoader(QThread):
    """Worker thread to handle data loading without freezing GUI"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    error_raised = pyqtSignal(str)
    loading_finished = pyqtSignal(object, dict)  # Added dict for file metadata

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.df = None
        self.file_metadata = {}

    def run(self):
        try:
            print(f"[DataLoader] Starting file load: {self.file_path}")
            self.status_update.emit("Loading file...")
            self.progress_update.emit(10)

            # Check file extension and initialize metadata
            _, ext = os.path.splitext(self.file_path)
            self.file_metadata.update({
                'extension': ext.lower(),
                'skiprows': 0,
                'header': 0
            })

            print(f"[DataLoader] Detected file extension: {ext.lower()}")

            # Load data based on file type
            try:
                if ext.lower() in ['.xlsx', '.xls']:
                    print("[DataLoader] Attempting to load Excel file")
                    self._load_excel_file()
                elif ext.lower() == '.csv':
                    print("[DataLoader] Attempting to load CSV file")
                    self._load_csv_file()
                else:
                    raise ValueError(f"Unsupported file format: {ext}")
            except Exception as e:
                print(f"[DataLoader] ERROR during file loading: {e}")
                self.error_raised.emit(f"Error processing file: {str(e)}")
                return

            print(f"[DataLoader] File loaded successfully, starting processing...")

            # Process data
            self._process_data()

            # Emit result
            self.loading_finished.emit(self.df, self.file_metadata)
            print(f"[DataLoader] File processing completed successfully!")

        except Exception as e:
            print(f"[DataLoader] CRITICAL ERROR: {e}")
            self.error_raised.emit(f"Error processing file: {str(e)}")

    def _load_excel_file(self):
        """Handle Excel file loading"""
        try:
            df_test = pd.read_excel(self.file_path)
            if 'Unnamed' in str(df_test.columns[0]) or df_test.shape[1] < 5:
                print("[DataLoader] Header detection: Skipping 2 rows for Excel")
                self.status_update.emit("Detected header rows, skipping first 2 rows...")
                self.file_metadata['skiprows'] = 2
                self.df = pd.read_excel(self.file_path, skiprows=2)
            else:
                print("[DataLoader] No header issues detected")
                self.df = df_test
        except Exception as e:
            print(f"[DataLoader] ERROR loading Excel file: {e}")
            raise

    def _load_csv_file(self):
        """Handle CSV file loading"""
        try:
            df_test = pd.read_csv(self.file_path)
            if 'Unnamed' in str(df_test.columns[0]) or df_test.shape[1] < 5:
                print("[DataLoader] Header detection: Possible header rows, adjusting for CSV")
                df_test = pd.read_csv(self.file_path, skiprows=1)
                if 'Unnamed' in str(df_test.columns[0]):
                    self.file_metadata['skiprows'] = 2
                    self.df = pd.read_csv(self.file_path, skiprows=2)
                else:
                    self.file_metadata['skiprows'] = 1
                    self.df = df_test
            else:
                print("[DataLoader] CSV structure looks fine")
                self.df = df_test
        except Exception as e:
            print(f"[DataLoader] ERROR loading CSV file: {e}")
            raise

    def _process_data(self):
        """Process loaded data"""
        try:
            self.progress_update.emit(40)
            self.status_update.emit("Cleaning data...")
            print(f"[DataLoader] Starting data cleaning...")

            # Handle unnamed columns
            unnamed_cols = [col for col in self.df.columns if 'Unnamed' in str(col)]
            if unnamed_cols and unnamed_cols[0] == self.df.columns[0]:
                print(f"[DataLoader] Unnamed first column detected, dropping it")
                self.file_metadata.update({
                    'skip_column': True,
                    'first_column_name': self.df.columns[0]
                })
                self.df = self.df.iloc[:, 1:]
            else:
                self.file_metadata['skip_column'] = False
                print(f"[DataLoader] No unnamed first column detected")

            # Convert numeric columns
            print(f"[DataLoader] Converting columns to numeric where possible...")
            for col in self.df.columns:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except Exception as e:
                    print(f"[DataLoader] WARNING: Could not convert column {col} to numeric: {e}")

            self.progress_update.emit(100)
            self.status_update.emit("Processing complete!")
            print(f"[DataLoader] Data cleaning complete, ready to emit results.")

        except Exception as e:
            print(f"[DataLoader] ERROR during data processing: {e}")
            raise


class MatplotlibCanvas(FigureCanvas):
    """Canvas for matplotlib plots"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class FeatureTransformationApp(QMainWindow):
    """Main application window for Feature Transformation Tool"""
    def __init__(self):
        super().__init__()
        print("[App] Initializing main window...")
        self.setWindowTitle("Feature Transformation Tool")
        self.setGeometry(100, 100, 1280, 800)
        
        # Data storage
        self.df = None
        self.file_path = None
        self.file_metadata = {}
        
        # Initialize UI elements
        self.init_ui()
        print("[App] UI initialized successfully.")

    def init_ui(self):
        """Initialize user interface"""
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create toolbar with buttons
        toolbar_layout = QHBoxLayout()
        
        # File selection button
        self.load_btn = QPushButton("Load Data File")
        self.load_btn.clicked.connect(self.load_data)
        toolbar_layout.addWidget(self.load_btn)
        
        # File info label
        self.file_label = QLabel("No file loaded")
        toolbar_layout.addWidget(self.file_label)
        
        toolbar_layout.addStretch()
        
        # Save results button
        self.save_btn = QPushButton("Save Transformed Data")
        self.save_btn.clicked.connect(self.save_data)
        self.save_btn.setEnabled(False)
        toolbar_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(toolbar_layout)
        
        # Progress bar for loading
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_data_preview_tab()
        self.create_coord_transform_tab()
        self.create_math_transform_tab()
        self.create_column_operations_tab()
        
        self.create_save_options_tab()
        self.create_visualization_tab()
        
        # Disable tabs initially
        self.tabs.setEnabled(False)
        
        # Status bar for messages
        self.statusBar().showMessage("Ready")

    def create_data_preview_tab(self):
        """Create tab for data preview"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Data info
        self.data_info_label = QLabel("No data loaded")
        controls_layout.addWidget(self.data_info_label)
        
        # File metadata info
        self.metadata_label = QLabel("")
        controls_layout.addWidget(self.metadata_label)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Data table
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)
        
        self.tabs.addTab(tab, "Data Preview")
    
    def create_coord_transform_tab(self):
        """Create tab for coordinate transformations"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Transformation section
        transform_group = QGroupBox("Coordinate Transformation")
        transform_layout = QVBoxLayout()
        transform_group.setLayout(transform_layout)
        
        # Select transformation type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Transformation Type:"))
        
        self.transform_type = QComboBox()
        self.transform_type.addItems([
            "Cartesian to Polar (X,Y → Angle,Radius)",
            "Polar to Cartesian (Angle,Radius → X,Y)"
        ])
        self.transform_type.currentIndexChanged.connect(self.update_transform_ui)
        type_layout.addWidget(self.transform_type)
        transform_layout.addLayout(type_layout)
        
        # Column selection section
        columns_group = QGroupBox("Column Selection")
        columns_layout = QFormLayout()
        
        # For X,Y → Angle,Radius
        self.x_column = QComboBox()
        self.y_column = QComboBox()
        columns_layout.addRow("X Column:", self.x_column)
        columns_layout.addRow("Y Column:", self.y_column)
        
        # For Angle,Radius → X,Y
        self.angle_column = QComboBox()
        self.radius_column = QComboBox()
        columns_layout.addRow("Angle Column:", self.angle_column)
        columns_layout.addRow("Radius Column:", self.radius_column)
        
        # Angle units selection
        self.angle_units = QComboBox()
        self.angle_units.addItems(["Degrees", "Radians"])
        columns_layout.addRow("Angle Units:", self.angle_units)
        
        # Output column names
        self.output_angle_name = QTextEdit("Angle")
        self.output_angle_name.setMaximumHeight(30)
        self.output_radius_name = QTextEdit("Radius")
        self.output_radius_name.setMaximumHeight(30)
        self.output_x_name = QTextEdit("X_calculated")
        self.output_x_name.setMaximumHeight(30)
        self.output_y_name = QTextEdit("Y_calculated")
        self.output_y_name.setMaximumHeight(30)
        
        columns_layout.addRow("Output Angle Column Name:", self.output_angle_name)
        columns_layout.addRow("Output Radius Column Name:", self.output_radius_name)
        columns_layout.addRow("Output X Column Name:", self.output_x_name)
        columns_layout.addRow("Output Y Column Name:", self.output_y_name)
        
        columns_group.setLayout(columns_layout)
        transform_layout.addWidget(columns_group)
        
        # Options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        # Calculate error option
        self.calc_error = QCheckBox("Calculate Error (for validation with original coordinates)")
        self.calc_error.setChecked(True)
        options_layout.addWidget(self.calc_error)
        
        # Add original columns option
        self.keep_original = QCheckBox("Keep Original Columns")
        self.keep_original.setChecked(True)
        options_layout.addWidget(self.keep_original)
        
        options_group.setLayout(options_layout)
        transform_layout.addWidget(options_group)
        
        # Transform button
        self.transform_btn = QPushButton("Transform Coordinates")
        self.transform_btn.clicked.connect(self.transform_coordinates)
        transform_layout.addWidget(self.transform_btn)
        
        layout.addWidget(transform_group)
        
        # Results preview
        results_group = QGroupBox("Results Preview")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.tabs.addTab(tab, "Coordinate Transformation")
        
        # Initially configure the UI for the selected transformation type
        self.update_transform_ui()
        
    def create_math_transform_tab(self):
        """Create tab for mathematical transformations"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Transform section
        transform_group = QGroupBox("Mathematical Transformation")
        transform_layout = QVBoxLayout()
        transform_group.setLayout(transform_layout)
        
        # Column selection and transformation type
        form_layout = QFormLayout()
        
        # Select column to transform
        self.math_column = QComboBox()
        form_layout.addRow("Select Column:", self.math_column)
        
        # Transformation type
        self.math_transform_type = QComboBox()
        self.math_transform_type.addItems([
            "Logarithm (Base 10)",
            "Natural Logarithm",
            "Square Root",
            "Square",
            "Cube",
            "Reciprocal",
            "Standardize (Z-score)",
            "Min-Max Scale (0-1)",
            "Absolute Value",
            "Signum (-1, 0, +1)",
            "Sine (sin)",
            "Cosine (cos)"
        ])
        form_layout.addRow("Transformation:", self.math_transform_type)
        
        # Output column name
        self.math_output_name = QTextEdit("")
        self.math_output_name.setMaximumHeight(30)
        self.math_output_name.setPlaceholderText("Leave empty for auto-naming")
        form_layout.addRow("Output Column Name:", self.math_output_name)
        
        transform_layout.addLayout(form_layout)
        
        # Options
        options_layout = QVBoxLayout()
        
        # Handle invalid values checkbox
        self.handle_invalid = QCheckBox("Handle Invalid Values (e.g., log of negative numbers)")
        self.handle_invalid.setChecked(True)
        options_layout.addWidget(self.handle_invalid)
        
        # Keep original column
        self.math_keep_original = QCheckBox("Keep Original Column")
        self.math_keep_original.setChecked(True)
        options_layout.addWidget(self.math_keep_original)
        
        transform_layout.addLayout(options_layout)
        
        # Transform button
        self.math_transform_btn = QPushButton("Apply Transformation")
        self.math_transform_btn.clicked.connect(self.apply_math_transformation)
        transform_layout.addWidget(self.math_transform_btn)
        
        layout.addWidget(transform_group)
        
        # Stats comparison
        stats_group = QGroupBox("Before/After Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Courier New", 10))
        stats_layout.addWidget(self.stats_text)
        
        # Visualization
        self.math_canvas = MatplotlibCanvas(self, width=5, height=3)
        self.math_toolbar = NavigationToolbar(self.math_canvas, self)
        stats_layout.addWidget(self.math_toolbar)
        stats_layout.addWidget(self.math_canvas)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        self.tabs.addTab(tab, "Mathematical Transformation")
        
        # Connect events
        self.math_transform_type.currentIndexChanged.connect(self.update_math_output_name)
        self.math_column.currentTextChanged.connect(self.update_math_output_name)
        
    def create_column_operations_tab(self):
        """Create tab for column operations (create new columns from existing ones)"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Operations section
        ops_group = QGroupBox("Column Operations")
        ops_layout = QVBoxLayout()
        ops_group.setLayout(ops_layout)
        
        # Operation type
        op_type_layout = QHBoxLayout()
        op_type_layout.addWidget(QLabel("Operation Type:"))
        
        self.op_type = QComboBox()
        self.op_type.addItems([
            "Basic Arithmetic",
            "Polynomial",
            "Trigonometric",
            "Custom Expression"
        ])
        self.op_type.currentIndexChanged.connect(self.update_operation_ui)
        op_type_layout.addWidget(self.op_type)
        
        ops_layout.addLayout(op_type_layout)
        
        # Column selection section
        columns_group = QGroupBox("Column Selection")
        columns_layout = QFormLayout()
        
        # First and second column selection
        self.op_col1 = QComboBox()
        self.op_col2 = QComboBox()
        columns_layout.addRow("First Column:", self.op_col1)
        columns_layout.addRow("Second Column:", self.op_col2)
        
        # Arithmetic operation selection
        self.arithmetic_op = QComboBox()
        self.arithmetic_op.addItems(["+", "-", "*", "/", "^", "min", "max"])
        columns_layout.addRow("Arithmetic Operation:", self.arithmetic_op)
        
        # Polynomial options
        self.poly_degree = QSpinBox()
        self.poly_degree.setRange(2, 5)
        self.poly_degree.setValue(2)
        columns_layout.addRow("Polynomial Degree:", self.poly_degree)
        
        # Trigonometric function
        self.trig_function = QComboBox()
        self.trig_function.addItems(["sin", "cos", "tan", "arcsin", "arccos", "arctan"])
        columns_layout.addRow("Trigonometric Function:", self.trig_function)
        
        # Custom expression
        self.custom_expr = QTextEdit()
        self.custom_expr.setMaximumHeight(60)
        self.custom_expr.setPlaceholderText("Example: col1 * np.sin(col2) + 5")
        columns_layout.addRow("Custom Expression:", self.custom_expr)
        
        # Output column name
        self.op_output_name = QTextEdit("")
        self.op_output_name.setMaximumHeight(30)
        self.op_output_name.setPlaceholderText("Leave empty for auto-naming")
        columns_layout.addRow("Output Column Name:", self.op_output_name)
        
        columns_group.setLayout(columns_layout)
        ops_layout.addWidget(columns_group)
        
        # Operation button
        self.apply_op_btn = QPushButton("Apply Operation")
        self.apply_op_btn.clicked.connect(self.apply_column_operation)
        ops_layout.addWidget(self.apply_op_btn)
        
        layout.addWidget(ops_group)
        
        # Results preview
        results_group = QGroupBox("Operation Results")
        results_layout = QVBoxLayout()
        
        self.op_results_table = QTableWidget()
        # Error Metrics Panel
        self.op_metrics_text = QTextEdit()
        self.op_metrics_text.setReadOnly(True)
        self.op_metrics_text.setFont(QFont("Courier New", 10))
        results_layout.addWidget(self.op_metrics_text)

        results_layout.addWidget(self.op_results_table)
        
        # Results plot
        self.op_canvas = MatplotlibCanvas(self, width=5, height=3)
        self.op_toolbar = NavigationToolbar(self.op_canvas, self)
        results_layout.addWidget(self.op_toolbar)
        results_layout.addWidget(self.op_canvas)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.tabs.addTab(tab, "Column Operations")
        
        # Initially configure the UI for the selected operation type
        self.update_operation_ui()
        
    def create_save_options_tab(self):
        """Create tab for save options"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Save options section
        save_group = QGroupBox("Save Options")
        save_layout = QVBoxLayout()
        save_group.setLayout(save_layout)
        
        # Column selection
        selection_layout = QHBoxLayout()
        
        # Available columns
        available_group = QGroupBox("Available Columns")
        available_layout = QVBoxLayout()
        self.available_columns = QListWidget()
        self.available_columns.setSelectionMode(QAbstractItemView.ExtendedSelection)
        available_layout.addWidget(self.available_columns)
        available_group.setLayout(available_layout)
        
        # Column transfer buttons
        transfer_layout = QVBoxLayout()
        self.add_col_btn = QPushButton(">")
        self.add_col_btn.clicked.connect(self.add_selected_columns)
        self.remove_col_btn = QPushButton("<")
        self.remove_col_btn.clicked.connect(self.remove_selected_columns)
        self.add_all_btn = QPushButton(">>")
        self.add_all_btn.clicked.connect(self.add_all_columns)
        self.remove_all_btn = QPushButton("<<")
        self.remove_all_btn.clicked.connect(self.remove_all_columns)
        
        transfer_layout.addStretch()
        transfer_layout.addWidget(self.add_col_btn)
        transfer_layout.addWidget(self.remove_col_btn)
        transfer_layout.addWidget(self.add_all_btn)
        transfer_layout.addWidget(self.remove_all_btn)
        transfer_layout.addStretch()
        
        # Selected columns
        selected_group = QGroupBox("Columns to Save")
        selected_layout = QVBoxLayout()
        self.selected_columns = QListWidget()
        self.selected_columns.setSelectionMode(QAbstractItemView.ExtendedSelection)
        selected_layout.addWidget(self.selected_columns)
        selected_group.setLayout(selected_layout)
        
        selection_layout.addWidget(available_group)
        selection_layout.addLayout(transfer_layout)
        selection_layout.addWidget(selected_group)
        
        save_layout.addLayout(selection_layout)
        
        # File format options
        format_group = QGroupBox("Output Format")
        format_layout = QHBoxLayout()
        
        self.save_format = QComboBox()
        self.save_format.addItems(["CSV", "Excel"])
        format_layout.addWidget(QLabel("File Format:"))
        format_layout.addWidget(self.save_format)
        
        format_group.setLayout(format_layout)
        save_layout.addWidget(format_group)
        
        # Save button
        self.select_save_btn = QPushButton("Select Columns and Save")
        self.select_save_btn.clicked.connect(self.save_selected_columns)
        save_layout.addWidget(self.select_save_btn)
        
        layout.addWidget(save_group)
        
        # Columns preview
        preview_group = QGroupBox("Selected Columns Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_table = QTableWidget()
        preview_layout.addWidget(self.preview_table)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        self.tabs.addTab(tab, "Save Options")
        
    def create_visualization_tab(self):
        """Create tab for visualizations"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Visualization options
        vis_options = QGroupBox("Visualization Options")
        vis_layout = QVBoxLayout()
        vis_options.setLayout(vis_layout)
        
        # Plot type and column selection
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems([
            "Scatter Plot",
            "Line Plot",
            "Histogram",
            "Box Plot",
            "Violin Plot",
            "Heatmap (Correlation)",
            "Pair Plot"
        ])
        self.plot_type.currentIndexChanged.connect(self.update_visualization_controls)
        controls_layout.addWidget(self.plot_type)
        
        # X and Y column selection
        controls_layout.addWidget(QLabel("X Column:"))
        self.vis_x_column = QComboBox()
        controls_layout.addWidget(self.vis_x_column)
        
        controls_layout.addWidget(QLabel("Y Column:"))
        self.vis_y_column = QComboBox()
        controls_layout.addWidget(self.vis_y_column)
        
        # Color by column
        controls_layout.addWidget(QLabel("Color By:"))
        self.vis_color_column = QComboBox()
        self.vis_color_column.addItem("None")
        controls_layout.addWidget(self.vis_color_column)
        
        vis_layout.addLayout(controls_layout)
        
        # Additional options
        options_layout = QHBoxLayout()
        
        # Add grid
        self.vis_grid = QCheckBox("Show Grid")
        self.vis_grid.setChecked(True)
        options_layout.addWidget(self.vis_grid)
        
        # Add trendline
        self.vis_trend = QCheckBox("Show Trendline")
        self.vis_trend.setChecked(True)
        options_layout.addWidget(self.vis_trend)
        
        # Add histogram bins for histogram plot
        options_layout.addWidget(QLabel("Bins:"))
        self.vis_bins = QSpinBox()
        self.vis_bins.setRange(5, 100)
        self.vis_bins.setValue(20)
        options_layout.addWidget(self.vis_bins)
        
        options_layout.addStretch()
        
        # Update and save buttons
        self.update_vis_btn = QPushButton("Update Plot")
        self.update_vis_btn.clicked.connect(self.update_visualization)
        options_layout.addWidget(self.update_vis_btn)
        
        self.save_vis_btn = QPushButton("Save Plot")
        self.save_vis_btn.clicked.connect(self.save_visualization)
        options_layout.addWidget(self.save_vis_btn)
        
        vis_layout.addLayout(options_layout)
        
        layout.addWidget(vis_options)
        
        # Plot canvas
        self.vis_canvas = MatplotlibCanvas(self, width=6, height=4)
        self.vis_toolbar = NavigationToolbar(self.vis_canvas, self)
        layout.addWidget(self.vis_toolbar)
        layout.addWidget(self.vis_canvas)
        
        self.tabs.addTab(tab, "Visualization")
        
        # Initialize controls visibility
        self.update_visualization_controls()

    def load_data(self):
        """Open file dialog and load selected data file"""
        print("[App] User triggered data load...")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", 
            "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if file_path:
            self.file_path = file_path
            print(f"[App] Selected file: {file_path}")
            self.file_label.setText(f"Loading: {os.path.basename(file_path)}...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            self.tabs.setEnabled(False)
            self.load_btn.setEnabled(False)

            # Start data loading thread
            self.data_loader = DataLoader(file_path)
            self.data_loader.progress_update.connect(self.update_progress)
            self.data_loader.status_update.connect(self.update_status)
            self.data_loader.error_raised.connect(self.show_error)
            self.data_loader.loading_finished.connect(self.data_loaded)
            self.data_loader.start()
        else:
            print("[App] No file selected for loading.")
            
    def update_progress(self, value):
        """Update progress bar value"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """Update status bar with message"""
        self.statusBar().showMessage(message)
    
    def show_error(self, message):
        """Display error message"""
        print(f"[App] ERROR during data load: {message}")
        self.progress_bar.setVisible(False)
        self.file_label.setText("Error loading file")
        self.load_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", message)
    
    def data_loaded(self, df, metadata):
        """Called when data is successfully loaded"""
        print("[App] Data load completed, updating UI...")
        self.df = df
        self.file_metadata = metadata
        self.file_label.setText(f"Loaded: {os.path.basename(self.file_path)}")
        self.progress_bar.setVisible(False)
        
        # Update data info
        rows, cols = self.df.shape
        print(f"[App] Loaded DataFrame: {rows} rows, {cols} columns")
        self.data_info_label.setText(f"Rows: {rows}, Columns: {cols}")
        
        metadata_text = f"File format: {metadata.get('extension', 'unknown')}, "
        metadata_text += f"Skiprows: {metadata.get('skiprows', 0)}, "
        metadata_text += f"Skip first column: {metadata.get('skip_column', False)}"
        self.metadata_label.setText(metadata_text)
        
        # Enable tabs
        self.tabs.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        self.update_data_preview()
        self.update_column_selectors()
        self.update_save_columns_lists()
        self.statusBar().showMessage(f"Data loaded: {rows} rows, {cols} columns")
    
    def update_data_preview(self):
        """Update the data preview table with loaded data"""
        if self.df is None:
            return
        
        # Setup table dimensions
        rows, cols = self.df.shape
        preview_rows = min(rows, 100)  # Limit to 100 rows for performance
        
        self.data_table.setRowCount(preview_rows)
        self.data_table.setColumnCount(cols)
        
        # Set headers
        self.data_table.setHorizontalHeaderLabels(self.df.columns)
        
        # Populate table cells
        for i in range(preview_rows):
            for j in range(cols):
                value = str(self.df.iloc[i, j])
                item = QTableWidgetItem(value)
                self.data_table.setItem(i, j, item)
        
        # Resize columns to content
        self.data_table.resizeColumnsToContents()
    
    def update_column_selectors(self):
        """Update all column selectors with current DataFrame columns"""
        if self.df is None:
            return
        
        print("[App] Updating column selectors with current columns...")
        columns = self.df.columns.tolist()
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        # Update coordinate transformation selectors
        self.x_column.clear()
        self.y_column.clear()
        self.angle_column.clear()
        self.radius_column.clear()
        
        if numeric_cols:
            self.x_column.addItems(numeric_cols)
            self.y_column.addItems(numeric_cols)
            self.angle_column.addItems(numeric_cols)
            self.radius_column.addItems(numeric_cols)
            
            # Set default second selections if possible
            if len(numeric_cols) > 1:
                self.y_column.setCurrentIndex(1)
                self.radius_column.setCurrentIndex(1)
        
        # Update math transformation selector
        self.math_column.clear()
        if numeric_cols:
            self.math_column.addItems(numeric_cols)
        
        # Update column operations selectors
        self.op_col1.clear()
        self.op_col2.clear()
        if numeric_cols:
            self.op_col1.addItems(numeric_cols)
            self.op_col2.addItems(numeric_cols)
            
            # Set default second selection if possible
            if len(numeric_cols) > 1:
                self.op_col2.setCurrentIndex(1)
        
        # Update visualization selectors
        self.vis_x_column.clear()
        self.vis_y_column.clear()
        self.vis_color_column.clear()
        
        self.vis_color_column.addItem("None")  # Always include None option for color
        
        if numeric_cols:
            self.vis_x_column.addItems(numeric_cols)
            self.vis_y_column.addItems(numeric_cols)
            self.vis_color_column.addItems(numeric_cols)
            
            # Set default second selection if possible
            if len(numeric_cols) > 1:
                self.vis_y_column.setCurrentIndex(1)
        
        print(f"[App] Column selectors updated with {len(columns)} columns")
        
        # Update math output name after updating column selection
        self.update_math_output_name()
        
    def update_save_columns_lists(self):
        """Update the available and selected columns in the save options tab"""
        if self.df is None:
            return
        
        # Clear existing items
        self.available_columns.clear()
        self.selected_columns.clear()
        
        # Add all columns to available list
        for col in self.df.columns:
            self.available_columns.addItem(col)
            
    def update_transform_ui(self):
        """Update coordinate transformation UI based on selected transformation type"""
        transform_type = self.transform_type.currentIndex()
        print(f"[Transform] UI updated for transform type: {transform_type}")
        
        if transform_type == 0:  # X,Y → Angle,Radius
            # Show X,Y input fields, hide Angle,Radius
            self.x_column.setVisible(True)
            self.y_column.setVisible(True)
            self.angle_column.setVisible(False)
            self.radius_column.setVisible(False)
            
            # Find label widgets and update them
            labels = self.findChildren(QLabel)
            for label in labels:
                if label.text() == "X Column:":
                    label.setVisible(True)
                elif label.text() == "Y Column:":
                    label.setVisible(True)
                elif label.text() == "Angle Column:":
                    label.setVisible(False)
                elif label.text() == "Radius Column:":
                    label.setVisible(False)
            
        else:  # Angle,Radius → X,Y
            # Show Angle,Radius input fields, hide X,Y
            self.x_column.setVisible(False)
            self.y_column.setVisible(False)
            self.angle_column.setVisible(True)
            self.radius_column.setVisible(True)
            
            # Find label widgets and update them
            labels = self.findChildren(QLabel)
            for label in labels:
                if label.text() == "X Column:":
                    label.setVisible(False)
                elif label.text() == "Y Column:":
                    label.setVisible(False)
                elif label.text() == "Angle Column:":
                    label.setVisible(True)
                elif label.text() == "Radius Column:":
                    label.setVisible(True)
            
    def update_math_output_name(self):
        """Update the suggested output name for math transformations"""
        if self.math_column.currentText() == "":
            return
            
        col_name = self.math_column.currentText()
        transform_type = self.math_transform_type.currentText()
        
        # Generate suffix based on transform type
        suffix = ""
        if "Logarithm (Base 10)" in transform_type:
            suffix = "_log10"
        elif "Natural Logarithm" in transform_type:
            suffix = "_ln"
        elif "Square Root" in transform_type:
            suffix = "_sqrt"
        elif "Square" in transform_type:
            suffix = "_sq"
        elif "Cube" in transform_type:
            suffix = "_cube"
        elif "Reciprocal" in transform_type:
            suffix = "_recip"
        elif "Standardize" in transform_type:
            suffix = "_zscore"
        elif "Min-Max" in transform_type:
            suffix = "_norm"
        elif "Absolute" in transform_type:
            suffix = "_abs"
        elif "Signum" in transform_type:
            suffix = "_sign"
        elif "Sine" in transform_type:
            suffix = "_sin"
        elif "Cosine" in transform_type:
            suffix = "_cos"
   
        suggested_name = f"{col_name}{suffix}"
        self.math_output_name.setText(suggested_name)
        
    def update_operation_ui(self):
        """Update column operations UI based on selected operation type"""
        op_type = self.op_type.currentIndex()
        print(f"[Operations] UI updated for operation type: {op_type}")
        
        # Hide all specific controls first
        self.arithmetic_op.setVisible(False)
        self.poly_degree.setVisible(False)
        self.trig_function.setVisible(False)
        self.custom_expr.setVisible(False)
        
        # Find label widgets and update them
        form_labels = self.findChildren(QLabel)
        for label in form_labels:
            if label.text() == "Arithmetic Operation:":
                label.setVisible(False)
            elif label.text() == "Polynomial Degree:":
                label.setVisible(False)
            elif label.text() == "Trigonometric Function:":
                label.setVisible(False)
            elif label.text() == "Custom Expression:":
                label.setVisible(False)
        
        # Show only relevant controls
        if op_type == 0:  # Basic Arithmetic
            self.arithmetic_op.setVisible(True)
            for label in form_labels:
                if label.text() == "Arithmetic Operation:":
                    label.setVisible(True)
                    
            # Update output name suggestion
            self.update_operation_output_name()
            
        elif op_type == 1:  # Polynomial
            self.poly_degree.setVisible(True)
            for label in form_labels:
                if label.text() == "Polynomial Degree:":
                    label.setVisible(True)
                    
            # Update output name suggestion
            self.update_operation_output_name()
            
        elif op_type == 2:  # Trigonometric
            self.trig_function.setVisible(True)
            for label in form_labels:
                if label.text() == "Trigonometric Function:":
                    label.setVisible(True)
                    
            # Update output name suggestion
            self.update_operation_output_name()
            
        elif op_type == 3:  # Custom Expression
            self.custom_expr.setVisible(True)
            for label in form_labels:
                if label.text() == "Custom Expression:":
                    label.setVisible(True)
                    
            # Clear output name for custom expressions
            self.op_output_name.clear()
            
    def update_operation_output_name(self):
        """Update the suggested output name for column operations"""
        if self.op_col1.currentText() == "" or self.op_col2.currentText() == "":
            return
            
        col1 = self.op_col1.currentText()
        col2 = self.op_col2.currentText()
        op_type = self.op_type.currentIndex()
        
        # Generate name based on operation type
        if op_type == 0:  # Basic Arithmetic
            op_symbol = self.arithmetic_op.currentText()
            if op_symbol == "+":
                suggested_name = f"{col1}_plus_{col2}"
            elif op_symbol == "-":
                suggested_name = f"{col1}_minus_{col2}"
            elif op_symbol == "*":
                suggested_name = f"{col1}_times_{col2}"
            elif op_symbol == "/":
                suggested_name = f"{col1}_div_{col2}"
            elif op_symbol == "^":
                suggested_name = f"{col1}_pow_{col2}"
            elif op_symbol == "min":
                suggested_name = f"min_{col1}_{col2}"
            elif op_symbol == "max":
                suggested_name = f"max_{col1}_{col2}"
            else:
                suggested_name = f"{col1}_{op_symbol}_{col2}"
                
        elif op_type == 1:  # Polynomial
            degree = self.poly_degree.value()
            suggested_name = f"{col1}_poly{degree}"
            
        elif op_type == 2:  # Trigonometric
            trig_func = self.trig_function.currentText()
            suggested_name = f"{trig_func}_{col1}"
            
        else:  # Custom Expression
            suggested_name = "custom_expr_result"
            
        self.op_output_name.setText(suggested_name)
        
    def update_visualization_controls(self):
        """Update visualization controls based on selected plot type"""
        plot_type = self.plot_type.currentIndex()
        print(f"[Visualization] UI updated for plot type: {plot_type}")
        
        # Show/hide X and Y column selectors
        self.vis_x_column.setVisible(True)
        self.vis_y_column.setVisible(True)
        self.vis_color_column.setVisible(True)
        self.vis_bins.setVisible(False)
        
        for label in self.findChildren(QLabel):
            if label.text() == "X Column:":
                label.setVisible(True)
            elif label.text() == "Y Column:":
                label.setVisible(True)
            elif label.text() == "Color By:":
                label.setVisible(True)
            elif label.text() == "Bins:":
                label.setVisible(False)
        
        # Customize based on plot type
        if plot_type == 0:  # Scatter Plot
            pass  # Default settings are fine
            
        elif plot_type == 1:  # Line Plot
            pass  # Default settings are fine
            
        elif plot_type == 2:  # Histogram
            self.vis_y_column.setVisible(False)
            self.vis_bins.setVisible(True)
            
            for label in self.findChildren(QLabel):
                if label.text() == "Y Column:":
                    label.setVisible(False)
                elif label.text() == "Bins:":
                    label.setVisible(True)
                    
        elif plot_type == 3:  # Box Plot
            pass  # Default settings are fine
            
        elif plot_type == 4:  # Violin Plot
            pass  # Default settings are fine
            
        elif plot_type == 5:  # Heatmap (Correlation)
            self.vis_x_column.setVisible(False)
            self.vis_y_column.setVisible(False)
            self.vis_color_column.setVisible(False)
            
            for label in self.findChildren(QLabel):
                if label.text() == "X Column:":
                    label.setVisible(False)
                elif label.text() == "Y Column:":
                    label.setVisible(False)
                elif label.text() == "Color By:":
                    label.setVisible(False)
                    
        elif plot_type == 6:  # Pair Plot
            self.vis_x_column.setVisible(False)
            self.vis_y_column.setVisible(False)
            
            for label in self.findChildren(QLabel):
                if label.text() == "X Column:":
                    label.setVisible(False)
                elif label.text() == "Y Column:":
                    label.setVisible(False)
    
    def transform_coordinates(self):
        """Perform coordinate transformation based on user selections"""
        print("[Transform] User triggered coordinate transformation...")
        if self.df is None:
            print("[Transform] ERROR: No data loaded to transform")
            QMessageBox.warning(self, "Error", "No data loaded")
            return

        transform_type = self.transform_type.currentIndex()

        try:
            if transform_type == 0:  # Cartesian (X,Y → Angle, Radius)
                print("[Transform] Mode: Cartesian (X,Y) ➔ Polar (Angle, Radius)")
                x_col = self.x_column.currentText()
                y_col = self.y_column.currentText()

                if not x_col or not y_col:
                    print("[Transform] ERROR: X or Y column not selected")
                    QMessageBox.warning(self, "Error", "Please select X and Y columns")
                    return

                use_degrees = self.angle_units.currentText() == "Degrees"
                print(f"[Transform] Angle units: {'Degrees' if use_degrees else 'Radians'}")

                angle_col_name = self.output_angle_name.toPlainText() or "Angle"
                radius_col_name = self.output_radius_name.toPlainText() or "Radius"

                radius = np.sqrt(self.df[x_col] ** 2 + self.df[y_col] ** 2)
                angle = np.arctan2(self.df[y_col], self.df[x_col])
                if use_degrees:
                    angle = np.degrees(angle)

                print("[Transform] Calculated Angle and Radius")

                result_df = self.df.copy()
                result_df[angle_col_name] = angle
                result_df[radius_col_name] = radius

                if self.calc_error.isChecked():
                    print("[Transform] Calculating transformation errors...")
                    rad_angle = np.radians(angle) if use_degrees else angle
                    x_calculated = radius * np.cos(rad_angle)
                    y_calculated = radius * np.sin(rad_angle)

                    result_df[self.output_x_name.toPlainText() or "X_calculated"] = x_calculated
                    result_df[self.output_y_name.toPlainText() or "Y_calculated"] = y_calculated

                    x_error = self.df[x_col] - x_calculated
                    y_error = self.df[y_col] - y_calculated

                    result_df["X_error"] = x_error
                    result_df["Y_error"] = y_error
                    result_df["Total_error"] = np.sqrt(x_error**2 + y_error**2)
                    print("[Transform] Errors calculated and added to result")

                if not self.keep_original.isChecked():
                    print("[Transform] Removing original columns")
                    result_df = result_df.drop([x_col, y_col], axis=1)

            else:  # Polar (Angle, Radius → X,Y)
                print("[Transform] Mode: Polar (Angle, Radius) ➔ Cartesian (X,Y)")
                angle_col = self.angle_column.currentText()
                radius_col = self.radius_column.currentText()

                if not angle_col or not radius_col:
                    print("[Transform] ERROR: Angle or Radius column not selected")
                    QMessageBox.warning(self, "Error", "Please select Angle and Radius columns")
                    return

                is_degrees = self.angle_units.currentText() == "Degrees"
                print(f"[Transform] Angle units: {'Degrees' if is_degrees else 'Radians'}")

                x_col_name = self.output_x_name.toPlainText() or "X_calculated"
                y_col_name = self.output_y_name.toPlainText() or "Y_calculated"

                angles_rad = np.radians(self.df[angle_col]) if is_degrees else self.df[angle_col]
                x_calculated = self.df[radius_col] * np.cos(angles_rad)
                y_calculated = self.df[radius_col] * np.sin(angles_rad)

                result_df = self.df.copy()
                result_df[x_col_name] = x_calculated
                result_df[y_col_name] = y_calculated

                if self.calc_error.isChecked():
                    print("[Transform] Calculating transformation errors...")
                    radius_calculated = np.sqrt(x_calculated**2 + y_calculated**2)
                    angle_calculated = np.arctan2(y_calculated, x_calculated)

                    if is_degrees:
                        angle_calculated = np.degrees(angle_calculated)

                    angle_diff = (self.df[angle_col] - angle_calculated) % (360 if is_degrees else 2*np.pi)
                    angle_error = np.minimum(angle_diff, (360 if is_degrees else 2*np.pi) - angle_diff)

                    radius_error = self.df[radius_col] - radius_calculated

                    result_df["Radius_error"] = radius_error
                    result_df["Angle_error"] = angle_error
                    result_df["Total_error"] = np.sqrt(radius_error**2 + (radius_calculated * angle_error)**2)
                    print("[Transform] Errors calculated and added to result")

                if not self.keep_original.isChecked():
                    print("[Transform] Removing original columns")
                    result_df = result_df.drop([angle_col, radius_col], axis=1)

            self.df = result_df
            self.update_data_preview()
            self.update_column_selectors()
            self.update_save_columns_lists()
            self.update_results_preview(result_df)
            
            # Focus on the results table
            self.tabs.setCurrentIndex(1)  # Coordinate transformation tab

            self.statusBar().showMessage("Coordinate transformation completed successfully!")
            print("[Transform] Transformation complete ✅")
            QMessageBox.information(self, "Success", "Coordinate transformation completed successfully!")

        except Exception as e:
            print(f"[Transform] ERROR during transformation: {e}")
            QMessageBox.critical(self, "Error", f"Error during transformation: {str(e)}")
            self.statusBar().showMessage("Error during coordinate transformation ❌")
    
    def update_results_preview(self, df):
        """Update the results preview table with transformed data"""
        if df is None:
            return
            
        rows, cols = df.shape
        preview_rows = min(rows, 100)  # Limit to 100 rows for performance
        
        self.results_table.setRowCount(preview_rows)
        self.results_table.setColumnCount(cols)
        
        # Set headers
        self.results_table.setHorizontalHeaderLabels(df.columns)
        
        # Populate table cells
        for i in range(preview_rows):
            for j in range(cols):
                value = str(df.iloc[i, j])
                item = QTableWidgetItem(value)
                self.results_table.setItem(i, j, item)
        
        # Resize columns to content
        self.results_table.resizeColumnsToContents()
    
    def apply_math_transformation(self):
        """Apply mathematical transformation to selected column"""
        print("[Math] User triggered mathematical transformation...")
        if self.df is None:
            print("[Math] ERROR: No data loaded to transform")
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        column = self.math_column.currentText()
        if not column:
            print("[Math] ERROR: No column selected")
            QMessageBox.warning(self, "Error", "Please select a column to transform")
            return
        
        transform_type = self.math_transform_type.currentText()
        output_name = self.math_output_name.toPlainText()
        
        # If output name is empty, use the auto-generated name
        if not output_name:
            self.update_math_output_name()
            output_name = self.math_output_name.toPlainText()
        
        # Check if output column already exists
        if output_name in self.df.columns:
            response = QMessageBox.question(
                self, "Column Exists", 
                f"Column '{output_name}' already exists. Do you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if response == QMessageBox.No:
                return
        
        try:
            # Make a copy of the source column to avoid data loss
            source_data = self.df[column].copy()
            
            # Store original statistics for comparison
            orig_stats = source_data.describe()
            
            # Apply transformation
            if "Logarithm (Base 10)" in transform_type:
                print(f"[Math] Applying log10 to column: {column}")
                if self.handle_invalid.isChecked():
                    # Handle negative values (add a constant to make all values positive)
                    min_val = source_data.min()
                    if min_val <= 0:
                        shift = abs(min_val) + 1
                        print(f"[Math] Shifting values by +{shift} to handle non-positive values")
                        transformed = np.log10(source_data + shift)
                    else:
                        transformed = np.log10(source_data)
                else:
                    transformed = np.log10(source_data)
                    
            elif "Natural Logarithm" in transform_type:
                print(f"[Math] Applying natural log to column: {column}")
                if self.handle_invalid.isChecked():
                    # Handle negative values
                    min_val = source_data.min()
                    if min_val <= 0:
                        shift = abs(min_val) + 1
                        print(f"[Math] Shifting values by +{shift} to handle non-positive values")
                        transformed = np.log(source_data + shift)
                    else:
                        transformed = np.log(source_data)
                else:
                    transformed = np.log(source_data)
                    
            elif "Square Root" in transform_type:
                print(f"[Math] Applying square root to column: {column}")
                if self.handle_invalid.isChecked():
                    # Handle negative values
                    transformed = np.sqrt(np.abs(source_data)) * np.sign(source_data)
                else:
                    transformed = np.sqrt(source_data)
                    
            elif "Square" in transform_type:
                print(f"[Math] Squaring column: {column}")
                transformed = source_data ** 2
                
            elif "Cube" in transform_type:
                print(f"[Math] Cubing column: {column}")
                transformed = source_data ** 3
                
            elif "Reciprocal" in transform_type:
                print(f"[Math] Taking reciprocal of column: {column}")
                if self.handle_invalid.isChecked():
                    # Avoid division by zero
                    transformed = 1 / (source_data + (source_data == 0) * 1e-10)
                else:
                    transformed = 1 / source_data
                    
            elif "Standardize" in transform_type:
                print(f"[Math] Standardizing column: {column}")
                transformed = (source_data - source_data.mean()) / source_data.std()
                
            elif "Min-Max Scale" in transform_type:
                print(f"[Math] Min-Max scaling column: {column}")
                min_val = source_data.min()
                max_val = source_data.max()
                if min_val == max_val:
                    transformed = np.zeros_like(source_data)
                else:
                    transformed = (source_data - min_val) / (max_val - min_val)
                    
            elif "Absolute Value" in transform_type:
                print(f"[Math] Taking absolute value of column: {column}")
                transformed = np.abs(source_data)
                
            elif "Signum" in transform_type:
                print(f"[Math] Applying signum function to column: {column}")
                transformed = np.sign(source_data)
            elif "Sine (sin)" in transform_type:
                print(f"[Math] Applying sine to column: {column}")
                transformed = np.sin(source_data)

            elif "Cosine (cos)" in transform_type:
                print(f"[Math] Applying cosine to column: {column}")
                transformed = np.cos(source_data)
 
            else:
                print(f"[Math] ERROR: Unknown transformation type: {transform_type}")
                QMessageBox.warning(self, "Error", f"Unknown transformation type: {transform_type}")
                return
            
            # Add transformed column to DataFrame
            result_df = self.df.copy()
            result_df[output_name] = transformed
            
            # Drop original column if requested
            if not self.math_keep_original.isChecked():
                print(f"[Math] Removing original column: {column}")
                if column != output_name:  # Only drop if different from output
                    result_df = result_df.drop(column, axis=1)
            
            # Update DataFrame
            self.df = result_df
            self.update_data_preview()
            self.update_column_selectors()
            self.update_save_columns_lists()
            
            # Update stats and visualization
            self.display_math_stats_comparison(source_data, transformed, orig_stats, transformed.describe())
            
            self.statusBar().showMessage(f"Mathematical transformation applied: {transform_type}")
            print(f"[Math] Transformation {transform_type} applied successfully ✅")
            QMessageBox.information(self, "Success", f"Transformation applied successfully: {transform_type}")
            
        except Exception as e:
            print(f"[Math] ERROR during transformation: {e}")
            QMessageBox.critical(self, "Error", f"Error during transformation: {str(e)}")
            self.statusBar().showMessage("Error during mathematical transformation ❌")
    
    def display_math_stats_comparison(self, original, transformed, orig_stats, trans_stats):
        """Display statistics comparison between original and transformed data"""
        # Update text statistics
        self.stats_text.clear()
        self.stats_text.append("=== Original vs. Transformed Statistics ===\n")
        
        stats_headers = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        
        self.stats_text.append("Statistic      Original       Transformed")
        self.stats_text.append("-----------------------------------------")
        
        for stat in stats_headers:
            self.stats_text.append(f"{stat:12} {orig_stats[stat]:13.6g} {trans_stats[stat]:13.6g}")
        
        # Visualize distributions
        self.math_canvas.fig.clear()
        
        # Create two subplots for original and transformed data
        ax1 = self.math_canvas.fig.add_subplot(121)
        ax2 = self.math_canvas.fig.add_subplot(122)
        
        # Plot histograms
        sns.histplot(original.dropna(), ax=ax1, kde=True)
        ax1.set_title("Original Distribution")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Frequency")
        
        sns.histplot(transformed.dropna(), ax=ax2, kde=True)
        ax2.set_title("Transformed Distribution")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Frequency")
        
        self.math_canvas.fig.tight_layout()
        self.math_canvas.draw()
    
    def apply_column_operation(self):
        """Apply selected operation to create a new column"""
        print("[Operations] User triggered column operation...")
        if self.df is None:
            print("[Operations] ERROR: No data loaded")
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        op_type = self.op_type.currentIndex()
        
        # Get output column name
        output_name = self.op_output_name.toPlainText()
        if not output_name:
            self.update_operation_output_name()
            output_name = self.op_output_name.toPlainText()
        
        # Check if output column already exists
        if output_name in self.df.columns:
            response = QMessageBox.question(
                self, "Column Exists", 
                f"Column '{output_name}' already exists. Do you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if response == QMessageBox.No:
                return
        
        try:
            result_df = self.df.copy()
            
            if op_type == 0:  # Basic Arithmetic
                col1 = self.op_col1.currentText()
                col2 = self.op_col2.currentText()
                operation = self.arithmetic_op.currentText()
                
                if not col1 or not col2:
                    print("[Operations] ERROR: Columns not selected")
                    QMessageBox.warning(self, "Error", "Please select both columns")
                    return
                
                print(f"[Operations] Applying {operation} between {col1} and {col2}")
                
                # Perform operation
                if operation == "+":
                    result = self.df[col1] + self.df[col2]
                elif operation == "-":
                    result = self.df[col1] - self.df[col2]
                elif operation == "*":
                    result = self.df[col1] * self.df[col2]
                elif operation == "/":
                    # Handle division by zero
                    if (self.df[col2] == 0).any():
                        print("[Operations] WARNING: Division by zero detected")
                        result = self.df[col1] / (self.df[col2] + (self.df[col2] == 0) * 1e-10)
                    else:
                        result = self.df[col1] / self.df[col2]
                elif operation == "^":
                    result = self.df[col1] ** self.df[col2]
                elif operation == "min":
                    result = np.minimum(self.df[col1], self.df[col2])
                elif operation == "max":
                    result = np.maximum(self.df[col1], self.df[col2])
                else:
                    print(f"[Operations] ERROR: Unknown operation: {operation}")
                    QMessageBox.warning(self, "Error", f"Unknown operation: {operation}")
                    return
                
                result_df[output_name] = result
                
            elif op_type == 1:  # Polynomial
                col1 = self.op_col1.currentText()
                degree = self.poly_degree.value()
                
                if not col1:
                    print("[Operations] ERROR: Column not selected")
                    QMessageBox.warning(self, "Error", "Please select a column")
                    return
                
                print(f"[Operations] Creating polynomial of degree {degree} for {col1}")
                
                # Create polynomial
                result = self.df[col1] ** degree
                result_df[output_name] = result
                
            elif op_type == 2:  # Trigonometric
                col1 = self.op_col1.currentText()
                trig_func = self.trig_function.currentText()
                
                if not col1:
                    print("[Operations] ERROR: Column not selected")
                    QMessageBox.warning(self, "Error", "Please select a column")
                    return
                
                print(f"[Operations] Applying {trig_func} to {col1}")
                
                # Apply trigonometric function
                if trig_func == "sin":
                    result = np.sin(self.df[col1])
                elif trig_func == "cos":
                    result = np.cos(self.df[col1])
                elif trig_func == "tan":
                    result = np.tan(self.df[col1])
                elif trig_func == "arcsin":
                    # Handle domain issues (-1 to 1)
                    data = np.clip(self.df[col1], -1, 1)
                    result = np.arcsin(data)
                elif trig_func == "arccos":
                    # Handle domain issues (-1 to 1)
                    data = np.clip(self.df[col1], -1, 1)
                    result = np.arccos(data)
                elif trig_func == "arctan":
                    result = np.arctan(self.df[col1])
                else:
                    print(f"[Operations] ERROR: Unknown trigonometric function: {trig_func}")
                    QMessageBox.warning(self, "Error", f"Unknown trigonometric function: {trig_func}")
                    return
                
                result_df[output_name] = result
                
            elif op_type == 3:  # Custom Expression
                expr = self.custom_expr.toPlainText()
                
                if not expr:
                    print("[Operations] ERROR: No expression provided")
                    QMessageBox.warning(self, "Error", "Please enter a custom expression")
                    return
                
                print(f"[Operations] Evaluating custom expression: {expr}")
                
                # Replace column references with DataFrame references
                for col in self.df.columns:
                    expr = expr.replace(f"col1", f"self.df['{self.op_col1.currentText()}']")
                    expr = expr.replace(f"col2", f"self.df['{self.op_col2.currentText()}']")
                    expr = expr.replace(col, f"self.df['{col}']")
                
                # Evaluate the expression
                result = eval(expr)
                result_df[output_name] = result
            
            # Update DataFrame
            self.df = result_df
            # After self.df = result_df
            if op_type in [0, 1]:  # Only if operation is simple (Basic Arithmetic or Polynomial)
                try:
                    col1 = self.op_col1.currentText()
                    if col1 in self.df.columns and output_name in self.df.columns:
                        print(f"[Metrics] Reporting error metrics for {col1} vs {output_name}")
                        metrics = report_error_metrics(self.df[col1], self.df[output_name])
                        for key, value in metrics.items():
                            print(f"{key}: {value:.6f}")
                except Exception as e:
                    print(f"[Metrics] Error calculating metrics: {e}")

            self.update_data_preview()
            self.update_column_selectors()
            self.update_save_columns_lists()
            
            # Update the operation results preview
            self.update_operation_results_preview(result, output_name)
            
            self.statusBar().showMessage(f"Column operation completed: {output_name}")
            print(f"[Operations] Column operation applied successfully ✅")
            QMessageBox.information(self, "Success", f"Operation applied successfully: {output_name}")
            
        except Exception as e:
            print(f"[Operations] ERROR during operation: {e}")
            QMessageBox.critical(self, "Error", f"Error during operation: {str(e)}")
            self.statusBar().showMessage("Error during column operation ❌")
    
    def update_operation_results_preview(self, result_series, output_name):
        """Update the operation results preview table and plot"""
        # Update table preview
        preview_rows = min(len(result_series), 100)
        
        self.op_results_table.setRowCount(preview_rows)
        self.op_results_table.setColumnCount(1)
        self.op_results_table.setHorizontalHeaderLabels([output_name])
        
        for i in range(preview_rows):
            value = str(result_series.iloc[i])
            item = QTableWidgetItem(value)
            self.op_results_table.setItem(i, 0, item)
        
        self.op_results_table.resizeColumnsToContents()

        # Update plot
        self.op_canvas.fig.clear()
        ax = self.op_canvas.fig.add_subplot(111)
        
        # Check if the result is numeric
        if pd.api.types.is_numeric_dtype(result_series):
            sns.histplot(result_series.dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {output_name}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            
            # Add mean and median lines
            mean_val = result_series.mean()
            median_val = result_series.median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Non-numeric data - Cannot plot histogram", 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"Column: {output_name}")
            ax.set_xticks([])
            ax.set_yticks([])
        
        self.op_canvas.fig.tight_layout()
        self.op_canvas.draw()
    
    def add_selected_columns(self):
        """Add selected columns from available to selected list"""
        selected_items = self.available_columns.selectedItems()
        
        for item in selected_items:
            column = item.text()
            # Check if it's already in the selected list
            existing_items = self.selected_columns.findItems(column, Qt.MatchExactly)
            if not existing_items:
                self.selected_columns.addItem(column)
                
        # Update preview
        self.update_column_selection_preview()
    
    def remove_selected_columns(self):
        """Remove selected columns from selected list"""
        selected_items = self.selected_columns.selectedItems()
        
        # We need to remove from bottom to top to avoid index issues
        rows = [self.selected_columns.row(item) for item in selected_items]
        for row in sorted(rows, reverse=True):
            self.selected_columns.takeItem(row)
            
        # Update preview
        self.update_column_selection_preview()
    
    def add_all_columns(self):
        """Add all available columns to selected list"""
        self.selected_columns.clear()
        
        for i in range(self.available_columns.count()):
            column = self.available_columns.item(i).text()
            self.selected_columns.addItem(column)
            
        # Update preview
        self.update_column_selection_preview()
    
    def remove_all_columns(self):
        """Remove all columns from selected list"""
        self.selected_columns.clear()
            
        # Update preview
        self.update_column_selection_preview()
    
    def update_column_selection_preview(self):
        """Update the preview table for selected columns"""
        if self.df is None:
            return
        
        # Get list of selected columns
        selected = []
        for i in range(self.selected_columns.count()):
            selected.append(self.selected_columns.item(i).text())
        
        if not selected:
            # Clear the preview if no columns are selected
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            return
        
        # Create preview with selected columns
        preview_df = self.df[selected]
        
        # Update table
        rows, cols = preview_df.shape
        preview_rows = min(rows, 100)
        
        self.preview_table.setRowCount(preview_rows)
        self.preview_table.setColumnCount(cols)
        
        # Set headers
        self.preview_table.setHorizontalHeaderLabels(preview_df.columns)
        
        # Populate table cells
        for i in range(preview_rows):
            for j in range(cols):
                value = str(preview_df.iloc[i, j])
                item = QTableWidgetItem(value)
                self.preview_table.setItem(i, j, item)
        
        # Resize columns to content
        self.preview_table.resizeColumnsToContents()
    
    def save_selected_columns(self):
        """Save only the selected columns to a file"""
        if self.df is None:
            print("[Save] ERROR: No data loaded to save")
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        # Get list of selected columns
        selected = []
        for i in range(self.selected_columns.count()):
            selected.append(self.selected_columns.item(i).text())
        
        if not selected:
            print("[Save] ERROR: No columns selected to save")
            QMessageBox.warning(self, "Error", "Please select at least one column to save")
            return
        
        try:
            # Create a DataFrame with only selected columns
            save_df = self.df[selected]
            
            # Get file format
            file_format = self.save_format.currentText()
            
            # Get save path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Selected Columns", "", 
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
            )
            
            if not file_path:
                print("[Save] Save operation cancelled")
                return
            
            # Ensure proper extension
            _, ext = os.path.splitext(file_path)
            if not ext:
                if file_format == "CSV":
                    file_path += ".csv"
                else:  # Excel
                    file_path += ".xlsx"
            
            # Save file based on format
            if file_format == "CSV" or (ext.lower() == ".csv"):
                print(f"[Save] Saving CSV file to: {file_path}")
                save_df.to_csv(file_path, index=False)
            else:  # Excel
                print(f"[Save] Saving Excel file to: {file_path}")
                save_df.to_excel(file_path, index=False)
            
            self.statusBar().showMessage(f"File saved successfully: {file_path}")
            print(f"[Save] File saved successfully ✅ ({len(selected)} columns)")
            QMessageBox.information(self, "Save Successful", f"File saved with {len(selected)} columns to:\n{file_path}")
            
        except Exception as e:
            print(f"[Save] ERROR during save: {e}")
            QMessageBox.critical(self, "Error", f"Error saving file: {str(e)}")
            self.statusBar().showMessage("Error during save ❌")
    
    def save_data(self):
        """Save the entire DataFrame to a file"""
        if self.df is None:
            print("[Save] ERROR: No data loaded to save")
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        try:
            # Get save path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Data", "", 
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
            )
            
            if not file_path:
                print("[Save] Save operation cancelled")
                return
            
            # Ensure proper extension
            _, ext = os.path.splitext(file_path)
            if not ext:
                file_path += ".csv"
                ext = ".csv"
            
            # Save file based on extension
            if ext.lower() == ".csv":
                print(f"[Save] Saving CSV file to: {file_path}")
                self.df.to_csv(file_path, index=False)
            elif ext.lower() in [".xlsx", ".xls"]:
                print(f"[Save] Saving Excel file to: {file_path}")
                self.df.to_excel(file_path, index=False)
            else:
                print(f"[Save] Unknown extension, saving as CSV to: {file_path}")
                self.df.to_csv(file_path, index=False)
            
            self.statusBar().showMessage(f"File saved successfully: {file_path}")
            print(f"[Save] Full dataset saved successfully ✅")
            QMessageBox.information(self, "Save Successful", f"File saved successfully to:\n{file_path}")
            
        except Exception as e:
            print(f"[Save] ERROR during save: {e}")
            QMessageBox.critical(self, "Error", f"Error saving file: {str(e)}")
            self.statusBar().showMessage("Error during save ❌")
    
    def update_visualization(self):
        """Update the visualization based on selected options"""
        print("[Visualization] User triggered visualization update...")
        if self.df is None:
            print("[Visualization] ERROR: No data loaded to visualize")
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        plot_type = self.plot_type.currentIndex()
        
        try:
            # Clear previous plot
            self.vis_canvas.fig.clear()
            
            # Different plot types require different approaches
            if plot_type == 0:  # Scatter Plot
                self._create_scatter_plot()
                
            elif plot_type == 1:  # Line Plot
                self._create_line_plot()
                
            elif plot_type == 2:  # Histogram
                self._create_histogram()
                
            elif plot_type == 3:  # Box Plot
                self._create_box_plot()
                
            elif plot_type == 4:  # Violin Plot
                self._create_violin_plot()
                
            elif plot_type == 5:  # Heatmap (Correlation)
                self._create_heatmap()
                
            elif plot_type == 6:  # Pair Plot
                self._create_pair_plot()
            
            # Update canvas
            self.vis_canvas.fig.tight_layout()
            self.vis_canvas.draw()
            
            self.statusBar().showMessage("Visualization updated successfully!")
            print("[Visualization] Plot updated successfully ✅")
            
        except Exception as e:
            print(f"[Visualization] ERROR during visualization: {e}")
            QMessageBox.critical(self, "Error", f"Error creating visualization: {str(e)}")
            self.statusBar().showMessage("Error during visualization ❌")
    
    def _create_scatter_plot(self):
        """Create a scatter plot"""
        x_col = self.vis_x_column.currentText()
        y_col = self.vis_y_column.currentText()
        color_col = self.vis_color_column.currentText()
        
        if not x_col or not y_col:
            print("[Visualization] ERROR: X or Y column not selected")
            QMessageBox.warning(self, "Error", "Please select X and Y columns")
            return
        
        print(f"[Visualization] Creating scatter plot: {y_col} vs {x_col}")
        
        # Create subplot
        ax = self.vis_canvas.fig.add_subplot(111)
        
        # Filter out NaN values
        valid_data = self.df[[x_col, y_col]].dropna()
        
        # Create scatter plot
        if color_col and color_col != "None":
            valid_data = self.df[[x_col, y_col, color_col]].dropna()
            scatter = ax.scatter(valid_data[x_col], valid_data[y_col], 
                               c=valid_data[color_col], cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label=color_col)
        else:
            scatter = ax.scatter(valid_data[x_col], valid_data[y_col], alpha=0.7)
        
        # Add trendline if requested
        if self.vis_trend.isChecked():
            # Calculate trend line
            z = np.polyfit(valid_data[x_col], valid_data[y_col], 1)
            p = np.poly1d(z)
            
            # Get x range for plotting trendline
            x_range = np.linspace(valid_data[x_col].min(), valid_data[x_col].max(), 100)
            
            # Plot trendline
            ax.plot(x_range, p(x_range), 'r--', label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
            
            # Calculate R² for trend
            r_squared = np.corrcoef(valid_data[x_col], valid_data[y_col])[0, 1]**2
            ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox={'boxstyle': 'round', 'alpha': 0.5})
        
        # Add grid if requested
        if self.vis_grid.isChecked():
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        
        # Add legend if applicable
        if self.vis_trend.isChecked():
            ax.legend()
    
    def _create_line_plot(self):
        """Create a line plot"""
        x_col = self.vis_x_column.currentText()
        y_col = self.vis_y_column.currentText()
        color_col = self.vis_color_column.currentText()
        
        if not x_col or not y_col:
            print("[Visualization] ERROR: X or Y column not selected")
            QMessageBox.warning(self, "Error", "Please select X and Y columns")
            return
        
        print(f"[Visualization] Creating line plot: {y_col} vs {x_col}")
        
        # Create subplot
        ax = self.vis_canvas.fig.add_subplot(111)
        
        # Filter out NaN values
        valid_data = self.df[[x_col, y_col]].dropna()
        
        # Sort data by x values for line plot
        valid_data = valid_data.sort_values(by=x_col)
        
        # Create line plot
        if color_col and color_col != "None":
            valid_data = self.df[[x_col, y_col, color_col]].dropna()
            valid_data = valid_data.sort_values(by=x_col)
            
            # We need to group by the color column for multiple lines
            for color_value, group in valid_data.groupby(color_col):
                group = group.sort_values(by=x_col)
                ax.plot(group[x_col], group[y_col], label=f"{color_col}={color_value}")
        else:
            ax.plot(valid_data[x_col], valid_data[y_col])
        
        # Add grid if requested
        if self.vis_grid.isChecked():
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        
        # Add legend if applicable
        if color_col and color_col != "None":
            ax.legend()
    
    def _create_histogram(self):
        """Create a histogram"""
        x_col = self.vis_x_column.currentText()
        color_col = self.vis_color_column.currentText()
        
        if not x_col:
            print("[Visualization] ERROR: X column not selected")
            QMessageBox.warning(self, "Error", "Please select a column")
            return
        
        bins = self.vis_bins.value()
        print(f"[Visualization] Creating histogram for: {x_col} with {bins} bins")
        
        # Create subplot
        ax = self.vis_canvas.fig.add_subplot(111)
        
        # Filter out NaN values
        valid_data = self.df[x_col].dropna()
        
        # Create histogram
        if color_col and color_col != "None":
            valid_data = self.df[[x_col, color_col]].dropna()
            for color_value, group in valid_data.groupby(color_col):
                ax.hist(group[x_col], bins=bins, alpha=0.5, label=f"{color_col}={color_value}")
        else:
            hist = ax.hist(valid_data, bins=bins, alpha=0.7)
            
            # Add mean and median lines
            mean_val = valid_data.mean()
            median_val = valid_data.median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
        
        # Add grid if requested
        if self.vis_grid.isChecked():
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of {x_col}")
        
        # Add legend
        ax.legend()
    
    def _create_box_plot(self):
        """Create a box plot"""
        y_col = self.vis_y_column.currentText()
        x_col = self.vis_x_column.currentText()
        
        if not y_col:
            print("[Visualization] ERROR: Y column not selected")
            QMessageBox.warning(self, "Error", "Please select Y column")
            return
        
        print(f"[Visualization] Creating box plot for: {y_col}")
        
        # Create subplot
        ax = self.vis_canvas.fig.add_subplot(111)
        
        # Create box plot
        if x_col and x_col != y_col:
            # Box plot grouped by x_col
            sns.boxplot(x=x_col, y=y_col, data=self.df, ax=ax)
        else:
            # Simple box plot
            sns.boxplot(y=y_col, data=self.df, ax=ax)
        
        # Add grid if requested
        if self.vis_grid.isChecked():
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add title
        if x_col and x_col != y_col:
            ax.set_title(f"Box Plot of {y_col} by {x_col}")
        else:
            ax.set_title(f"Box Plot of {y_col}")
    
    def _create_violin_plot(self):
        """Create a violin plot"""
        y_col = self.vis_y_column.currentText()
        x_col = self.vis_x_column.currentText()
        
        if not y_col:
            print("[Visualization] ERROR: Y column not selected")
            QMessageBox.warning(self, "Error", "Please select Y column")
            return
        
        print(f"[Visualization] Creating violin plot for: {y_col}")
        
        # Create subplot
        ax = self.vis_canvas.fig.add_subplot(111)
        
        # Create violin plot
        if x_col and x_col != y_col:
            # Violin plot grouped by x_col
            sns.violinplot(x=x_col, y=y_col, data=self.df, ax=ax)
        else:
            # Simple violin plot
            sns.violinplot(y=y_col, data=self.df, ax=ax)
        
        # Add grid if requested
        if self.vis_grid.isChecked():
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add title
        if x_col and x_col != y_col:
            ax.set_title(f"Violin Plot of {y_col} by {x_col}")
        else:
            ax.set_title(f"Violin Plot of {y_col}")
    
    def _create_heatmap(self):
        """Create a correlation heatmap"""
        print("[Visualization] Creating correlation heatmap")
        
        # Create subplot
        ax = self.vis_canvas.fig.add_subplot(111)
        
        # Get numeric columns only
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if numeric_df.shape[1] < 2:
            print("[Visualization] ERROR: Not enough numeric columns for correlation")
            QMessageBox.warning(self, "Error", "Need at least 2 numeric columns for correlation")
            return
        
        # Calculate correlation matrix
        corr = numeric_df.corr()
        
        # Create heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, 
                   vmin=-1, vmax=1, center=0, fmt='.2f', ax=ax)
        
        # Add title
        ax.set_title("Correlation Heatmap")
    
    def _create_pair_plot(self):
        """Create a pair plot"""
        color_col = self.vis_color_column.currentText()
        
        # Get numeric columns only
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if numeric_df.shape[1] < 2:
            print("[Visualization] ERROR: Not enough numeric columns for pair plot")
            QMessageBox.warning(self, "Error", "Need at least 2 numeric columns for pair plot")
            return
        
        print(f"[Visualization] Creating pair plot for {numeric_df.shape[1]} numeric columns")
        
        # Limit to 5 columns maximum for performance
        if numeric_df.shape[1] > 5:
            print(f"[Visualization] Limiting pair plot to 5 columns (out of {numeric_df.shape[1]})")
            # Choose columns with highest variance
            variances = numeric_df.var().sort_values(ascending=False)
            selected_cols = variances.index[:5].tolist()
            numeric_df = numeric_df[selected_cols]
        
        # Add color column if needed
        plot_data = numeric_df.copy()
        if color_col and color_col != "None" and color_col not in plot_data.columns:
            plot_data[color_col] = self.df[color_col]
        
        # Create pair plot
        g = sns.pairplot(plot_data, hue=color_col if color_col and color_col != "None" else None,
                        diag_kind='kde')
        
        # Update the current figure to the pairplot figure
        self.vis_canvas.fig = g.fig
    
    def save_visualization(self):
        """Save the current visualization to a file"""
        if self.df is None or self.vis_canvas.fig is None:
            print("[Visualization] ERROR: No visualization to save")
            QMessageBox.warning(self, "Error", "No visualization to save")
            return
        
        try:
            # Get save path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Visualization", "", 
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
            )
            
            if not file_path:
                print("[Visualization] Save operation cancelled")
                return
            
            # Ensure proper extension
            _, ext = os.path.splitext(file_path)
            if not ext:
                file_path += ".png"
            
            # Save figure
            print(f"[Visualization] Saving visualization to: {file_path}")
            self.vis_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            
            self.statusBar().showMessage(f"Visualization saved successfully: {file_path}")
            print(f"[Visualization] Visualization saved successfully ✅")
            QMessageBox.information(self, "Save Successful", f"Visualization saved to:\n{file_path}")
            
        except Exception as e:
            print(f"[Visualization] ERROR during save: {e}")
            QMessageBox.critical(self, "Error", f"Error saving visualization: {str(e)}")
            self.statusBar().showMessage("Error during visualization save ❌")


def main():
    app = QApplication(sys.argv)
    window = FeatureTransformationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()