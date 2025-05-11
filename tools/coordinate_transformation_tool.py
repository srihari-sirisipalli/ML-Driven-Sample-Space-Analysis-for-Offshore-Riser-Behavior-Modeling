
    
    
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


class CoordinateTransformApp(QMainWindow):
    """Main application window for Coordinate Transformation Tool"""
    def __init__(self):
        super().__init__()
        print("[App] Initializing main window...")
        self.setWindowTitle("Coordinate Transformation Tool")
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
        self.create_column_management_tab()
        self.create_visualization_tab()
        self.create_error_summary_tab()
        
        # Add a button for total error analysis in the toolbar
        self.total_error_btn = QPushButton("Calculate Total Dataset Error")
        self.total_error_btn.clicked.connect(self.calculate_total_dataset_error)
        toolbar_layout.addWidget(self.total_error_btn)
        
        # Disable tabs initially
        self.tabs.setEnabled(False)
        
        # Status bar for messages
        self.statusBar().showMessage("Ready")
    def calculate_total_dataset_error(self):
        """Calculate and display total error statistics for all error columns in the dataset"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        # Find all columns that might contain error data
        error_cols = [col for col in self.df.columns if 'error' in col.lower()]
        
        if not error_cols:
            QMessageBox.information(self, "Information", 
                                   "No error columns found. Please perform a coordinate transformation first.")
            return
        
        # Create error dictionary for analysis
        error_dict = {}
        for col in error_cols:
            error_dict[col] = self.df[col]
            
        # Add additional columns for total error across dimensions
        
        # Check for X,Y error columns
        if "X_error" in error_cols and "Y_error" in error_cols:
            error_dict["XY_Total"] = np.sqrt(self.df["X_error"]**2 + self.df["Y_error"]**2)
            
        # Check for Radius,Angle error columns
        if "Radius_error" in error_cols and "Angle_error" in error_cols:
            # Need to be careful with angle error - convert to equivalent distance
            # This is an approximation using average radius
            avg_radius = self.df.get("Radius", np.ones(len(self.df))).mean()
            
            # Check if angles are in degrees
            angle_values = self.df.get("Angle_error", None)
            if angle_values is not None and angle_values.max() > 6.28:  # Probably degrees
                # Convert to radians for calculation
                angle_error_rad = np.radians(self.df["Angle_error"])
            else:
                angle_error_rad = self.df["Angle_error"]
                
            # Calculate total polar error (approximation)
            error_dict["Polar_Total"] = np.sqrt(
                self.df["Radius_error"]**2 + (avg_radius * angle_error_rad)**2
            )
        
        # Calculate dataset-wide error metrics
        # Create a dictionary for the summary
        dataset_summary = {}
        
        for col_name, error_values in error_dict.items():
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(error_values):
                continue
                
            # Calculate key metrics for the entire column
            valid_values = error_values.dropna()
            
            # Calculate RMSE (Root Mean Square Error)
            rmse = np.sqrt(np.mean(np.square(valid_values)))
            
            # Calculate MAE (Mean Absolute Error)
            mae = np.mean(np.abs(valid_values))
            
            # Calculate error percentiles
            p_values = np.percentile(np.abs(valid_values), [50, 90, 95, 99])
            
            # Calculate % within different thresholds
            within_001 = 100 * np.mean(np.abs(valid_values) < 0.01)
            within_01 = 100 * np.mean(np.abs(valid_values) < 0.1)
            within_1 = 100 * np.mean(np.abs(valid_values) < 1.0)
            
            # Store metrics
            dataset_summary[col_name] = {
                "RMSE": rmse,
                "MAE": mae,
                "Median Error": p_values[0],
                "90% Error": p_values[1],
                "95% Error": p_values[2], 
                "99% Error": p_values[3],
                "Within 0.01": within_001,
                "Within 0.1": within_01,
                "Within 1.0": within_1
            }
        
        # Display results in a dialog
        self.show_dataset_error_dialog(dataset_summary)
        
        # Also update the error summary tab
        self.display_error_statistics(error_dict)
    def show_dataset_error_dialog(self, dataset_summary):
        """Display a dialog with dataset-wide error metrics"""
        if not dataset_summary:
            return
            
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Total Dataset Error Analysis")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        
        # Add header
        header_label = QLabel("Dataset-Wide Error Analysis")
        header_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(header_label)
        
        # Create table for error metrics
        table = QTableWidget()
        
        # Set up table dimensions
        cols = len(dataset_summary)
        metrics = list(next(iter(dataset_summary.values())).keys())
        rows = len(metrics)
        
        table.setRowCount(rows)
        table.setColumnCount(cols)
        
        # Set headers
        table.setHorizontalHeaderLabels(dataset_summary.keys())
        table.setVerticalHeaderLabels(metrics)
        
        # Populate table cells
        for col_idx, (col_name, metrics_dict) in enumerate(dataset_summary.items()):
            for row_idx, (metric_name, value) in enumerate(metrics_dict.items()):
                if metric_name.startswith("Within"):
                    # Format percentage values
                    cell_text = f"{value:.2f}%"
                else:
                    # Format numeric values
                    cell_text = f"{value:.6g}"
                    
                item = QTableWidgetItem(cell_text)
                table.setItem(row_idx, col_idx, item)
        
        # Resize columns to content
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        layout.addWidget(table)
        
        # Add summary text
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        
        # Add overall insights
        summary_text.append("<h3>Summary Insights:</h3>")
        
        # Find column with best RMSE
        best_rmse_col = min(dataset_summary.items(), key=lambda x: x[1]['RMSE'])[0]
        best_rmse = dataset_summary[best_rmse_col]['RMSE']
        
        # Find column with best MAE
        best_mae_col = min(dataset_summary.items(), key=lambda x: x[1]['MAE'])[0]
        best_mae = dataset_summary[best_mae_col]['MAE']
        
        # Find column with highest within 0.1 percentage
        best_within01_col = max(dataset_summary.items(), key=lambda x: x[1]['Within 0.1'])[0]
        best_within01 = dataset_summary[best_within01_col]['Within 0.1']
        
        summary_text.append(f"<p><b>Best RMSE:</b> {best_rmse_col} ({best_rmse:.6g})</p>")
        summary_text.append(f"<p><b>Best MAE:</b> {best_mae_col} ({best_mae:.6g})</p>")
        summary_text.append(f"<p><b>Best accuracy within 0.1:</b> {best_within01_col} ({best_within01:.2f}%)</p>")
        
        # Add recommendations
        summary_text.append("<h3>Recommendations:</h3>")
        
        # Identify total error columns
        total_error_cols = [col for col in dataset_summary.keys() if "total" in col.lower()]
        
        if total_error_cols:
            best_total_col = min(total_error_cols, key=lambda x: dataset_summary[x]['RMSE'])
            summary_text.append(f"<p>Best overall error metric: <b>{best_total_col}</b> with RMSE = {dataset_summary[best_total_col]['RMSE']:.6g}</p>")
            
            # Add interpretation based on error source
            if "xy" in best_total_col.lower():
                summary_text.append("<p>The Cartesian (X,Y) coordinate representation shows better accuracy overall.</p>")
            elif "polar" in best_total_col.lower():
                summary_text.append("<p>The Polar (Angle, Radius) coordinate representation shows better accuracy overall.</p>")
        
        # Additional insights on component errors
        if "X_error" in dataset_summary and "Y_error" in dataset_summary:
            x_rmse = dataset_summary["X_error"]["RMSE"]
            y_rmse = dataset_summary["Y_error"]["RMSE"]
            if x_rmse < y_rmse:
                summary_text.append(f"<p>X coordinates show better accuracy than Y coordinates (X RMSE = {x_rmse:.6g}, Y RMSE = {y_rmse:.6g}).</p>")
            else:
                summary_text.append(f"<p>Y coordinates show better accuracy than X coordinates (Y RMSE = {y_rmse:.6g}, X RMSE = {x_rmse:.6g}).</p>")
                
        if "Radius_error" in dataset_summary and "Angle_error" in dataset_summary:
            radius_rmse = dataset_summary["Radius_error"]["RMSE"]
            angle_rmse = dataset_summary["Angle_error"]["RMSE"]
            summary_text.append(f"<p>Radius RMSE = {radius_rmse:.6g}, Angle RMSE = {angle_rmse:.6g}</p>")
        
        layout.addWidget(summary_text)
        
        dialog.setLayout(layout)
        dialog.exec_()
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
    
    def create_error_summary_tab(self):
        """Create tab for error summary"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Add controls for selecting error columns
        controls_layout = QHBoxLayout()
        
        # Add multi-select list for error columns
        error_cols_group = QGroupBox("Select Error Columns to Analyze")
        error_cols_layout = QVBoxLayout()
        
        self.error_cols_list = QListWidget()
        self.error_cols_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        error_cols_layout.addWidget(self.error_cols_list)
        
        # Add buttons for column selection
        buttons_layout = QHBoxLayout()
        self.select_all_error_cols_btn = QPushButton("Select All Error Columns")
        self.select_all_error_cols_btn.clicked.connect(self.select_all_error_columns)
        self.analyze_error_cols_btn = QPushButton("Analyze Selected Columns")
        self.analyze_error_cols_btn.clicked.connect(self.analyze_selected_error_columns)
        
        buttons_layout.addWidget(self.select_all_error_cols_btn)
        buttons_layout.addWidget(self.analyze_error_cols_btn)
        
        error_cols_layout.addLayout(buttons_layout)
        error_cols_group.setLayout(error_cols_layout)
        
        controls_layout.addWidget(error_cols_group)
        layout.addLayout(controls_layout)
        
        # Error summary text area
        self.error_summary_text = QTextEdit()
        self.error_summary_text.setReadOnly(True)
        self.error_summary_text.setFont(QFont("Courier New", 10))
        layout.addWidget(QLabel("Error Statistics Summary:"))
        layout.addWidget(self.error_summary_text)
        
        # Error visualization
        self.error_canvas = MatplotlibCanvas(self, width=6, height=4)
        self.error_toolbar = NavigationToolbar(self.error_canvas, self)
        
        layout.addWidget(self.error_toolbar)
        layout.addWidget(self.error_canvas)
        
        # Controls for error visualization
        vis_controls_layout = QHBoxLayout()
        
        self.error_column_selector = QComboBox()
        vis_controls_layout.addWidget(QLabel("Select Error Column:"))
        vis_controls_layout.addWidget(self.error_column_selector)
        
        self.update_error_plot_btn = QPushButton("Update Plot")
        self.update_error_plot_btn.clicked.connect(self.update_error_plot)
        vis_controls_layout.addWidget(self.update_error_plot_btn)
        
        layout.addLayout(vis_controls_layout)
        
        self.tabs.addTab(tab, "Error Summary")

    def display_error_statistics(self, error_dict):
        """Display error statistics in the error summary tab"""
        print(1)
        if not error_dict:
            return
            
        # Clear previous content
        self.error_summary_text.clear()
        self.error_column_selector.clear()
        
        # Update column selector
        self.error_column_selector.addItems(error_dict.keys())
        
        # Calculate and display statistics for each error column
        for col_name, error_values in error_dict.items():
            if not pd.api.types.is_numeric_dtype(error_values):
                continue
                
            valid_values = error_values.dropna()
            
            # Calculate statistics
            rmse = np.sqrt(np.mean(np.square(valid_values)))
            mae = np.mean(np.abs(valid_values))
            std_dev = np.std(valid_values)
            
            # Calculate percentiles
            percentiles = np.percentile(np.abs(valid_values), [50, 90, 95, 99])
            
            # Format statistics
            stats_text = f"\n=== {col_name} Statistics ===\n"
            stats_text += f"RMSE: {rmse:.6g}\n"
            stats_text += f"MAE: {mae:.6g}\n"
            stats_text += f"Standard Deviation: {std_dev:.6g}\n"
            stats_text += f"Median Error: {percentiles[0]:.6g}\n"
            stats_text += f"90th Percentile: {percentiles[1]:.6g}\n"
            stats_text += f"95th Percentile: {percentiles[2]:.6g}\n"
            stats_text += f"99th Percentile: {percentiles[3]:.6g}\n"
            
            # Calculate percentage within thresholds
            within_001 = 100 * np.mean(np.abs(valid_values) < 0.01)
            within_01 = 100 * np.mean(np.abs(valid_values) < 0.1)
            within_1 = 100 * np.mean(np.abs(valid_values) < 1.0)
            
            stats_text += f"\nPercentage of values:\n"
            stats_text += f"Within 0.01: {within_001:.2f}%\n"
            stats_text += f"Within 0.1: {within_01:.2f}%\n"
            stats_text += f"Within 1.0: {within_1:.2f}%\n"
            
            self.error_summary_text.append(stats_text)
        
        # Update the plot for the first column
        if self.error_column_selector.count() > 0:
            self.update_error_plot()
    
    def select_all_error_columns(self):
        """Select all error columns in the list"""
        for i in range(self.error_cols_list.count()):
            item = self.error_cols_list.item(i)
            item.setSelected(True)
    
    def analyze_selected_error_columns(self):
        """Analyze the selected error columns"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        # Get selected error columns
        selected_items = self.error_cols_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "Error", "No error columns selected")
            return
        
        selected_cols = [item.text() for item in selected_items]
        
        # Create error dictionary for analysis
        error_dict = {}
        for col in selected_cols:
            error_dict[col] = self.df[col]
        
        # Display error statistics
        self.display_error_statistics(error_dict)
    
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
    
    def create_column_management_tab(self):
        """Create tab for column management operations"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Split layout horizontally
        h_layout = QHBoxLayout()
        
        # Left side - Column operations
        left_group = QGroupBox("Column Operations")
        left_layout = QVBoxLayout()
        
        # Column selection
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        left_layout.addWidget(QLabel("Select Columns:"))
        left_layout.addWidget(self.column_list)
        
        # Operations buttons
        ops_layout = QHBoxLayout()
        
        self.delete_cols_btn = QPushButton("Delete Selected")
        self.delete_cols_btn.clicked.connect(self.delete_selected_columns)
        ops_layout.addWidget(self.delete_cols_btn)
        
        self.rename_col_btn = QPushButton("Rename Selected")
        self.rename_col_btn.clicked.connect(self.rename_selected_column)
        ops_layout.addWidget(self.rename_col_btn)
        
        left_layout.addLayout(ops_layout)
        
        # Column reordering
        reorder_layout = QHBoxLayout()
        self.move_up_btn = QPushButton("Move Up")
        self.move_up_btn.clicked.connect(self.move_column_up)
        self.move_down_btn = QPushButton("Move Down")
        self.move_down_btn.clicked.connect(self.move_column_down)
        
        reorder_layout.addWidget(self.move_up_btn)
        reorder_layout.addWidget(self.move_down_btn)
        
        left_layout.addLayout(reorder_layout)
        
        left_group.setLayout(left_layout)
        
        # Right side - Column creation
        right_group = QGroupBox("Create New Column")
        right_layout = QVBoxLayout()
        
        # Column formula type
        formula_layout = QFormLayout()
        
        self.formula_type = QComboBox()
        self.formula_type.addItems([
            "Simple Arithmetic",
            "Pythagorean Distance",
            "Angle Between Vectors",
            "Custom Formula"
        ])
        formula_layout.addRow("Formula Type:", self.formula_type)
        
        # Input columns for formula
        self.formula_col1 = QComboBox()
        self.formula_col2 = QComboBox()
        formula_layout.addRow("Column 1:", self.formula_col1)
        formula_layout.addRow("Column 2:", self.formula_col2)
        
        # Operation for simple arithmetic
        self.arithmetic_op = QComboBox()
        self.arithmetic_op.addItems(["+", "-", "*", "/", "^"])
        formula_layout.addRow("Operation:", self.arithmetic_op)
        
        # Custom formula input
        self.custom_formula = QTextEdit()
        self.custom_formula.setPlaceholderText("Enter formula using column names in curly braces, e.g.: {col1} * sin({col2})")
        self.custom_formula.setMaximumHeight(60)
        formula_layout.addRow("Custom Formula:", self.custom_formula)
        
        # New column name
        self.new_col_name = QTextEdit()
        self.new_col_name.setMaximumHeight(30)
        self.new_col_name.setPlaceholderText("New Column Name")
        formula_layout.addRow("New Column Name:", self.new_col_name)
        
        right_layout.addLayout(formula_layout)
        
        # Create column button
        self.create_col_btn = QPushButton("Create Column")
        self.create_col_btn.clicked.connect(self.create_new_column)
        right_layout.addWidget(self.create_col_btn)
        
        right_group.setLayout(right_layout)
        
        # Add left and right groups to horizontal layout
        h_layout.addWidget(left_group, 60)
        h_layout.addWidget(right_group, 40)
        
        layout.addLayout(h_layout)
        
        # Column management preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        self.col_mgmt_table = QTableWidget()
        preview_layout.addWidget(self.col_mgmt_table)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        self.tabs.addTab(tab, "Column Management")
        
        # Connect formula type changed event
        self.formula_type.currentIndexChanged.connect(self.update_formula_ui)
    
    def update_error_plot(self):
        """Update the error plot based on selected column"""
        if self.df is None or self.error_column_selector.currentText() == "":
            return
            
        # Clear the current figure
        self.error_canvas.fig.clear()
        
        # Get selected column
        selected_col = self.error_column_selector.currentText()
        error_data = self.df[selected_col].dropna()
        
        # Create subplot
        ax = self.error_canvas.fig.add_subplot(111)
        
        # Create histogram with KDE
        sns.histplot(data=error_data, kde=True, ax=ax)
        
        # Add labels and title
        ax.set_xlabel("Error Value")
        ax.set_ylabel("Count")
        ax.set_title(f"Error Distribution for {selected_col}")
        
        # Update canvas
        self.error_canvas.draw()

    def create_visualization_tab(self):
        """Create tab for coordinate visualization"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Controls panel
        controls_layout = QHBoxLayout()
        
        # Plot type selection
        plot_type_label = QLabel("Plot Type:")
        self.plot_type = QComboBox()
        self.plot_type.addItems([
            "Scatter Plot (X vs Y)",
            "Polar Plot",
            "Coordinate Comparison",
            "Error Visualization"
        ])
        controls_layout.addWidget(plot_type_label)
        controls_layout.addWidget(self.plot_type)
        
        # Column selection based on plot type
        self.vis_x_column = QComboBox()
        self.vis_y_column = QComboBox()
        self.vis_angle_column = QComboBox()
        self.vis_radius_column = QComboBox()
        self.vis_error_column = QComboBox()
        
        self.vis_x_label = QLabel("X Column:")
        self.vis_y_label = QLabel("Y Column:")
        self.vis_angle_label = QLabel("Angle Column:")
        self.vis_radius_label = QLabel("Radius Column:")
        self.vis_error_label = QLabel("Error Column:")
        
        controls_layout.addWidget(self.vis_x_label)
        controls_layout.addWidget(self.vis_x_column)
        controls_layout.addWidget(self.vis_y_label)
        controls_layout.addWidget(self.vis_y_column)
        controls_layout.addWidget(self.vis_angle_label)
        controls_layout.addWidget(self.vis_angle_column)
        controls_layout.addWidget(self.vis_radius_label)
        controls_layout.addWidget(self.vis_radius_column)
        controls_layout.addWidget(self.vis_error_label)
        controls_layout.addWidget(self.vis_error_column)
        
        # Update button
        self.update_plot_btn = QPushButton("Update Plot")
        self.update_plot_btn.clicked.connect(self.update_visualization)
        controls_layout.addWidget(self.update_plot_btn)
        
        layout.addLayout(controls_layout)
        
        # Plotting area
        self.vis_canvas = MatplotlibCanvas(self, width=8, height=6)
        self.vis_toolbar = NavigationToolbar(self.vis_canvas, self)
        
        layout.addWidget(self.vis_toolbar)
        layout.addWidget(self.vis_canvas)
        
        # Save plot button
        self.save_plot_btn = QPushButton("Save Plot")
        self.save_plot_btn.clicked.connect(self.save_plot)
        layout.addWidget(self.save_plot_btn)
        
        self.tabs.addTab(tab, "Visualization")
        
        # Connect plot type changed event
        self.plot_type.currentIndexChanged.connect(self.update_vis_controls)
    
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
        
        # Also update column management table
        self.col_mgmt_table.setRowCount(preview_rows)
        self.col_mgmt_table.setColumnCount(cols)
        self.col_mgmt_table.setHorizontalHeaderLabels(self.df.columns)
        
        for i in range(preview_rows):
            for j in range(cols):
                value = str(self.df.iloc[i, j])
                item = QTableWidgetItem(value)
                self.col_mgmt_table.setItem(i, j, item)
        
        self.col_mgmt_table.resizeColumnsToContents()
    
    def update_column_selectors(self):
        """Update all combo boxes with column names"""
        if self.df is None:
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = self.df.columns.tolist()
        
        # For coordinate transformation
        self.x_column.clear()
        self.y_column.clear()
        self.angle_column.clear()
        self.radius_column.clear()
        
        if numeric_cols:
            self.x_column.addItems(numeric_cols)
            self.y_column.addItems(numeric_cols)
            self.angle_column.addItems(numeric_cols)
            self.radius_column.addItems(numeric_cols)
            
            # Set default selections if possible
            if len(numeric_cols) > 1:
                self.y_column.setCurrentIndex(1)
                self.radius_column.setCurrentIndex(1)
        
        # For column management
        self.column_list.clear()
        for col in all_cols:
            self.column_list.addItem(col)
        
        self.formula_col1.clear()
        self.formula_col2.clear()
        if numeric_cols:
            self.formula_col1.addItems(numeric_cols)
            self.formula_col2.addItems(numeric_cols)
            
            # Set default selections if possible
            if len(numeric_cols) > 1:
                self.formula_col2.setCurrentIndex(1)
        
        # For visualization
        self.vis_x_column.clear()
        self.vis_y_column.clear()
        self.vis_angle_column.clear()
        self.vis_radius_column.clear()
        self.vis_error_column.clear()
        
        if numeric_cols:
            self.vis_x_column.addItems(numeric_cols)
            self.vis_y_column.addItems(numeric_cols)
            self.vis_angle_column.addItems(numeric_cols)
            self.vis_radius_column.addItems(numeric_cols)
            self.vis_error_column.addItems(numeric_cols)
            
            # Set default selections if possible
            if len(numeric_cols) > 1:
                self.vis_y_column.setCurrentIndex(1)
                self.vis_radius_column.setCurrentIndex(1)
        
        # For error summary
        error_cols = [col for col in numeric_cols if 'error' in col.lower()]
        if error_cols:
            self.error_column_selector.clear()
            self.error_column_selector.addItems(error_cols)
            
            # Update error columns list
            self.error_cols_list.clear()
            for col in error_cols:
                self.error_cols_list.addItem(col)
        
        # Update UI based on selections
        self.update_transform_ui()
        self.update_formula_ui()
        self.update_vis_controls()
    
    def update_transform_ui(self):
        """Update UI elements based on selected transformation type"""
        transform_type = self.transform_type.currentIndex()
        
        if transform_type == 0:  # X,Y → Angle,Radius
            # Show X,Y input fields
            self.x_column.setVisible(True)
            self.y_column.setVisible(True)
            self.angle_column.setVisible(False)
            self.radius_column.setVisible(False)
            
            # Update labels
            self.findChild(QLabel, "", Qt.FindChildrenRecursively).setText("X Column:")
            self.findChild(QLabel, "", Qt.FindChildrenRecursively).setText("Y Column:")
            
        else:  # Angle,Radius → X,Y
            # Show Angle,Radius input fields
            self.x_column.setVisible(False)
            self.y_column.setVisible(False)
            self.angle_column.setVisible(True)
            self.radius_column.setVisible(True)
            
            # Update labels
            self.findChild(QLabel, "", Qt.FindChildrenRecursively).setText("Angle Column:")
            self.findChild(QLabel, "", Qt.FindChildrenRecursively).setText("Radius Column:")
    
    def update_formula_ui(self):
        """Update UI elements based on selected formula type"""
        formula_type = self.formula_type.currentIndex()
        
        # Show/hide relevant controls based on formula type
        if formula_type == 0:  # Simple Arithmetic
            self.formula_col1.setVisible(True)
            self.formula_col2.setVisible(True)
            self.arithmetic_op.setVisible(True)
            self.custom_formula.setVisible(False)
            
        elif formula_type == 1 or formula_type == 2:  # Pythagorean Distance or Angle Between Vectors
            self.formula_col1.setVisible(True)
            self.formula_col2.setVisible(True)
            self.arithmetic_op.setVisible(False)
            self.custom_formula.setVisible(False)
            
        elif formula_type == 3:  # Custom Formula
            self.formula_col1.setVisible(False)
            self.formula_col2.setVisible(False)
            self.arithmetic_op.setVisible(False)
            self.custom_formula.setVisible(True)
    
    def update_vis_controls(self):
        """Update visualization controls based on selected plot type"""
        plot_type = self.plot_type.currentIndex()
        
        # Hide all column selectors first
        self.vis_x_label.setVisible(False)
        self.vis_x_column.setVisible(False)
        self.vis_y_label.setVisible(False)
        self.vis_y_column.setVisible(False)
        self.vis_angle_label.setVisible(False)
        self.vis_angle_column.setVisible(False)
        self.vis_radius_label.setVisible(False)
        self.vis_radius_column.setVisible(False)
        self.vis_error_label.setVisible(False)
        self.vis_error_column.setVisible(False)
        
        if plot_type == 0:  # Scatter Plot (X vs Y)
            self.vis_x_label.setVisible(True)
            self.vis_x_column.setVisible(True)
            self.vis_y_label.setVisible(True)
            self.vis_y_column.setVisible(True)
            
        elif plot_type == 1:  # Polar Plot
            self.vis_angle_label.setVisible(True)
            self.vis_angle_column.setVisible(True)
            self.vis_radius_label.setVisible(True)
            self.vis_radius_column.setVisible(True)
            
        elif plot_type == 2:  # Coordinate Comparison
            self.vis_x_label.setVisible(True)
            self.vis_x_column.setVisible(True)
            self.vis_y_label.setVisible(True)
            self.vis_y_column.setVisible(True)
            self.vis_angle_label.setVisible(True)
            self.vis_angle_column.setVisible(True)
            self.vis_radius_label.setVisible(True)
            self.vis_radius_column.setVisible(True)
            
        elif plot_type == 3:  # Error Visualization
            self.vis_error_label.setVisible(True)
            self.vis_error_column.setVisible(True)
    
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
            self.update_results_preview(result_df)
            self.tabs.setCurrentIndex(1)

            self.statusBar().showMessage("Coordinate transformation completed successfully!")
            print("[Transform] Transformation complete ✅")
            QMessageBox.information(self, "Success", "Coordinate transformation completed successfully!")

        except Exception as e:
            print(f"[Transform] ERROR during transformation: {e}")
            QMessageBox.critical(self, "Error", f"Error during transformation: {str(e)}")
            self.statusBar().showMessage("Error during coordinate transformation ❌")

    
    def update_results_preview(self, df):
        """Update the results preview table with transformed data"""
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
    
    def delete_selected_columns(self):
        """Delete selected columns from the dataframe"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        # Get selected column names
        selected_items = self.column_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "Error", "No columns selected")
            return
        
        selected_cols = [item.text() for item in selected_items]
        
        # Confirm deletion
        response = QMessageBox.question(self, "Confirm Deletion", 
                                       f"Are you sure you want to delete {len(selected_cols)} columns?",
                                       QMessageBox.Yes | QMessageBox.No)
        
        if response == QMessageBox.Yes:
            try:
                # Drop selected columns
                self.df = self.df.drop(selected_cols, axis=1)
                
                # Update displays
                self.update_data_preview()
                self.update_column_selectors()
                
                QMessageBox.information(self, "Success", f"Deleted {len(selected_cols)} columns")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error deleting columns: {str(e)}")
    
    def rename_selected_column(self):
        """Rename selected column"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        # Get selected column names
        selected_items = self.column_list.selectedItems()
        
        if not selected_items or len(selected_items) != 1:
            QMessageBox.warning(self, "Error", "Please select exactly one column to rename")
            return
        
        col_to_rename = selected_items[0].text()
        
        # Get new name from user
        from PyQt5.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(self, "Rename Column", 
                                           f"Enter new name for column '{col_to_rename}':")
        
        if ok and new_name:
            try:
                # Check if new name already exists
                if new_name in self.df.columns:
                    response = QMessageBox.question(self, "Column Exists", 
                                                  f"Column '{new_name}' already exists. Overwrite?",
                                                  QMessageBox.Yes | QMessageBox.No)
                    if response == QMessageBox.No:
                        return
                    
                    # If overwriting, drop the existing column
                    self.df = self.df.drop(new_name, axis=1)
                
                # Rename column
                self.df = self.df.rename(columns={col_to_rename: new_name})
                
                # Update displays
                self.update_data_preview()
                self.update_column_selectors()
                
                QMessageBox.information(self, "Success", f"Renamed column '{col_to_rename}' to '{new_name}'")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error renaming column: {str(e)}")
    
    def move_column_up(self):
        """Move selected column up in the dataframe"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        # Get selected column names
        selected_items = self.column_list.selectedItems()
        
        if not selected_items or len(selected_items) != 1:
            QMessageBox.warning(self, "Error", "Please select exactly one column to move")
            return
        
        col_to_move = selected_items[0].text()
        
        # Get current column order
        col_list = self.df.columns.tolist()
        
        # Find index of the column to move
        idx = col_list.index(col_to_move)
        
        # Can't move further up if already at the top
        if idx == 0:
            QMessageBox.information(self, "Information", "Column is already at the top")
            return
        
        # Swap with the column above
        col_list[idx], col_list[idx-1] = col_list[idx-1], col_list[idx]
        
        # Reorder columns
        self.df = self.df[col_list]
        
        # Update displays
        self.update_data_preview()
        self.update_column_selectors()
    
    def move_column_down(self):
        """Move selected column down in the dataframe"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        # Get selected column names
        selected_items = self.column_list.selectedItems()
        
        if not selected_items or len(selected_items) != 1:
            QMessageBox.warning(self, "Error", "Please select exactly one column to move")
            return
        
        col_to_move = selected_items[0].text()
        
        # Get current column order
        col_list = self.df.columns.tolist()
        
        # Find index of the column to move
        idx = col_list.index(col_to_move)
        
        # Can't move further down if already at the bottom
        if idx == len(col_list) - 1:
            QMessageBox.information(self, "Information", "Column is already at the bottom")
            return
        
        # Swap with the column below
        col_list[idx], col_list[idx+1] = col_list[idx+1], col_list[idx]
        
        # Reorder columns
        self.df = self.df[col_list]
        
        # Update displays
        self.update_data_preview()
        self.update_column_selectors()
    
    def create_new_column(self):
        """Create a new column based on formula"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        # Get formula parameters
        formula_type = self.formula_type.currentIndex()
        new_col_name = self.new_col_name.toPlainText()
        
        if not new_col_name:
            QMessageBox.warning(self, "Error", "Please enter a name for the new column")
            return
        
        # Check if column name already exists
        if new_col_name in self.df.columns:
            response = QMessageBox.question(self, "Column Exists", 
                                           f"Column '{new_col_name}' already exists. Overwrite?",
                                           QMessageBox.Yes | QMessageBox.No)
            if response == QMessageBox.No:
                return
        
        try:
            if formula_type == 0:  # Simple Arithmetic
                col1 = self.formula_col1.currentText()
                col2 = self.formula_col2.currentText()
                operation = self.arithmetic_op.currentText()
                
                if not col1 or not col2:
                    QMessageBox.warning(self, "Error", "Please select both columns")
                    return
                
                # Perform arithmetic operation
                if operation == "+":
                    result = self.df[col1] + self.df[col2]
                elif operation == "-":
                    result = self.df[col1] - self.df[col2]
                elif operation == "*":
                    result = self.df[col1] * self.df[col2]
                elif operation == "/":
                    # Handle division by zero
                    result = self.df[col1] / self.df[col2].replace(0, np.nan)
                elif operation == "^":
                    result = self.df[col1] ** self.df[col2]
                
            elif formula_type == 1:  # Pythagorean Distance
                col1 = self.formula_col1.currentText()
                col2 = self.formula_col2.currentText()
                
                if not col1 or not col2:
                    QMessageBox.warning(self, "Error", "Please select both columns")
                    return
                
                # Calculate Euclidean distance (sqrt(x^2 + y^2))
                result = np.sqrt(self.df[col1]**2 + self.df[col2]**2)
                
            elif formula_type == 2:  # Angle Between Vectors
                col1 = self.formula_col1.currentText()
                col2 = self.formula_col2.currentText()
                
                if not col1 or not col2:
                    QMessageBox.warning(self, "Error", "Please select both columns")
                    return
                
                # Calculate angle (atan2(y, x)) in degrees
                result = np.degrees(np.arctan2(self.df[col2], self.df[col1]))
                
            elif formula_type == 3:  # Custom Formula
                custom_formula = self.custom_formula.toPlainText()
                
                if not custom_formula:
                    QMessageBox.warning(self, "Error", "Please enter a custom formula")
                    return
                
                # Replace column names in custom formula
                formula_eval = custom_formula
                
                for col in self.df.columns:
                    formula_eval = formula_eval.replace(f"{{{col}}}", f"self.df['{col}']")
                
                # Evaluate formula
                result = eval(formula_eval)
            
            # Add result to dataframe
            self.df[new_col_name] = result
            
            # Update displays
            self.update_data_preview()
            self.update_column_selectors()
            
            QMessageBox.information(self, "Success", f"Created new column '{new_col_name}'")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating column: {str(e)}")
    
    def update_visualization(self):
        """Update visualization plot based on selected options"""
        print("[Visualization] User triggered update plot...")
        if self.df is None:
            print("[Visualization] ERROR: No data loaded for visualization")
            QMessageBox.warning(self, "Error", "No data loaded")
            return

        plot_type = self.plot_type.currentIndex()

        try:
            self.vis_canvas.fig.clear()
            print(f"[Visualization] Plot type selected: {self.plot_type.currentText()}")

            if plot_type == 0:  # Scatter Plot
                print("[Visualization] Creating Scatter Plot")
                ...
            
            elif plot_type == 1:  # Polar Plot
                print("[Visualization] Creating Polar Plot")
                ...
            
            elif plot_type == 2:  # Coordinate Comparison
                print("[Visualization] Creating Coordinate Comparison Plot")
                ...
            
            elif plot_type == 3:  # Error Visualization
                print("[Visualization] Creating Error Visualization (Histogram + Boxplot)")
                ...

            self.vis_canvas.fig.tight_layout()
            self.vis_canvas.draw()
            self.statusBar().showMessage("Plot updated successfully!")
            print("[Visualization] Plot updated successfully ✅")

        except Exception as e:
            print(f"[Visualization] ERROR updating plot: {e}")
            QMessageBox.critical(self, "Error", f"Error updating plot: {str(e)}")
            self.statusBar().showMessage("Error during plot update ❌")

    
    def save_plot(self):
        """Save current plot to file"""
        print("[Plot] User triggered save plot...")
        if self.vis_canvas.fig is None:
            print("[Plot] ERROR: No plot to save")
            QMessageBox.warning(self, "Error", "No plot to save")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Plot", "", 
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
            )
            
            if not file_path:
                print("[Plot] Save cancelled by user.")
                return

            self.vis_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"[Plot] Plot saved successfully: {file_path}")
            self.statusBar().showMessage(f"Plot saved: {file_path}")
            QMessageBox.information(self, "Success", f"Plot saved to {file_path}")

        except Exception as e:
            print(f"[Plot] ERROR saving plot: {e}")
            QMessageBox.warning(self, "Save Error", f"Error saving plot: {str(e)}")
            self.statusBar().showMessage("Error during plot save ❌")

    
    def save_data(self):
        """Save transformed data to file"""
        print("[App] User triggered save data...")
        if self.df is None:
            print("[App] ERROR: No data to save")
            QMessageBox.warning(self, "Error", "No data to save")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Data", "", 
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
            )
            
            if not file_path:
                print("[App] Save cancelled by user.")
                return
            
            _, ext = os.path.splitext(file_path)
            if not ext:
                file_path += ".csv"
                ext = ".csv"
            
            if ext.lower() == '.csv':
                print("[App] Saving as CSV...")
                self.df.to_csv(file_path, index=False)
            elif ext.lower() in ['.xlsx', '.xls']:
                print("[App] Saving as Excel...")
                self.df.to_excel(file_path, index=False)
            else:
                print(f"[App] Unknown extension {ext}, saving as CSV by default")
                self.df.to_csv(file_path, index=False)
            
            print(f"[App] File saved successfully: {file_path}")
            self.statusBar().showMessage(f"File saved: {file_path}")
            QMessageBox.information(self, "Success", f"Data saved to {file_path}")
            
        except Exception as e:
            print(f"[App] ERROR during saving: {e}")
            QMessageBox.warning(self, "Save Error", f"Error saving data: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = CoordinateTransformApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()