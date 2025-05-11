import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# GUI Libraries
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QTabWidget, 
                             QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QComboBox, 
                             QSplitter, QTextEdit, QMessageBox, QProgressBar,
                             QSizePolicy, QListWidget, QAbstractItemView,
                             QCheckBox, QGroupBox, QRadioButton, QSpinBox, 
                             QDoubleSpinBox, QFormLayout, QGridLayout, QFrame,
                             QListWidgetItem, QDialog, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QBrush

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
    loading_finished = pyqtSignal(object)
    
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.df = None
        
    def run(self):
        try:
            self.status_update.emit("Loading file...")
            self.progress_update.emit(10)
            
            # Check file extension
            _, ext = os.path.splitext(self.file_path)
            
            if ext.lower() in ['.xlsx', '.xls']:
                # Try to detect if we need to skip rows
                try:
                    df_test = pd.read_excel(self.file_path)
                    self.status_update.emit("Checking file structure...")
                    
                    # Check if we need to skip rows
                    if 'Unnamed' in str(df_test.columns[0]) or df_test.shape[1] < 5:
                        self.status_update.emit("Detected header rows, skipping first 2 rows...")
                        self.df = pd.read_excel(self.file_path, skiprows=2)
                    else:
                        self.df = df_test
                except Exception as e:
                    self.error_raised.emit(f"Error pre-processing Excel file: {str(e)}")
                    return
            elif ext.lower() == '.csv':
                self.df = pd.read_csv(self.file_path)
            else:
                self.error_raised.emit(f"Unsupported file format: {ext}")
                return
                
            # Basic data cleaning
            self.progress_update.emit(40)
            self.status_update.emit("Cleaning data...")
            
            # Handle unnamed columns
            unnamed_cols = [col for col in self.df.columns if 'Unnamed' in str(col)]
            if unnamed_cols:
                self.status_update.emit(f"Found {len(unnamed_cols)} unnamed columns")
                if len(unnamed_cols) == 1 and unnamed_cols[0] == self.df.columns[0]:
                    self.df = self.df.iloc[:, 1:]
            
            # Convert numeric columns
            for col in self.df.columns:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except:
                    pass  # Keep as is if not numeric
            
            self.progress_update.emit(70)
            self.status_update.emit("Processing complete!")
            self.progress_update.emit(100)
            
            # Emit result
            self.loading_finished.emit(self.df)
            
        except Exception as e:
            self.error_raised.emit(f"Error processing file: {str(e)}")


class ModelComparisonWorker(QThread):
    """Worker thread to handle model comparison without freezing GUI"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    error_raised = pyqtSignal(str)
    comparison_finished = pyqtSignal(object)
    
    def __init__(self, df, input_features, target_variable, test_sizes, models, 
                 random_state, scaling_method, cross_validation, cv_folds, parent=None):
        super().__init__(parent)
        self.df = df
        self.input_features = input_features
        self.target_variable = target_variable
        self.test_sizes = test_sizes
        self.models = models
        self.random_state = random_state
        self.scaling_method = scaling_method
        self.cross_validation = cross_validation
        self.cv_folds = cv_folds
        
    def run(self):
        try:
            comparison_results = {}
            
            # Prepare X and y
            X = self.df[self.input_features].copy()
            y = self.df[self.target_variable].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Initialize results data structure
            results_data = []
            overall_best_model = None
            overall_best_score = -float('inf')
            
            # Calculate total iterations for progress updates
            total_iterations = len(self.test_sizes) * len(self.models)
            current_iteration = 0
            
            # For each test size
            for test_size in self.test_sizes:
                self.status_update.emit(f"Evaluating test size: {test_size:.2f}")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state
                )
                
                # Scale features if selected
                if self.scaling_method == "StandardScaler":
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                elif self.scaling_method == "MinMaxScaler":
                    scaler = MinMaxScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                elif self.scaling_method == "RobustScaler":
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_train_scaled = X_train.values
                    X_test_scaled = X_test.values
                
                # For each selected model
                for model_name, model in self.models.items():
                    self.status_update.emit(f"Testing {model_name} with test size {test_size:.2f}")
                    
                    # Create a fresh model instance
                    model_instance = clone(model)
                    
                    # Fit the model
                    model_instance.fit(X_train_scaled, y_train)
                    
                    # Calculate metrics
                    y_train_pred = model_instance.predict(X_train_scaled)
                    y_test_pred = model_instance.predict(X_test_scaled)
                    
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    
                    train_rmse = np.sqrt(train_mse)
                    test_rmse = np.sqrt(test_mse)
                    
                    train_mae = mean_absolute_error(y_train, y_train_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    
                    # Cross-validation if requested
                    cv_score_mean = None
                    cv_score_std = None
                    
                    if self.cross_validation:
                        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                        cv_scores = cross_val_score(model_instance, X_train_scaled, y_train, cv=kf, scoring='r2')
                        cv_score_mean = np.mean(cv_scores)
                        cv_score_std = np.std(cv_scores)
                    
                    # Store results
                    result_entry = {
                        'Model': model_name,
                        'Test Size': test_size,
                        'Train R²': train_r2,
                        'Test R²': test_r2,
                        'Gap': train_r2 - test_r2,
                        'Train RMSE': train_rmse,
                        'Test RMSE': test_rmse,
                        'Train MAE': train_mae,
                        'Test MAE': test_mae
                    }
                    
                    if self.cross_validation:
                        result_entry['CV R² Mean'] = cv_score_mean
                        result_entry['CV R² Std'] = cv_score_std
                    
                    # Check if this is the best model so far based on test R²
                    if test_r2 > overall_best_score:
                        overall_best_model = model_name
                        overall_best_score = test_r2
                    
                    results_data.append(result_entry)
                    
                    # Update progress
                    current_iteration += 1
                    progress = int((current_iteration / total_iterations) * 100)
                    self.progress_update.emit(progress)
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results_data)
            
            # Organize comparison results
            comparison_results = {
                'results_df': results_df,
                'best_model': overall_best_model,
                'best_score': overall_best_score,
                'features': self.input_features,
                'target': self.target_variable
            }
            
            self.status_update.emit("Comparison complete!")
            self.progress_update.emit(100)
            
            # Emit results
            self.comparison_finished.emit(comparison_results)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_raised.emit(f"Error during model comparison: {str(e)}")


class MatplotlibCanvas(FigureCanvas):
    """Canvas for matplotlib plots"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class ModelComparisonApp(QMainWindow):
    """Main application window for Model Comparison Tool"""
    
    def __init__(self):
        super().__init__()
        warnings.filterwarnings('ignore')
        
        # Initialize UI
        self.setWindowTitle("Model Comparison Tool")
        self.setGeometry(100, 100, 1280, 800)
        
        # Data storage
        self.df = None
        self.file_path = None
        self.comparison_results = None
        
        # Initialize UI elements
        self.init_ui()

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
        self.save_btn = QPushButton("Save Comparison Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        toolbar_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(toolbar_layout)
        
        # Progress bar for loading and comparison
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_data_preview_tab()
        self.create_feature_selection_tab()
        self.create_model_selection_tab()
        self.create_comparison_results_tab()
        
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
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Data table
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)
        
        self.tabs.addTab(tab, "Data Preview")
    
    def create_feature_selection_tab(self):
        """Create tab for selecting input features and target variable"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Split layout horizontally
        h_layout = QHBoxLayout()
        
        # Left side - Features selection
        left_group = QGroupBox("Input Features")
        left_layout = QVBoxLayout()
        left_group.setLayout(left_layout)
        
        # Feature list with checkboxes
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        left_layout.addWidget(QLabel("Select Input Features:"))
        left_layout.addWidget(self.feature_list)
        
        # Select/Deselect All buttons
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_features)
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all_features)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.deselect_all_btn)
        left_layout.addLayout(btn_layout)
        
        # Right side - Target variable selection
        right_group = QGroupBox("Target Variable")
        right_layout = QVBoxLayout()
        right_group.setLayout(right_layout)
        
        right_layout.addWidget(QLabel("Select Target Variable:"))
        self.target_var = QComboBox()
        right_layout.addWidget(self.target_var)
        
        # Target variable info
        self.target_info_frame = QFrame()
        self.target_info_frame.setFrameShape(QFrame.StyledPanel)
        self.target_info_layout = QVBoxLayout()
        self.target_info_frame.setLayout(self.target_info_layout)
        
        self.target_info_label = QLabel("No target variable selected")
        self.target_info_layout.addWidget(self.target_info_label)
        
        self.target_dist_canvas = MatplotlibCanvas(self, width=4, height=3)
        self.target_info_layout.addWidget(self.target_dist_canvas)
        
        right_layout.addWidget(self.target_info_frame)
        right_layout.addStretch()
        
        # Add button to analyze selected target
        self.analyze_target_btn = QPushButton("Analyze Target Variable")
        self.analyze_target_btn.clicked.connect(self.analyze_target_variable)
        right_layout.addWidget(self.analyze_target_btn)
        
        # Add left and right groups to horizontal layout
        h_layout.addWidget(left_group, 60)
        h_layout.addWidget(right_group, 40)
        
        layout.addLayout(h_layout)
        
        self.tabs.addTab(tab, "Feature Selection")
        
        # Connect target variable change to handler
        self.target_var.currentTextChanged.connect(self.target_variable_changed)
    
    def create_model_selection_tab(self):
        """Create tab for model selection and comparison settings"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Split layout horizontally
        h_layout = QHBoxLayout()
        
        # Left side - Model selection
        left_group = QGroupBox("Select Models to Compare")
        left_layout = QVBoxLayout()
        left_group.setLayout(left_layout)
        
        # Model checkboxes
        self.model_checkboxes = {}
        
        self.model_checkboxes["LinearRegression"] = QCheckBox("Linear Regression")
        self.model_checkboxes["Ridge"] = QCheckBox("Ridge Regression")
        self.model_checkboxes["Lasso"] = QCheckBox("Lasso Regression")
        self.model_checkboxes["ElasticNet"] = QCheckBox("Elastic Net")
        self.model_checkboxes["SVR"] = QCheckBox("Support Vector Regression (SVR)")
        self.model_checkboxes["RandomForest"] = QCheckBox("Random Forest")
        self.model_checkboxes["GradientBoosting"] = QCheckBox("Gradient Boosting")
        
        # Add checkboxes to layout
        for checkbox in self.model_checkboxes.values():
            left_layout.addWidget(checkbox)
        
        # Select/Deselect All buttons
        btn_layout = QHBoxLayout()
        self.select_all_models_btn = QPushButton("Select All Models")
        self.select_all_models_btn.clicked.connect(self.select_all_models)
        self.deselect_all_models_btn = QPushButton("Deselect All Models")
        self.deselect_all_models_btn.clicked.connect(self.deselect_all_models)
        btn_layout.addWidget(self.select_all_models_btn)
        btn_layout.addWidget(self.deselect_all_models_btn)
        left_layout.addLayout(btn_layout)
        
        # Right side - Comparison settings
        right_group = QGroupBox("Comparison Settings")
        right_layout = QFormLayout()
        right_group.setLayout(right_layout)
        
        # Test size range
        test_size_layout = QHBoxLayout()
        self.min_test_size = QDoubleSpinBox()
        self.min_test_size.setRange(0.1, 0.9)
        self.min_test_size.setValue(0.2)
        self.min_test_size.setSingleStep(0.05)
        self.min_test_size.valueChanged.connect(self.validate_test_size_range)
        
        self.max_test_size = QDoubleSpinBox()
        self.max_test_size.setRange(0.1, 0.9)
        self.max_test_size.setValue(0.4)
        self.max_test_size.setSingleStep(0.05)
        self.max_test_size.valueChanged.connect(self.validate_test_size_range)
        
        self.test_size_steps = QSpinBox()
        self.test_size_steps.setRange(1, 10)
        self.test_size_steps.setValue(3)
        
        test_size_layout.addWidget(QLabel("Min:"))
        test_size_layout.addWidget(self.min_test_size)
        test_size_layout.addWidget(QLabel("Max:"))
        test_size_layout.addWidget(self.max_test_size)
        test_size_layout.addWidget(QLabel("Steps:"))
        test_size_layout.addWidget(self.test_size_steps)
        
        right_layout.addRow("Test Size Range:", test_size_layout)
        
        # Random seed
        self.random_state = QSpinBox()
        self.random_state.setRange(0, 1000)
        self.random_state.setValue(42)
        right_layout.addRow("Random Seed:", self.random_state)
        
        # Data scaling
        self.scaling_method = QComboBox()
        self.scaling_method.addItems(["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])
        right_layout.addRow("Data Scaling:", self.scaling_method)
        
        # Cross-validation
        self.use_cross_val = QCheckBox("Use cross-validation")
        self.use_cross_val.setChecked(True)
        right_layout.addRow("", self.use_cross_val)
        
        self.cv_folds = QSpinBox()
        self.cv_folds.setRange(2, 10)
        self.cv_folds.setValue(5)
        right_layout.addRow("CV Folds:", self.cv_folds)
        
        # Add left and right groups to horizontal layout
        h_layout.addWidget(left_group, 40)
        h_layout.addWidget(right_group, 60)
        
        layout.addLayout(h_layout)
        
        # Run comparison button
        self.run_comparison_btn = QPushButton("Run Model Comparison")
        self.run_comparison_btn.clicked.connect(self.run_model_comparison)
        layout.addWidget(self.run_comparison_btn)
        
        self.tabs.addTab(tab, "Model Selection")
    
    def create_comparison_results_tab(self):
        """Create tab for displaying comparison results"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Results summary section
        summary_group = QGroupBox("Comparison Summary")
        summary_layout = QVBoxLayout()
        summary_group.setLayout(summary_layout)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Courier New", 10))
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        # Results table
        results_group = QGroupBox("Detailed Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        
        self.results_table = QTableWidget()
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        # Results visualization
        viz_group = QGroupBox("Results Visualization")
        viz_layout = QVBoxLayout()
        viz_group.setLayout(viz_layout)
        
        # Visualization controls
        controls_layout = QHBoxLayout()
        
        self.viz_metric = QComboBox()
        self.viz_metric.addItems(["Test R²", "Train R²", "Test RMSE", "Train RMSE", "Gap"])
        controls_layout.addWidget(QLabel("Select Metric:"))
        controls_layout.addWidget(self.viz_metric)
        
        self.update_viz_btn = QPushButton("Update Visualization")
        self.update_viz_btn.clicked.connect(self.update_visualization)
        controls_layout.addWidget(self.update_viz_btn)
        
        viz_layout.addLayout(controls_layout)
        
        # Plot canvas
        self.results_canvas = MatplotlibCanvas(self, width=6, height=4)
        self.results_toolbar = NavigationToolbar(self.results_canvas, self)
        viz_layout.addWidget(self.results_toolbar)
        viz_layout.addWidget(self.results_canvas)
        
        layout.addWidget(viz_group)
        
        self.tabs.addTab(tab, "Comparison Results")
    
    def load_data(self):
        """Open file dialog and load selected data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", 
            "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"Loading: {os.path.basename(file_path)}...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Disable UI during loading
            self.tabs.setEnabled(False)
            self.load_btn.setEnabled(False)
            
            # Start data loading thread
            self.data_loader = DataLoader(file_path)
            self.data_loader.progress_update.connect(self.update_progress)
            self.data_loader.status_update.connect(self.update_status)
            self.data_loader.error_raised.connect(self.show_error)
            self.data_loader.loading_finished.connect(self.data_loaded)
            self.data_loader.start()
    
    def update_progress(self, value):
        """Update progress bar value"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """Update status bar with message"""
        self.statusBar().showMessage(message)
    
    def show_error(self, message):
        """Display error message"""
        self.progress_bar.setVisible(False)
        self.file_label.setText("Error loading file")
        self.load_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", message)
    
    def data_loaded(self, df):
        """Called when data is successfully loaded"""
        self.df = df
        self.file_label.setText(f"Loaded: {os.path.basename(self.file_path)}")
        self.progress_bar.setVisible(False)
        
        # Update data info
        rows, cols = self.df.shape
        self.data_info_label.setText(f"Rows: {rows}, Columns: {cols}")
        
        # Enable tabs
        self.tabs.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        # Update all UI elements with the new data
        self.update_data_preview()
        self.update_feature_target_selectors()
        
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
    
    def update_feature_target_selectors(self):
        """Update feature and target variable selectors with column names"""
        if self.df is None:
            return
        
        # Clear existing items
        self.feature_list.clear()
        self.target_var.clear()
        
        # Get column names
        columns = self.df.columns.tolist()
        
        # Update feature list
        for col in columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.feature_list.addItem(item)
        
        # Update target variable selector
        self.target_var.addItems(columns)
    
    def select_all_features(self):
        """Select all features in the feature list"""
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            item.setCheckState(Qt.Checked)
    
    def deselect_all_features(self):
        """Deselect all features in the feature list"""
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            item.setCheckState(Qt.Unchecked)
    
    def select_all_models(self):
        """Select all models"""
        for checkbox in self.model_checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all_models(self):
        """Deselect all models"""
        for checkbox in self.model_checkboxes.values():
            checkbox.setChecked(False)
    
    def validate_test_size_range(self):
        """Ensure min test size is less than max test size"""
        min_val = self.min_test_size.value()
        max_val = self.max_test_size.value()
        
        if min_val >= max_val:
            # Reset to valid values
            self.min_test_size.setValue(max_val - 0.1)
            QMessageBox.warning(self, "Invalid Range", "Minimum test size must be less than maximum test size")
    
    def target_variable_changed(self, target_var):
        """Handle target variable change"""
        if not target_var or self.df is None:
            self.target_info_label.setText("No target variable selected")
            return
        
        # Update target info
        self.target_info_label.setText(f"Selected Target: {target_var}")
        
        # Analyze basic statistics (if numeric)
        if target_var in self.df.columns and pd.api.types.is_numeric_dtype(self.df[target_var]):
            stats = self.df[target_var].describe()
            stats_text = "\n".join([f"{idx}: {val:.4f}" for idx, val in stats.items()])
            self.target_info_label.setText(f"Selected Target: {target_var}\n\nStatistics:\n{stats_text}")
        
        # Create initial distribution plot
        self.plot_target_distribution(target_var)
    
    def plot_target_distribution(self, target_var):
        """Plot distribution of the target variable"""
        if self.df is None or target_var not in self.df.columns:
            return
        
        # Clear the canvas
        self.target_dist_canvas.fig.clear()
        ax = self.target_dist_canvas.fig.add_subplot(111)
        
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(self.df[target_var]):
            # Create histogram with density plot
            sns.histplot(self.df[target_var], kde=True, ax=ax)
            
            # Add mean and median lines
            mean_val = self.df[target_var].mean()
            median_val = self.df[target_var].median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
            ax.legend()
            
            ax.set_title(f"Distribution of {target_var}")
            ax.set_xlabel(target_var)
            ax.set_ylabel("Frequency")
        else:
            # For categorical variables, create a count plot
            sns.countplot(y=self.df[target_var], ax=ax)
            ax.set_title(f"Value Counts for {target_var}")
            ax.set_xlabel("Count")
            ax.set_ylabel(target_var)
        
        # Draw the plot
        self.target_dist_canvas.fig.tight_layout()
        self.target_dist_canvas.draw()
    
    def analyze_target_variable(self):
        """Create a detailed analysis of the target variable"""
        target_var = self.target_var.currentText()
        
        if not target_var or self.df is None:
            QMessageBox.warning(self, "Error", "No target variable selected")
            return
        
        if not pd.api.types.is_numeric_dtype(self.df[target_var]):
            QMessageBox.warning(self, "Error", "Target variable must be numeric for regression analysis")
            return
        
        # Create a dialog with detailed analysis
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Target Analysis: {target_var}")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        
        # Create tabs for different visualizations
        tabs = QTabWidget()
        
        # Create distribution tab
        dist_tab = QWidget()
        dist_layout = QVBoxLayout()
        dist_canvas = MatplotlibCanvas(width=6, height=4)
        dist_toolbar = NavigationToolbar(dist_canvas, dialog)
        dist_layout.addWidget(dist_toolbar)
        dist_layout.addWidget(dist_canvas)
        dist_tab.setLayout(dist_layout)
        
        # Create box plot tab
        box_tab = QWidget()
        box_layout = QVBoxLayout()
        box_canvas = MatplotlibCanvas(width=6, height=4)
        box_toolbar = NavigationToolbar(box_canvas, dialog)
        box_layout.addWidget(box_toolbar)
        box_layout.addWidget(box_canvas)
        box_tab.setLayout(box_layout)
        
        # Create QQ-plot tab
        qq_tab = QWidget()
        qq_layout = QVBoxLayout()
        qq_canvas = MatplotlibCanvas(width=6, height=4)
        qq_toolbar = NavigationToolbar(qq_canvas, dialog)
        qq_layout.addWidget(qq_toolbar)
        qq_layout.addWidget(qq_canvas)
        qq_tab.setLayout(qq_layout)
        
        # Create statistics tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setFont(QFont("Courier New", 10))
        stats_layout.addWidget(stats_text)
        stats_tab.setLayout(stats_layout)
        
        # Add tabs
        tabs.addTab(dist_tab, "Distribution")
        tabs.addTab(box_tab, "Box Plot")
        tabs.addTab(qq_tab, "Q-Q Plot")
        tabs.addTab(stats_tab, "Statistics")
        
        layout.addWidget(tabs)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        
        # Create plots
        # Distribution plot
        dist_canvas.fig.clear()
        ax1 = dist_canvas.fig.add_subplot(111)
        sns.histplot(self.df[target_var], kde=True, ax=ax1)
        ax1.set_title(f"Distribution of {target_var}")
        ax1.set_xlabel(target_var)
        ax1.set_ylabel("Frequency")
        
        # Add mean and median lines
        mean_val = self.df[target_var].mean()
        median_val = self.df[target_var].median()
        ax1.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
        ax1.legend()
        
        dist_canvas.fig.tight_layout()
        dist_canvas.draw()
        
        # Box plot
        box_canvas.fig.clear()
        ax2 = box_canvas.fig.add_subplot(111)
        sns.boxplot(y=self.df[target_var], ax=ax2)
        ax2.set_title(f"Box Plot for {target_var}")
        ax2.set_ylabel(target_var)
        
        # Add annotations for quartiles
        q1 = self.df[target_var].quantile(0.25)
        q2 = self.df[target_var].quantile(0.5)
        q3 = self.df[target_var].quantile(0.75)
        iqr = q3 - q1
        whisker_low = max(self.df[target_var].min(), q1 - 1.5 * iqr)
        whisker_high = min(self.df[target_var].max(), q3 + 1.5 * iqr)
        
        ax2.text(0.05, q1, f'Q1: {q1:.2f}', horizontalalignment='center', size='small', color='blue', weight='semibold')
        ax2.text(0.05, q2, f'Q2: {q2:.2f}', horizontalalignment='center', size='small', color='blue', weight='semibold')
        ax2.text(0.05, q3, f'Q3: {q3:.2f}', horizontalalignment='center', size='small', color='blue', weight='semibold')
        ax2.text(0.05, whisker_low, f'Lower: {whisker_low:.2f}', horizontalalignment='center', size='small', color='red')
        ax2.text(0.05, whisker_high, f'Upper: {whisker_high:.2f}', horizontalalignment='center', size='small', color='red')
        
        box_canvas.fig.tight_layout()
        box_canvas.draw()
        
        # QQ-plot
        qq_canvas.fig.clear()
        ax3 = qq_canvas.fig.add_subplot(111)
        stats.probplot(self.df[target_var].dropna(), plot=ax3)
        ax3.set_title(f"Q-Q Plot for {target_var}")
        
        qq_canvas.fig.tight_layout()
        qq_canvas.draw()
        
        # Statistics
        data = self.df[target_var]
        basic_stats = data.describe()
        
        # Additional statistics
        skewness = data.skew()
        kurtosis = data.kurtosis()
        missing = data.isnull().sum()
        missing_pct = missing / len(data) * 100
        
        # Shapiro-Wilk test for normality
        if len(data.dropna()) >= 3 and len(data.dropna()) <= 5000:  # Shapiro-Wilk has sample size limits
            shapiro_test = stats.shapiro(data.dropna())
            shapiro_pvalue = shapiro_test[1]
            normality_test = f"Shapiro-Wilk Test p-value: {shapiro_pvalue:.6f}"
            if shapiro_pvalue < 0.05:
                normality_test += " (Data is not normally distributed)"
            else:
                normality_test += " (Data is normally distributed)"
        else:
            normality_test = "Shapiro-Wilk Test: Not applicable (sample size constraints)"
        
        # Format statistics text
        stats_text.append(f"Statistical Analysis for {target_var}\n")
        stats_text.append("=" * 50 + "\n")
        stats_text.append("Basic Statistics:")
        for idx, val in basic_stats.items():
            stats_text.append(f"  {idx}: {val:.6f}")
        
        stats_text.append("\nAdditional Statistics:")
        stats_text.append(f"  Skewness: {skewness:.6f}")
        stats_text.append(f"  Kurtosis: {kurtosis:.6f}")
        stats_text.append(f"  Missing Values: {missing} ({missing_pct:.2f}%)")
        
        stats_text.append(f"\nNormality Test:")
        stats_text.append(f"  {normality_test}")
        
        # Show the dialog
        dialog.exec_()
    
    def get_selected_features(self):
        """Get the list of selected input features"""
        selected_features = []
        
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_features.append(item.text())
        
        return selected_features
    
    def get_selected_models(self):
        """Get dictionary of selected models with their instances"""
        selected_models = {}
        
        if self.model_checkboxes["LinearRegression"].isChecked():
            selected_models["LinearRegression"] = LinearRegression()
        
        if self.model_checkboxes["Ridge"].isChecked():
            selected_models["Ridge"] = Ridge(alpha=1.0, random_state=self.random_state.value())
        
        if self.model_checkboxes["Lasso"].isChecked():
            selected_models["Lasso"] = Lasso(alpha=0.1, random_state=self.random_state.value())
        
        if self.model_checkboxes["ElasticNet"].isChecked():
            selected_models["ElasticNet"] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state.value())
        
        if self.model_checkboxes["SVR"].isChecked():
            selected_models["SVR"] = SVR(C=1.0, epsilon=0.1)
        
        if self.model_checkboxes["RandomForest"].isChecked():
            selected_models["RandomForest"] = RandomForestRegressor(n_estimators=100, random_state=self.random_state.value())
        
        if self.model_checkboxes["GradientBoosting"].isChecked():
            selected_models["GradientBoosting"] = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state.value())
        
        return selected_models
    
    def get_test_sizes(self):
        """Get list of test sizes to evaluate"""
        min_size = self.min_test_size.value()
        max_size = self.max_test_size.value()
        steps = self.test_size_steps.value()
        
        return np.linspace(min_size, max_size, steps)
    
    def run_model_comparison(self):
        """Run model comparison with selected settings"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        # Get selected features and target
        input_features = self.get_selected_features()
        target_variable = self.target_var.currentText()
        
        if not input_features:
            QMessageBox.warning(self, "Error", "No input features selected")
            return
        
        if not target_variable:
            QMessageBox.warning(self, "Error", "No target variable selected")
            return
        
        if target_variable in input_features:
            QMessageBox.warning(self, "Error", "Target variable cannot be used as an input feature")
            return
        
        # Check if target is numeric (required for regression)
        if not pd.api.types.is_numeric_dtype(self.df[target_variable]):
            QMessageBox.warning(self, "Error", "Target variable must be numeric for regression analysis")
            return
        
        # Get selected models
        selected_models = self.get_selected_models()
        
        if not selected_models:
            QMessageBox.warning(self, "Error", "No models selected for comparison")
            return
        
        # Get test sizes to evaluate
        test_sizes = self.get_test_sizes()
        
        # Get other comparison settings
        random_state = self.random_state.value()
        scaling_method = self.scaling_method.currentText()
        use_cross_val = self.use_cross_val.isChecked()
        cv_folds = self.cv_folds.value()
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Disable UI during comparison
        self.tabs.setEnabled(False)
        self.run_comparison_btn.setEnabled(False)
        
        # Start comparison worker thread
        self.comparison_worker = ModelComparisonWorker(
            self.df, input_features, target_variable, test_sizes, selected_models,
            random_state, scaling_method, use_cross_val, cv_folds
        )
        self.comparison_worker.progress_update.connect(self.update_progress)
        self.comparison_worker.status_update.connect(self.update_status)
        self.comparison_worker.error_raised.connect(self.show_error)
        self.comparison_worker.comparison_finished.connect(self.comparison_completed)
        self.comparison_worker.start()
    
    def comparison_completed(self, results):
        """Called when model comparison is complete"""
        self.comparison_results = results
        self.progress_bar.setVisible(False)
        
        # Enable UI
        self.tabs.setEnabled(True)
        self.run_comparison_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        # Update results display
        self.update_comparison_results()
        
        # Switch to results tab
        self.tabs.setCurrentIndex(3)  # Results tab
        
        self.statusBar().showMessage("Model comparison completed!")
    
    def update_comparison_results(self):
        """Update the comparison results display"""
        if not self.comparison_results:
            return
        
        # Get results data
        results_df = self.comparison_results['results_df']
        best_model = self.comparison_results['best_model']
        best_score = self.comparison_results['best_score']
        features = self.comparison_results['features']
        target = self.comparison_results['target']
        
        # Update summary text
        self.summary_text.clear()
        self.summary_text.append(f"Model Comparison Results\n{'='*50}\n")
        self.summary_text.append(f"Target Variable: {target}")
        self.summary_text.append(f"Number of Features: {len(features)}")
        self.summary_text.append(f"Number of Models Compared: {len(results_df['Model'].unique())}")
        self.summary_text.append(f"Test Size Range: {min(results_df['Test Size']):.2f} - {max(results_df['Test Size']):.2f}")
        self.summary_text.append(f"\nBest Performing Model: {best_model}")
        self.summary_text.append(f"Best Test R² Score: {best_score:.6f}")
        
        # Find best model for each test size
        self.summary_text.append(f"\nBest Model by Test Size:")
        for test_size in sorted(results_df['Test Size'].unique()):
            subset = results_df[results_df['Test Size'] == test_size]
            best_idx = subset['Test R²'].idxmax()
            best_model_at_size = subset.loc[best_idx, 'Model']
            best_r2 = subset.loc[best_idx, 'Test R²']
            self.summary_text.append(f"  Test Size {test_size:.2f}: {best_model_at_size} (R² = {best_r2:.6f})")
        
        # Update results table
        self.display_results_table(results_df)
        
        # Update visualization
        self.update_visualization()
    
    def display_results_table(self, results_df):
        """Display results in the table widget"""
        # Reset table
        self.results_table.clear()
        
        # Get columns to display
        columns = list(results_df.columns)
        
        # Set dimensions
        self.results_table.setRowCount(len(results_df))
        self.results_table.setColumnCount(len(columns))
        
        # Set headers
        self.results_table.setHorizontalHeaderLabels(columns)
        
        # Fill data
        for i in range(len(results_df)):
            for j, col in enumerate(columns):
                value = results_df.iloc[i, j]
                
                # Format numeric values
                if isinstance(value, (int, float, np.number)):
                    if col == 'Test Size':
                        text = f"{value:.2f}"
                    else:
                        text = f"{value:.6f}"
                else:
                    text = str(value)
                
                item = QTableWidgetItem(text)
                
                # Color code R² cells
                if 'R²' in col:
                    if value > 0.9:
                        item.setBackground(QBrush(QColor(200, 255, 200)))  # Light green
                    elif value > 0.8:
                        item.setBackground(QBrush(QColor(230, 255, 230)))  # Lighter green
                    elif value < 0.5:
                        item.setBackground(QBrush(QColor(255, 200, 200)))  # Light red
                
                # Color code gap cells (difference between train and test R²)
                if col == 'Gap':
                    if value > 0.2:
                        item.setBackground(QBrush(QColor(255, 200, 200)))  # Light red (high overfit)
                    elif value < 0.05:
                        item.setBackground(QBrush(QColor(200, 255, 200)))  # Light green (low overfit)
                        
                self.results_table.setItem(i, j, item)
        
        # Resize columns to content
        self.results_table.resizeColumnsToContents()
        
        # Enable sorting
        self.results_table.setSortingEnabled(True)
    
    def update_visualization(self):
        """Update the results visualization based on selected metric"""
        if not self.comparison_results:
            return
        
        results_df = self.comparison_results['results_df']
        selected_metric = self.viz_metric.currentText()
        
        # Clear the canvas
        self.results_canvas.fig.clear()
        ax = self.results_canvas.fig.add_subplot(111)
        
        # Create a lineplot of the selected metric vs test size for each model
        for model_name in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model_name]
            ax.plot(model_data['Test Size'], model_data[selected_metric], marker='o', label=model_name)
        
        # Add labels and title
        ax.set_xlabel('Test Size')
        ax.set_ylabel(selected_metric)
        ax.set_title(f'{selected_metric} vs Test Size by Model')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend()
        
        # Draw the plot
        self.results_canvas.fig.tight_layout()
        self.results_canvas.draw()
    
    def save_results(self):
        """Save comparison results to a file"""
        if not self.comparison_results:
            QMessageBox.warning(self, "Error", "No comparison results to save")
            return
        
        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Comparison Results", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Ensure proper extension
            _, ext = os.path.splitext(file_path)
            if not ext:
                file_path += ".csv"
                ext = ".csv"
            
            # Get results dataframe
            results_df = self.comparison_results['results_df']
            
            # Format summary information
            summary_info = pd.DataFrame({
                'Info': ['Target Variable', 'Features', 'Best Model', 'Best R² Score'],
                'Value': [
                    self.comparison_results['target'],
                    ', '.join(self.comparison_results['features']),
                    self.comparison_results['best_model'],
                    f"{self.comparison_results['best_score']:.6f}"
                ]
            })
            
            # Save based on file extension
            if ext.lower() == '.csv':
                # For CSV, save summary and results in separate files
                results_df.to_csv(file_path, index=False)
                
                # Create a summary file path
                summary_path = file_path.replace('.csv', '_summary.csv')
                summary_info.to_csv(summary_path, index=False)
                
                self.statusBar().showMessage(f"Results saved to {file_path} and {summary_path}")
                QMessageBox.information(self, "Save Successful", 
                                       f"Results saved to:\n{file_path}\n{summary_path}")
                
            elif ext.lower() == '.xlsx':
                # For Excel, save summary and results in different sheets
                with pd.ExcelWriter(file_path) as writer:
                    summary_info.to_excel(writer, sheet_name='Summary', index=False)
                    results_df.to_excel(writer, sheet_name='Results', index=False)
                
                self.statusBar().showMessage(f"Results saved to {file_path}")
                QMessageBox.information(self, "Save Successful", f"Results saved to:\n{file_path}")
                
            else:
                # Default to CSV
                results_df.to_csv(file_path, index=False)
                self.statusBar().showMessage(f"Results saved to {file_path}")
                QMessageBox.information(self, "Save Successful", f"Results saved to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving results: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = ModelComparisonApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()