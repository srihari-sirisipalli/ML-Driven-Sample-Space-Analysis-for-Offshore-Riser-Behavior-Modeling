import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import warnings
from sklearn.inspection import permutation_importance
from scipy import stats
# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
import statsmodels
# For Cook's Distance calculation
from statsmodels.api import OLS
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot
from statsmodels.stats.outliers_influence import OLSInfluence
import gc  # 🔥 import this if not already at top

# GUI Libraries
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QTabWidget, 
                             QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QComboBox, 
                             QSplitter, QTextEdit, QMessageBox, QProgressBar,
                             QSizePolicy, QListWidget, QAbstractItemView,
                             QCheckBox, QGroupBox, QRadioButton, QSpinBox, 
                             QDoubleSpinBox, QFormLayout, QGridLayout, QFrame,
                             QListWidgetItem, QDialog, QHeaderView, QInputDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QBrush

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Optionally import SHAP if available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP library not available. SHAP-based methods will be disabled.")

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


class ImportanceAnalysisWorker(QThread):
    """Worker thread to handle importance calculation and analysis"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    error_raised = pyqtSignal(str)
    analysis_finished = pyqtSignal(object)
    
    def __init__(self, df, input_features, target_variable, model_type, importance_method,
                 test_size, random_state, scaling_method, step_size, parent=None):
        super().__init__(parent)
        self.df = df
        self.input_features = input_features
        self.target_variable = target_variable
        self.model_type = model_type
        self.importance_method = importance_method
        self.test_size = test_size
        self.random_state = random_state
        self.scaling_method = scaling_method
        self.step_size = step_size
        
    def run(self):
        try:
            # Initialize results container
            results = {
                'importance_scores': None,
                'sorted_indices': None,
                'metrics_by_size': None,
                'optimal_size': None,
                'optimal_metrics': None,
                'model': None,
                'features': self.input_features,
                'target': self.target_variable,
                'method': self.importance_method
            }
            
            # Prepare data
            self.status_update.emit("Preparing data for analysis...")
            self.progress_update.emit(5)
            
            # Extract features and target
            X = self.df[self.input_features].copy()
            y = self.df[self.target_variable].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Apply scaling if selected
            X_scaled = X.copy()
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            if self.scaling_method == "StandardScaler":
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            elif self.scaling_method == "MinMaxScaler":
                scaler = MinMaxScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            elif self.scaling_method == "RobustScaler":
                scaler = RobustScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            
            # Create the model based on specified type
            self.status_update.emit(f"Creating {self.model_type} model...")
            self.progress_update.emit(10)
            
            if self.model_type == "LinearRegression":
                model = LinearRegression()
            elif self.model_type == "Ridge":
                model = Ridge(alpha=1.0, random_state=self.random_state)
            elif self.model_type == "RandomForest":
                model = RandomForestRegressor(n_estimators=10, random_state=self.random_state)
            elif self.model_type == "GradientBoosting":
                model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
            elif self.model_type == "SVR":
                model = SVR(C=1.0, epsilon=0.1)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Calculate importance scores based on method
            self.status_update.emit(f"Calculating importance using {self.importance_method}...")
            self.progress_update.emit(20)
            
            importance_scores = None
            
            if self.importance_method == "Cook's Distance":
                importance_scores = self._calculate_cooks_distance(X_scaled, y)
            elif self.importance_method == "Permutation Importance":
                importance_scores = self._calculate_permutation_importance(model, X_train_scaled, y_train)
            elif self.importance_method == "SHAP Values":
                if not SHAP_AVAILABLE:
                    raise ImportError("SHAP library is not available. Please install it to use SHAP values.")
                importance_scores = self._calculate_shap_values(model, X_train_scaled, y_train, X_scaled)
            elif self.importance_method == "Leverage":
                importance_scores = self._calculate_leverage(X_scaled)
            else:
                raise ValueError(f"Unsupported importance method: {self.importance_method}")
            
            # Sort data points by importance
            self.status_update.emit("Sorting data points by importance...")
            self.progress_update.emit(30)
            
            # Store original indices for reference
            importance_df = pd.DataFrame({
                'original_index': X.index,
                'importance': importance_scores
            })
            
            # Sort by importance in descending order (higher = more important)
            sorted_importance = importance_df.sort_values('importance', ascending=False)
            sorted_indices = sorted_importance['original_index'].values
            
            # Store results
            results['importance_scores'] = importance_scores
            results['sorted_indices'] = sorted_indices
            results['importance_df'] = sorted_importance
            
            # Evaluate model with different dataset sizes
            self.status_update.emit("Evaluating model performance with different dataset sizes...")
            self.progress_update.emit(40)
            
            metrics_by_size = self._evaluate_dataset_sizes(
                X, y, sorted_indices, model, self.step_size
            )
            
            results['metrics_by_size'] = metrics_by_size
            
            # Find optimal dataset size
            self.status_update.emit("Finding optimal dataset size...")
            self.progress_update.emit(90)
            
            optimal_size, optimal_metrics = self._find_optimal_size(metrics_by_size)
            
            results['optimal_size'] = optimal_size
            results['optimal_metrics'] = optimal_metrics
            results['model'] = self.model_type
            
            # Complete analysis
            self.status_update.emit("Analysis complete!")
            self.progress_update.emit(100)
            
            # Emit results
            self.analysis_finished.emit(results)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_raised.emit(f"Error during importance analysis: {str(e)}")
    
    def _calculate_cooks_distance(self, X, y):
        """Calculate Cook's Distance for each data point"""
        try:
            # Fit OLS model
            model = sm.OLS(y, sm.add_constant(X)).fit()
            
            # Calculate influence
            influence = OLSInfluence(model)
            
            # Get Cook's distances
            cooks_d = influence.cooks_distance[0]
            
            return cooks_d
        except Exception as e:
            print(f"Error calculating Cook's Distance: {str(e)}")
            raise
    
    def _calculate_permutation_importance(self, model, X, y):
        """Calculate permutation importance for each data point"""
        # Train the model
        model.fit(X, y)
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate baseline error
        baseline_error = mean_squared_error(y, y_pred)
        
        # Initialize importance scores
        importance = np.zeros(len(X))
        
        # For each data point, measure its impact on prediction
        for i in range(len(X)):
            # Create a copy of data excluding the current point
            X_without_i = X.drop(X.index[i])
            y_without_i = y.drop(y.index[i])
            
            # Train model without this point
            model_without_i = clone(model)
            model_without_i.fit(X_without_i, y_without_i)
            
            # Predict on full dataset
            y_pred_without_i = model_without_i.predict(X)
            
            # Calculate new error
            new_error = mean_squared_error(y, y_pred_without_i)
            
            # Importance is the change in error
            importance[i] = abs(baseline_error - new_error)
        
        return importance
    
    def _calculate_shap_values(self, model, X_train, y_train, X_full):
        """Calculate SHAP values safely for large datasets using batching"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library is not available")

        # Train the model
        model.fit(X_train, y_train)

        # Choose explainer
        if self.model_type in ["LinearRegression", "Ridge", "Lasso"]:
            explainer = shap.LinearExplainer(model, X_train)
        elif self.model_type in ["RandomForest", "GradientBoosting"]:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        else:
            explainer = shap.KernelExplainer(model.predict, X_train)

        # Safe batch processing
        batch_size = 500  # 🔥 Safer for 50k datapoints
        datapoint_importance = []

        for start_idx in range(0, len(X_full), batch_size):
            end_idx = min(start_idx + batch_size, len(X_full))
            batch = X_full.iloc[start_idx:end_idx]

            shap_values_batch = explainer.shap_values(batch)

            # Sum absolute SHAP values across features
            importance_batch = np.sum(np.abs(shap_values_batch), axis=1)

            datapoint_importance.extend(importance_batch)

            # 🔥 Free memory
            del shap_values_batch
            del batch
            gc.collect()

        return np.array(datapoint_importance)

    
    def _calculate_leverage(self, X):
        """Calculate leverage scores for each data point"""
        import numpy as np
        import pandas as pd

        # Add constant term for intercept
        X_with_const = np.hstack((np.ones((X.shape[0], 1)), X.values))

        # Compute hat matrix H = X (X^T X)^-1 X^T
        XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        H = X_with_const @ XtX_inv @ X_with_const.T

        # Diagonal elements are the leverage scores
        hat_matrix_diag = np.diag(H)

        return hat_matrix_diag

    
    def _evaluate_dataset_sizes(self, X, y, sorted_indices, model, step_size):
        """Evaluate model performance with different dataset sizes"""
        # Define dataset sizes to evaluate (percentages)
        size_percentages = np.arange(step_size, 100 + step_size, step_size)
        
        # Initialize results container
        metrics = []
        
        # For each dataset size
        for size_pct in size_percentages:
            # Calculate number of points to include
            n_points = int(np.ceil(len(X) * size_pct / 100))
            
            # Get indices of top n points
            top_indices = sorted_indices[:n_points]
            
            # Create dataset with only these points
            X_subset = X.loc[top_indices]
            y_subset = y.loc[top_indices]
            
            # Split into train and test
            X_subset_train, X_subset_test, y_subset_train, y_subset_test = train_test_split(
                X_subset, y_subset, test_size=self.test_size, random_state=self.random_state
            )
            
            # Apply scaling if needed
            if self.scaling_method == "StandardScaler":
                scaler = StandardScaler()
                X_subset_train = pd.DataFrame(
                    scaler.fit_transform(X_subset_train), 
                    columns=X_subset_train.columns, 
                    index=X_subset_train.index
                )
                X_subset_test = pd.DataFrame(
                    scaler.transform(X_subset_test), 
                    columns=X_subset_test.columns, 
                    index=X_subset_test.index
                )
            elif self.scaling_method == "MinMaxScaler":
                scaler = MinMaxScaler()
                X_subset_train = pd.DataFrame(
                    scaler.fit_transform(X_subset_train), 
                    columns=X_subset_train.columns, 
                    index=X_subset_train.index
                )
                X_subset_test = pd.DataFrame(
                    scaler.transform(X_subset_test), 
                    columns=X_subset_test.columns, 
                    index=X_subset_test.index
                )
            elif self.scaling_method == "RobustScaler":
                scaler = RobustScaler()
                X_subset_train = pd.DataFrame(
                    scaler.fit_transform(X_subset_train), 
                    columns=X_subset_train.columns, 
                    index=X_subset_train.index
                )
                X_subset_test = pd.DataFrame(
                    scaler.transform(X_subset_test), 
                    columns=X_subset_test.columns, 
                    index=X_subset_test.index
                )
            
            # Train model
            model_instance = clone(model)
            model_instance.fit(X_subset_train, y_subset_train)
            
            # Evaluate model
            y_train_pred = model_instance.predict(X_subset_train)
            y_test_pred = model_instance.predict(X_subset_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_subset_train, y_train_pred)
            test_r2 = r2_score(y_subset_test, y_test_pred)
            train_ev = explained_variance_score(y_subset_train, y_train_pred)
            test_ev = explained_variance_score(y_subset_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_subset_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_subset_test, y_test_pred))
            
            # Calculate efficiency metrics
            efficiency = size_pct / 100
            performance_efficiency = test_r2 / efficiency if efficiency > 0 else 0
            
            # Store metrics
            metrics.append({
                'Size (%)': size_pct,
                'Datapoints': n_points,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Train Explained Variance': train_ev,
                'Test Explained Variance': test_ev,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Efficiency': efficiency,
                'Performance Efficiency': performance_efficiency
            })
            
            # Update progress
            progress = 40 + int(50 * (size_pct / 100))
            self.progress_update.emit(progress)
            self.status_update.emit(f"Evaluating {size_pct}% dataset size...")
        
        return pd.DataFrame(metrics)
    
    def _find_optimal_size(self, metrics_df):
        """Find optimal dataset size based on performance and efficiency"""
        # We want to maximize R² while minimizing dataset size
        # One approach is to find the "elbow point" where adding more data gives diminishing returns
        
        # First, check if we have reasonable performance at smaller sizes
        good_performance_sizes = metrics_df[metrics_df['Test R²'] > 0.8]
        
        if not good_performance_sizes.empty:
            # If we have sizes with R² > 0.8, pick the smallest such size
            optimal_row = good_performance_sizes.iloc[0]
            return optimal_row['Size (%)'], optimal_row
        
        # Alternative: maximize performance efficiency (R² / size)
        max_efficiency_idx = metrics_df['Performance Efficiency'].idxmax()
        optimal_row = metrics_df.iloc[max_efficiency_idx]
        
        return optimal_row['Size (%)'], optimal_row


class MatplotlibCanvas(FigureCanvas):
    """Canvas for matplotlib plots"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class DatapointImportanceApp(QMainWindow):
    """Main application window for the Datapoint Importance Analysis Tool"""
    
    def __init__(self):
        super().__init__()
        warnings.filterwarnings('ignore')
        
        # Initialize UI
        self.setWindowTitle("Datapoint Importance Analysis Tool")
        self.setGeometry(100, 100, 1280, 800)
        
        # Data storage
        self.df = None
        self.file_path = None
        self.analysis_results = None
        
        # Initialize UI elements
        self.init_ui()
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
        
        # Update UI elements with the new data
        self.update_data_preview()
        self.update_feature_target_selectors()
        
        self.statusBar().showMessage(f"Data loaded: {rows} rows, {cols} columns")
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
        self.save_btn = QPushButton("Save Analysis Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        toolbar_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(toolbar_layout)
        
        # Progress bar for loading and analysis
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_data_preview_tab()
        self.create_feature_selection_tab()
        self.create_importance_analysis_tab()
        self.create_results_tab()
        self.create_optimization_tab()
        
        # Disable tabs initially
        self.tabs.setEnabled(False)
        
        # Status bar for messages
        self.statusBar().showMessage("Ready")
        self.create_optimal_report_tab()

    def create_optimal_report_tab(self):
        """Create tab for reporting the optimal dataset feature ranges"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Add checkbox for rounding
        self.round_to_int_checkbox = QCheckBox("Round all values to Integer")
        self.round_to_int_checkbox.setChecked(False)
        self.round_to_int_checkbox.stateChanged.connect(self.generate_optimal_dataset_report)
        layout.addWidget(self.round_to_int_checkbox)

        # Big text box for report
        self.optimal_report_text = QTextEdit()
        self.optimal_report_text.setReadOnly(True)
        self.optimal_report_text.setFont(QFont("Courier New", 10))
        layout.addWidget(self.optimal_report_text)

        # Button to export report
        export_btn = QPushButton("Export Report to File")
        export_btn.clicked.connect(self.export_optimal_report)
        layout.addWidget(export_btn)

        self.tabs.addTab(tab, "Optimal Dataset Report")


    def run_importance_analysis(self):
        """Run the importance analysis with selected settings"""
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
        
        # Get analysis settings
        model_type = self.model_type.currentText()
        importance_method = self.importance_method.currentText()
        test_size = self.test_size.value()
        random_state = self.random_state.value()
        scaling_method = self.scaling_method.currentText()
        step_size = self.step_size.value()
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Disable UI during analysis
        self.tabs.setEnabled(False)
        self.run_analysis_btn.setEnabled(False)
        
        # Start analysis worker thread
        self.analysis_worker = ImportanceAnalysisWorker(
            self.df, input_features, target_variable, model_type, importance_method,
            test_size, random_state, scaling_method, step_size
        )
        self.analysis_worker.progress_update.connect(self.update_progress)
        self.analysis_worker.status_update.connect(self.update_status)
        self.analysis_worker.error_raised.connect(self.show_error)
        self.analysis_worker.analysis_finished.connect(self.analysis_completed)
        self.analysis_worker.start()

    def get_selected_features(self):
        """Get the list of selected input features"""
        selected_features = []
        
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_features.append(item.text())
        
        return selected_features

    def analysis_completed(self, results):
        """Called when importance analysis is complete"""
        self.analysis_results = results
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Enable UI
        self.tabs.setEnabled(True)
        self.run_analysis_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        # Update results display
        self.update_analysis_results()
        
        # Switch to results tab
        self.tabs.setCurrentIndex(3)  # Results tab
        
        self.statusBar().showMessage("Importance analysis completed successfully!")

    def update_analysis_results(self):
        """Update the analysis results display"""
        if not self.analysis_results:
            return
        
        # Update summary
        self.update_summary_text()
        
        # Update importance plot
        self.update_importance_plot()
        
        # Update optimization results
        self.update_optimization_results()

    def update_summary_text(self):
        """Update the summary text with analysis results"""
        if not self.analysis_results:
            return
        
        # Clear previous content
        self.summary_text.clear()
        
        # Get basic info
        model_type = self.analysis_results['model']
        method = self.analysis_results['method']
        features = self.analysis_results['features']
        target = self.analysis_results['target']
        optimal_size = self.analysis_results['optimal_size']
        optimal_metrics = self.analysis_results['optimal_metrics']
        
        # Format summary text
        self.summary_text.append(f"Datapoint Importance Analysis Summary\n{'='*50}\n")
        self.summary_text.append(f"Model: {model_type}")
        self.summary_text.append(f"Importance Method: {method}")
        self.summary_text.append(f"Target Variable: {target}")
        self.summary_text.append(f"Number of Features: {len(features)}")
        self.summary_text.append(f"Features: {', '.join(features)}")
        
        if optimal_size and optimal_metrics is not None:
            self.summary_text.append(f"\nOptimal Dataset Size: {optimal_size:.1f}% ({int(optimal_metrics['Datapoints'])} datapoints)")
            self.summary_text.append(f"Test R²: {optimal_metrics['Test R²']:.4f}")
            
            # Calculate reduction
            total_points = len(self.df)
            reduced_points = int(optimal_metrics['Datapoints'])
            reduction = total_points - reduced_points
            reduction_pct = (reduction / total_points) * 100
            
            self.summary_text.append(f"Potential Reduction: {reduction} points ({reduction_pct:.1f}%)")

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



    def analyze_target_variable(self):
        """Create a detailed analysis of the target variable"""
        target_var = self.target_var.currentText()
        
        if not target_var or self.df is None:
            QMessageBox.warning(self, "Error", "No target variable selected")
            return
        
        if not pd.api.types.is_numeric_dtype(self.df[target_var]):
            QMessageBox.warning(self, "Error", "Target variable must be numeric for analysis")
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
    def create_importance_analysis_tab(self):
        """Create tab for importance analysis settings"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Split layout horizontally
        h_layout = QHBoxLayout()
        
        # Left side - Model selection
        left_group = QGroupBox("Model & Method Selection")
        left_layout = QFormLayout()
        left_group.setLayout(left_layout)
        
        # Model type selection
        self.model_type = QComboBox()
        self.model_type.addItems([
            "LinearRegression",
            "Ridge",
            "RandomForest",
            "GradientBoosting",
            "SVR"
        ])
        self.model_type.currentTextChanged.connect(self.update_importance_methods)
        left_layout.addRow("Model Type:", self.model_type)
        
        # Importance method selection
        self.importance_method = QComboBox()
        # Will be populated based on model type
        left_layout.addRow("Importance Method:", self.importance_method)
        
        # Test size
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.1, 0.5)
        self.test_size.setValue(0.2)
        self.test_size.setSingleStep(0.05)
        left_layout.addRow("Test Size:", self.test_size)
        
        # Random seed
        self.random_state = QSpinBox()
        self.random_state.setRange(0, 1000)
        self.random_state.setValue(42)
        left_layout.addRow("Random Seed:", self.random_state)
        
        # Data scaling
        self.scaling_method = QComboBox()
        self.scaling_method.addItems(["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])
        left_layout.addRow("Data Scaling:", self.scaling_method)
        
        # Dataset size step
        self.step_size = QDoubleSpinBox()
        self.step_size.setRange(5, 25)
        self.step_size.setValue(10)
        self.step_size.setSingleStep(5)
        left_layout.addRow("Size Step (%):", self.step_size)
        
        h_layout.addWidget(left_group)
        
        # Right side - Analysis description
        right_group = QGroupBox("Analysis Description")
        right_layout = QVBoxLayout()
        right_group.setLayout(right_layout)
        
        description_text = QTextEdit()
        description_text.setReadOnly(True)
        description_text.setHtml("""
            <h3>Datapoint Importance Analysis</h3>
            <p>This tool analyzes the importance of each datapoint in your dataset using various methods:</p>
            <ul>
                <li><b>Cook's Distance</b>: Measures influence of datapoints on regression parameters</li>
                <li><b>Leverage</b>: Identifies datapoints with extreme predictor values</li>
                <li><b>Permutation Importance</b>: Measures impact of excluding datapoints on model performance</li>
                <li><b>SHAP Values</b>: Calculates contribution of each datapoint to predictions</li>
            </ul>
            <p>The analysis will:</p>
            <ol>
                <li>Calculate importance scores for each datapoint</li>
                <li>Identify the most influential points</li>
                <li>Evaluate model performance with different dataset sizes</li>
                <li>Find the optimal dataset size for efficiency</li>
            </ol>
        """)
        right_layout.addWidget(description_text)
        
        h_layout.addWidget(right_group)
        
        layout.addLayout(h_layout)
        
        # Analysis button
        self.run_analysis_btn = QPushButton("Run Importance Analysis")
        self.run_analysis_btn.clicked.connect(self.run_importance_analysis)
        layout.addWidget(self.run_analysis_btn)
        
        self.tabs.addTab(tab, "Importance Analysis")
        
        # Initialize importance methods
        self.update_importance_methods(self.model_type.currentText())


    
    def update_importance_methods(self, model_type):
        """Update available importance methods based on selected model type"""
        self.importance_method.clear()
        
        # Add common methods available for all models
        common_methods = ["Cook's Distance", "Leverage"]
        self.importance_method.addItems(common_methods)
        
        # Add model-specific methods
        if model_type in ["LinearRegression", "Ridge"]:
            self.importance_method.addItem("Permutation Importance")
        elif model_type in ["RandomForest", "GradientBoosting"]:
            self.importance_method.addItem("Permutation Importance")
            if SHAP_AVAILABLE:
                self.importance_method.addItem("SHAP Values")
        elif model_type == "SVR":
            self.importance_method.addItem("Permutation Importance")

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

    def create_results_tab(self):
        """Create tab for displaying analysis results"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Summary section
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout()
        summary_group.setLayout(summary_layout)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Courier New", 10))
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        # Importance distribution plot
        plot_group = QGroupBox("Importance Distribution")
        plot_layout = QVBoxLayout()
        plot_group.setLayout(plot_layout)
        
        self.importance_canvas = MatplotlibCanvas(self, width=5, height=3)
        self.importance_toolbar = NavigationToolbar(self.importance_canvas, self)
        plot_layout.addWidget(self.importance_toolbar)
        plot_layout.addWidget(self.importance_canvas)
        
        # Top N points control
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Number of Top Datapoints:"))
        self.top_n_points = QSpinBox()
        self.top_n_points.setRange(10, 10000)
        self.top_n_points.setValue(100)
        self.top_n_points.setSingleStep(10)
        controls_layout.addWidget(self.top_n_points)
        
        self.update_top_n_btn = QPushButton("Update Plot")
        self.update_top_n_btn.clicked.connect(self.update_importance_plot)
        controls_layout.addWidget(self.update_top_n_btn)
        
        plot_layout.addLayout(controls_layout)
        
        layout.addWidget(plot_group)
        
        # Buttons for exporting
        export_layout = QHBoxLayout()
        
        self.export_important_btn = QPushButton("Export Important Datapoints")
        self.export_important_btn.clicked.connect(self.export_important_datapoints)
        export_layout.addWidget(self.export_important_btn)
        
        layout.addLayout(export_layout)
        
        self.tabs.addTab(tab, "Analysis Results")

    def create_optimization_tab(self):
        """Create tab for dataset size optimization"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Performance by dataset size
        perf_group = QGroupBox("Performance by Dataset Size")
        perf_layout = QVBoxLayout()
        perf_group.setLayout(perf_layout)
        
        # Table for metrics by size
        self.size_metrics_table = QTableWidget()
        perf_layout.addWidget(self.size_metrics_table)
        
        layout.addWidget(perf_group)
        
        # Performance visualization
        viz_group = QGroupBox("Performance Visualization")
        viz_layout = QVBoxLayout()
        viz_group.setLayout(viz_layout)
        
        self.performance_canvas = MatplotlibCanvas(self, width=5, height=3)
        self.performance_toolbar = NavigationToolbar(self.performance_canvas, self)
        viz_layout.addWidget(self.performance_toolbar)
        viz_layout.addWidget(self.performance_canvas)
        
        layout.addWidget(viz_group)
        
        # Efficiency visualization
        eff_group = QGroupBox("Efficiency Metrics")
        eff_layout = QVBoxLayout()
        eff_group.setLayout(eff_layout)
        
        self.efficiency_canvas = MatplotlibCanvas(self, width=5, height=3)
        self.efficiency_toolbar = NavigationToolbar(self.efficiency_canvas, self)
        eff_layout.addWidget(self.efficiency_toolbar)
        eff_layout.addWidget(self.efficiency_canvas)
        
        layout.addWidget(eff_group)
        
        # Optimal size info
        optimal_group = QGroupBox("Optimal Dataset Size")
        optimal_layout = QVBoxLayout()
        optimal_group.setLayout(optimal_layout)
        
        self.optimal_text = QTextEdit()
        self.optimal_text.setReadOnly(True)
        self.optimal_text.setFont(QFont("Courier New", 10))
        self.optimal_text.setMaximumHeight(150)
        optimal_layout.addWidget(self.optimal_text)
        
        export_btn = QPushButton("Export Optimal Dataset")
        export_btn.clicked.connect(self.export_optimal_dataset)
        optimal_layout.addWidget(export_btn)
        
        layout.addWidget(optimal_group)
        
        self.tabs.addTab(tab, "Size Optimization")

    def update_importance_plot(self):
        """Update the importance distribution plot with top N datapoints"""
        if not self.analysis_results or 'importance_scores' not in self.analysis_results:
            return
        
        n_points = self.top_n_points.value()
        importance_df = self.analysis_results['importance_df']
        
        # Clear canvas
        self.importance_canvas.fig.clear()
        ax = self.importance_canvas.fig.add_subplot(111)
        
        # Plot top N importance scores
        top_importance = importance_df.head(n_points)
        
        # Create plot
        sns.histplot(top_importance['importance'], kde=True, ax=ax)
        
        # Add mean and median lines
        mean_val = top_importance['importance'].mean()
        median_val = top_importance['importance'].median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.4f}')
        
        # Add labels and title
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Top {n_points} Importance Scores')
        ax.legend()
        
        # Draw plot
        self.importance_canvas.fig.tight_layout()
        self.importance_canvas.draw()
    def display_metrics_table(self, metrics_df):
        """Display performance metrics by dataset size in the table"""
        if metrics_df is None or metrics_df.empty:
            return
        
        # Get columns to display
        display_cols = metrics_df.columns
        
        # Setup table
        self.size_metrics_table.setRowCount(len(metrics_df))
        self.size_metrics_table.setColumnCount(len(display_cols))
        
        # Set headers
        self.size_metrics_table.setHorizontalHeaderLabels(display_cols)
        
        # Populate table
        for i in range(len(metrics_df)):
            for j, col in enumerate(display_cols):
                value = metrics_df.iloc[i, j]
                
                # Format numeric values
                if isinstance(value, (int, float, np.number)):
                    if 'Size' in col or 'Datapoints' in col:
                        text = f"{value:.0f}"
                    elif 'RMSE' in col:
                        text = f"{value:.4f}"
                    else:
                        text = f"{value:.4f}"
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
                
                # Highlight performance efficiency
                if col == 'Performance Efficiency':
                    if value > 1.5:
                        item.setBackground(QBrush(QColor(200, 255, 200)))  # Light green
                    elif value > 1.0:
                        item.setBackground(QBrush(QColor(230, 255, 230)))  # Lighter green
                
                self.size_metrics_table.setItem(i, j, item)
        
        # Resize columns to content
        self.size_metrics_table.resizeColumnsToContents()
        
        # Enable sorting
        self.size_metrics_table.setSortingEnabled(True)
    
    def visualize_performance_vs_size(self, metrics_df, optimal_size):
        """Visualize performance metrics vs dataset size"""
        if metrics_df is None or metrics_df.empty:
            return
        
        # Clear canvas
        self.performance_canvas.fig.clear()
        ax = self.performance_canvas.fig.add_subplot(111)
        
        # Plot R² vs dataset size
        ax.plot(metrics_df['Size (%)'], metrics_df['Train R²'], marker='o', label='Train R²')
        ax.plot(metrics_df['Size (%)'], metrics_df['Test R²'], marker='s', label='Test R²')
        
        # Add RMSE on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(metrics_df['Size (%)'], metrics_df['Test RMSE'], marker='^', color='r', label='Test RMSE')
        
        # Mark optimal size
        idx = metrics_df[metrics_df['Size (%)'] == optimal_size].index[0]
        test_r2 = metrics_df.loc[idx, 'Test R²']
        train_r2 = metrics_df.loc[idx, 'Train R²']
        test_rmse = metrics_df.loc[idx, 'Test RMSE']
        
        ax.axvline(x=optimal_size, color='k', linestyle='--', alpha=0.7)
        ax.scatter([optimal_size], [test_r2], color='orange', s=100, zorder=5, label=f'Optimal: {optimal_size}%')
        
        # Add labels and title
        ax.set_xlabel('Dataset Size (%)')
        ax.set_ylabel('R²')
        ax2.set_ylabel('RMSE')
        ax.set_title('Model Performance vs Dataset Size')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        self.performance_canvas.fig.tight_layout()
        self.performance_canvas.draw()


    def update_optimization_results(self):
        """Update the optimization results display"""
        if not self.analysis_results or 'metrics_by_size' not in self.analysis_results:
            return
        
        # Get metrics data
        metrics_df = self.analysis_results['metrics_by_size']
        optimal_size = self.analysis_results['optimal_size']
        optimal_metrics = self.analysis_results['optimal_metrics']
        
        # Display metrics table
        self.display_metrics_table(metrics_df)
        
        # Update performance visualization
        self.visualize_performance_vs_size(metrics_df, optimal_size)
        
        # Update efficiency visualization
        self.visualize_efficiency_metrics(metrics_df, optimal_size)
        
        # Update optimal size text
        self.display_optimal_size(optimal_size, optimal_metrics)
        self.generate_optimal_dataset_report()
    def generate_optimal_dataset_report(self):
        """Generate the report for the optimal dataset (only input features)"""
        if not self.analysis_results:
            self.optimal_report_text.setPlainText("No analysis results available.")
            return

        sorted_indices = self.analysis_results['sorted_indices']
        optimal_metrics = self.analysis_results['optimal_metrics']
        optimal_points = int(optimal_metrics['Datapoints'])
        optimal_indices = sorted_indices[:optimal_points]
        optimal_data = self.df.loc[optimal_indices]

        # Only selected input features
        input_features = self.analysis_results['features']
        selected_data = optimal_data[input_features]

        # Whether to round to integers
        round_to_int = self.round_to_int_checkbox.isChecked()

        report = []

        report.append(f"Optimal Dataset Report (Input Features Only)")
        report.append("="*60)
        report.append(f"Total datapoints: {len(selected_data)}\n")

        for col in selected_data.columns:
            if pd.api.types.is_numeric_dtype(selected_data[col]):
                values = selected_data[col].dropna().sort_values()

                if round_to_int:
                    values = values.round(0).astype(int)
                    values_list = values.tolist()
                    min_val = int(values.min())
                    max_val = int(values.max())
                    mean_val = int(values.mean())
                    median_val = int(values.median())
                else:
                    values = values.round(6)
                    values_list = values.tolist()
                    min_val = values.min()
                    max_val = values.max()
                    mean_val = values.mean()
                    median_val = values.median()

                total_points = len(values_list)
                unique_points = len(set(values_list))  # 🔥 UNIQUE POINT COUNT

                report.append(f"Feature: {col}")
                report.append(f"  Total Points: {total_points}")
                report.append(f"  Unique Points: {unique_points}")  # 🔥
                report.append(f"  Min: {min_val}")
                report.append(f"  Max: {max_val}")
                report.append(f"  Mean: {mean_val}")
                report.append(f"  Median: {median_val}")
                report.append(f"  Sorted Values:")
                report.append("    " + ", ".join(str(v) for v in set(values_list)))
                report.append("-"*60)

        self.optimal_report_text.setPlainText("\n".join(report))


    def export_optimal_report(self):
        """Export the optimal dataset report to a text file"""
        if self.optimal_report_text.toPlainText().strip() == "":
            QMessageBox.warning(self, "Error", "No report available to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            with open(file_path, "w") as f:
                f.write(self.optimal_report_text.toPlainText())
            QMessageBox.information(self, "Success", f"Report saved to:\n{file_path}")

        
    def visualize_efficiency_metrics(self, metrics_df, optimal_size):
        """Visualize efficiency metrics vs dataset size"""
        if metrics_df is None or metrics_df.empty:
            return
        
        # Clear canvas
        self.efficiency_canvas.fig.clear()
        ax = self.efficiency_canvas.fig.add_subplot(111)
        
        # Plot performance efficiency vs dataset size
        ax.plot(metrics_df['Size (%)'], metrics_df['Performance Efficiency'], marker='o', label='Performance Efficiency')
        
        # Mark optimal size
        idx = metrics_df[metrics_df['Size (%)'] == optimal_size].index[0]
        performance_efficiency = metrics_df.loc[idx, 'Performance Efficiency']
        
        ax.axvline(x=optimal_size, color='k', linestyle='--', alpha=0.7)
        ax.scatter([optimal_size], [performance_efficiency], color='orange', s=100, zorder=5, 
                  label=f'Optimal: {optimal_size}%')
        
        # Find the maximum efficiency point
        max_efficiency_idx = metrics_df['Performance Efficiency'].idxmax()
        max_efficiency_size = metrics_df.loc[max_efficiency_idx, 'Size (%)']
        max_efficiency = metrics_df.loc[max_efficiency_idx, 'Performance Efficiency']
        
        if max_efficiency_size != optimal_size:
            ax.scatter([max_efficiency_size], [max_efficiency], color='purple', s=100, zorder=5,
                      label=f'Max Efficiency: {max_efficiency_size}%')
        
        # Add 1.0 reference line for baseline efficiency
        ax.axhline(y=1.0, color='r', linestyle='-.', alpha=0.5, label='Baseline Efficiency')
        
        # Add labels and title
        ax.set_xlabel('Dataset Size (%)')
        ax.set_ylabel('Performance Efficiency (R² / Size Ratio)')
        ax.set_title('Performance Efficiency vs Dataset Size')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc='best')
        
        self.efficiency_canvas.fig.tight_layout()
        self.efficiency_canvas.draw()
    
    def display_optimal_size(self, optimal_size, optimal_metrics):
        """Display optimal dataset size information"""
        if optimal_metrics is None:
            return
        
        self.optimal_text.clear()
        
        # Format metrics as a readable string
        optimal_test_r2 = optimal_metrics['Test R²']
        optimal_train_r2 = optimal_metrics['Train R²']
        optimal_datapoints = optimal_metrics['Datapoints']
        optimal_efficiency = optimal_metrics['Performance Efficiency']
        
        self.optimal_text.append(f"Optimal Dataset Size: {optimal_size:.1f}% ({optimal_datapoints} datapoints)")
        self.optimal_text.append(f"Train R²: {optimal_train_r2:.4f}")
        self.optimal_text.append(f"Test R²: {optimal_test_r2:.4f}")
        self.optimal_text.append(f"Performance Efficiency: {optimal_efficiency:.4f}")
        
        # Add recommendation
        total_points = len(self.df)
        savings = total_points - optimal_datapoints
        savings_pct = (savings / total_points) * 100
        
        self.optimal_text.append(f"\nDataset Reduction: {savings} points ({savings_pct:.1f}% savings)")
        
        if savings_pct > 50:
            self.optimal_text.append("\nRecommendation: Significant dataset reduction possible with minimal performance impact.")
        elif savings_pct > 20:
            self.optimal_text.append("\nRecommendation: Moderate dataset reduction possible with minimal performance impact.")
        else:
            self.optimal_text.append("\nRecommendation: Limited dataset reduction possible. Most datapoints are important.")
    
    def export_important_datapoints(self):
        """Export the top important datapoints to a file"""
        if not self.analysis_results or 'sorted_indices' not in self.analysis_results:
            QMessageBox.warning(self, "Error", "No analysis results available")
            return
        
        # Get export path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Important Datapoints", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Get data
            sorted_indices = self.analysis_results['sorted_indices']
            importance_df = self.analysis_results['importance_df']
            
            # Ask how many top points to export
            top_n, ok = QInputDialog.getInt(
                self, "Export Top N", "Enter number of top datapoints to export:", 
                min(self.top_n_points.value(), len(sorted_indices)), 1, len(sorted_indices), 1)
            
            if not ok:
                return
            
            # Get datapoints
            top_indices = sorted_indices[:top_n]
            top_data = self.df.loc[top_indices].copy()
            
            # Add importance scores
            top_data['Importance'] = importance_df.loc[top_indices, 'importance'].values
            
            # Ensure proper extension
            _, ext = os.path.splitext(file_path)
            if not ext:
                file_path += ".csv"
                ext = ".csv"
            
            # Save based on extension
            if ext.lower() == '.csv':
                top_data.to_csv(file_path, index=False)
            elif ext.lower() == '.xlsx':
                top_data.to_excel(file_path, index=False)
            else:
                # Default to CSV
                top_data.to_csv(file_path, index=False)
            
            QMessageBox.information(self, "Export Successful", 
                                   f"Top {top_n} datapoints exported to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting datapoints: {str(e)}")
    
    def export_optimal_dataset(self):
        """Export the optimal dataset to a file"""
        if not self.analysis_results or 'optimal_size' not in self.analysis_results:
            QMessageBox.warning(self, "Error", "No analysis results available")
            return
        
        # Get export path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Optimal Dataset", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Get data
            sorted_indices = self.analysis_results['sorted_indices']
            optimal_metrics = self.analysis_results['optimal_metrics']
            
            # Get optimal datapoints
            optimal_points = int(optimal_metrics['Datapoints'])
            optimal_indices = sorted_indices[:optimal_points]
            optimal_data = self.df.loc[optimal_indices].copy()
            
            # Ensure proper extension
            _, ext = os.path.splitext(file_path)
            if not ext:
                file_path += ".csv"
                ext = ".csv"
            
            # Save based on extension
            if ext.lower() == '.csv':
                optimal_data.to_csv(file_path, index=False)
            elif ext.lower() == '.xlsx':
                optimal_data.to_excel(file_path, index=False)
            else:
                # Default to CSV
                optimal_data.to_csv(file_path, index=False)
            
            QMessageBox.information(self, "Export Successful", 
                                   f"Optimal dataset ({optimal_points} points) exported to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting optimal dataset: {str(e)}")
    
    def save_results(self):
        """Save all analysis results to a directory"""
        if not self.analysis_results:
            QMessageBox.warning(self, "Error", "No analysis results available")
            return
        
        # Get directory path
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Results"
        )
        
        if not dir_path:
            return
        
        try:
            # Create timestamped folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join(dir_path, f"ImportanceAnalysis_{timestamp}")
            
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Save analysis summary
            self._save_analysis_summary(results_dir)
            
            # Save importance scores
            self._save_importance_scores(results_dir)
            
            # Save performance metrics
            self._save_performance_metrics(results_dir)
            
            # Save optimal dataset
            self._save_optimal_dataset(results_dir)
            
            # Save plots
            self._save_analysis_plots(results_dir)
            
            QMessageBox.information(self, "Save Successful", f"Analysis results saved to:\n{results_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving results: {str(e)}")
    
    def _save_analysis_summary(self, results_dir):
        """Save analysis summary to a text file"""
        summary_path = os.path.join(results_dir, "analysis_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write(self.summary_text.toPlainText())
    
    def _save_importance_scores(self, results_dir):
        """Save importance scores to a CSV file"""
        importance_path = os.path.join(results_dir, "importance_scores.csv")
        
        importance_df = self.analysis_results['importance_df']
        importance_df.to_csv(importance_path, index=False)
    
    def _save_performance_metrics(self, results_dir):
        """Save performance metrics to a CSV file"""
        metrics_path = os.path.join(results_dir, "performance_metrics.csv")
        
        metrics_df = self.analysis_results['metrics_by_size']
        metrics_df.to_csv(metrics_path, index=False)
    
    def _save_optimal_dataset(self, results_dir):
        """Save optimal dataset to a CSV file"""
        optimal_path = os.path.join(results_dir, "optimal_dataset.csv")
        
        sorted_indices = self.analysis_results['sorted_indices']
        optimal_metrics = self.analysis_results['optimal_metrics']
        
        # Get optimal datapoints
        optimal_points = int(optimal_metrics['Datapoints'])
        optimal_indices = sorted_indices[:optimal_points]
        optimal_data = self.df.loc[optimal_indices].copy()
        
        optimal_data.to_csv(optimal_path, index=False)
    
    def _save_analysis_plots(self, results_dir):
        """Save analysis plots to PNG files"""
        plots_dir = os.path.join(results_dir, "plots")
        
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Save importance distribution plot
        importance_plot_path = os.path.join(plots_dir, "importance_distribution.png")
        self.importance_canvas.fig.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
        
        # Save performance plot
        performance_plot_path = os.path.join(plots_dir, "performance_vs_size.png")
        self.performance_canvas.fig.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
        
        # Save efficiency plot
        efficiency_plot_path = os.path.join(plots_dir, "efficiency_metrics.png")
        self.efficiency_canvas.fig.savefig(efficiency_plot_path, dpi=300, bbox_inches='tight')


def main():
    app = QApplication(sys.argv)
    window = DatapointImportanceApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()