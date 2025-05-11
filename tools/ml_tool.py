import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.base import clone


# GUI Libraries
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QTabWidget, 
                             QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QComboBox, 
                             QSplitter, QTextEdit, QMessageBox, QProgressBar,
                             QSizePolicy, QListWidget, QAbstractItemView,
                             QCheckBox, QGroupBox, QRadioButton, QSpinBox, 
                             QDoubleSpinBox, QFormLayout, QGridLayout, QFrame,
                             QListWidgetItem, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

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
                # Try to detect if we need to skip rows (like in the provided scripts)
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


class ModelTrainer(QThread):
    """Worker thread to handle model training without freezing GUI"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    error_raised = pyqtSignal(str)
    training_finished = pyqtSignal(object)
    
    def __init__(self, df, input_features, target_variable, model_type, 
                 test_size, random_state, scaling_method, hyperparameters, 
                 use_cross_val, cv_folds, do_feature_selection, feature_selection_method,
                 num_features, parent=None):
        super().__init__(parent)
        self.df = df
        self.input_features = input_features
        self.target_variable = target_variable
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.scaling_method = scaling_method
        self.hyperparameters = hyperparameters
        self.use_cross_val = use_cross_val
        self.cv_folds = cv_folds
        self.do_feature_selection = do_feature_selection
        self.feature_selection_method = feature_selection_method
        self.num_features = num_features
    
    def run(self):
        try:
            results = {}
            
            self.status_update.emit("Preparing data...")
            self.progress_update.emit(10)
            
            # Prepare X and y
            X = self.df[self.input_features].copy()
            y = self.df[self.target_variable].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            self.status_update.emit("Applying preprocessing...")
            self.progress_update.emit(20)
            
            # Initialize scaler
            if self.scaling_method == "StandardScaler":
                scaler = StandardScaler()
            elif self.scaling_method == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif self.scaling_method == "RobustScaler":
                scaler = RobustScaler()
            else:
                scaler = None
            
            # Apply scaling if selected
            if scaler:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values
            
            # Feature selection if requested
            feature_selector = None
            selected_features = self.input_features
            
            if self.do_feature_selection:
                self.status_update.emit("Performing feature selection...")
                self.progress_update.emit(30)
                
                if self.feature_selection_method == "SelectKBest":
                    feature_selector = SelectKBest(f_regression, k=min(self.num_features, len(self.input_features)))
                    X_train_scaled = feature_selector.fit_transform(X_train_scaled, y_train)
                    X_test_scaled = feature_selector.transform(X_test_scaled)
                    
                    # Get selected feature names
                    feature_mask = feature_selector.get_support()
                    selected_features = [self.input_features[i] for i in range(len(self.input_features)) if feature_mask[i]]
                
                elif self.feature_selection_method == "RFE":
                    base_estimator = None
                    if self.model_type == "LinearRegression":
                        base_estimator = LinearRegression()
                    elif self.model_type == "Ridge":
                        base_estimator = Ridge()
                    elif self.model_type == "RandomForest":
                        base_estimator = RandomForestRegressor(random_state=self.random_state)
                    else:
                        base_estimator = LinearRegression()
                    
                    feature_selector = RFE(base_estimator, n_features_to_select=min(self.num_features, len(self.input_features)))
                    X_train_scaled = feature_selector.fit_transform(X_train_scaled, y_train)
                    X_test_scaled = feature_selector.transform(X_test_scaled)
                    
                    # Get selected feature names
                    feature_mask = feature_selector.get_support()
                    selected_features = [self.input_features[i] for i in range(len(self.input_features)) if feature_mask[i]]
            
            self.status_update.emit("Training model...")
            self.progress_update.emit(50)
            
            # Initialize model based on type
            if self.model_type == "LinearRegression":
                model = LinearRegression()
            elif self.model_type == "Ridge":
                alpha = self.hyperparameters.get("alpha", 1.0)
                model = Ridge(alpha=alpha, random_state=self.random_state)
            elif self.model_type == "Lasso":
                alpha = self.hyperparameters.get("alpha", 1.0)
                model = Lasso(alpha=alpha, random_state=self.random_state)
            elif self.model_type == "ElasticNet":
                alpha = self.hyperparameters.get("alpha", 1.0)
                l1_ratio = self.hyperparameters.get("l1_ratio", 0.5)
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=self.random_state)
            elif self.model_type == "SVR":
                C = self.hyperparameters.get("C", 1.0)
                epsilon = self.hyperparameters.get("epsilon", 0.1)
                kernel = self.hyperparameters.get("kernel", "rbf")
                model = SVR(C=C, epsilon=epsilon, kernel=kernel)
            elif self.model_type == "RandomForest":
                n_estimators = self.hyperparameters.get("n_estimators", 100)
                max_depth = self.hyperparameters.get("max_depth", None)
                min_samples_split = self.hyperparameters.get("min_samples_split", 2)
                model = RandomForestRegressor(
                    n_estimators=n_estimators, 
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=self.random_state
                )
            elif self.model_type == "GradientBoosting":
                n_estimators = self.hyperparameters.get("n_estimators", 100)
                learning_rate = self.hyperparameters.get("learning_rate", 0.1)
                max_depth = self.hyperparameters.get("max_depth", 3)
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=self.random_state
                )
            else:
                model = LinearRegression()
            
            # Store original data for later plotting
            results['X_train'] = X_train
            results['X_test'] = X_test
            results['y_train'] = y_train
            results['y_test'] = y_test
            results['scaled_data'] = {
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled
            }
            results['selected_features'] = selected_features
            
            # Cross validation if requested
            cv_scores = None
            if self.use_cross_val:
                self.status_update.emit("Performing cross-validation...")
                self.progress_update.emit(60)
                
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='r2')
            
            # Train the model
            self.status_update.emit("Fitting model...")
            self.progress_update.emit(70)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate the model
            self.status_update.emit("Evaluating model...")
            self.progress_update.emit(80)
            
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            train_explained_var = explained_variance_score(y_train, y_train_pred)
            
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_explained_var = explained_variance_score(y_test, y_test_pred)


            # Store results
            results['model'] = model
            results['model_type'] = self.model_type
            results['predictions'] = {
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred
            }
            results['metrics'] = {
                'train': {
                    'mse': train_mse,
                    'rmse': train_rmse,
                    'mae': train_mae,
                    'r2': train_r2,
                    'explained_variance': train_explained_var
                },
                'test': {
                    'mse': test_mse,
                    'rmse': test_rmse,
                    'mae': test_mae,
                    'r2': test_r2,
                    'explained_variance': test_explained_var
                }
            }
            # Additional error metrics
            train_absolute_errors = np.abs(y_train - y_train_pred)
            test_absolute_errors = np.abs(y_test - y_test_pred)

            train_ape = np.abs((y_train - y_train_pred) / np.where(y_train != 0, y_train, np.nan)) * 100
            test_ape = np.abs((y_test - y_test_pred) / np.where(y_test != 0, y_test, np.nan)) * 100

            train_total_error = np.sum(train_absolute_errors)
            test_total_error = np.sum(test_absolute_errors)

            train_mape = np.nanmean(train_ape)
            test_mape = np.nanmean(test_ape)
            train_total_signed_error = np.sum(y_train - y_train_pred)
            test_total_signed_error = np.sum(y_test - y_test_pred)

            # Add to results dictionary
            results['metrics']['train'].update({
                'total_error': train_total_error,
                'mape': train_mape
            })
            results['metrics']['test'].update({
                'total_error': test_total_error,
                'mape': test_mape
            })
            results['metrics']['train'].update({
                'total_signed_error': train_total_signed_error
            })
            results['metrics']['test'].update({
                'total_signed_error': test_total_signed_error
            })




            if cv_scores is not None:
                results['cv_scores'] = {
                    'scores': cv_scores,
                    'mean': np.mean(cv_scores),
                    'std': np.std(cv_scores)
                }
            
            # Extract feature importances if applicable
            if hasattr(model, 'coef_'):
                if self.do_feature_selection:
                    results['feature_importances'] = model.coef_
                else:
                    coefficients = model.coef_
                    if len(coefficients.shape) > 1:
                        coefficients = coefficients.ravel()
                    results['feature_importances'] = {
                        'features': self.input_features,
                        'importances': coefficients
                    }
            elif hasattr(model, 'feature_importances_'):
                results['feature_importances'] = {
                    'features': selected_features,
                    'importances': model.feature_importances_
                }
            
            self.status_update.emit("Training complete!")
            self.progress_update.emit(100)
            
            # Emit results
            self.training_finished.emit(results)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_raised.emit(f"Error during model training: {str(e)}")


class MatplotlibCanvas(FigureCanvas):
    """Canvas for matplotlib plots"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class MLApp(QMainWindow):
    """Main application window for Machine Learning Tool"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.setWindowTitle("PyQt5 Machine Learning Tool")
        self.setGeometry(100, 100, 1280, 800)
        
        # Data storage
        self.df = None
        self.file_path = None
        self.model_results = None
        
        # Initialize UI elements
        self.init_ui()
        self.create_test_size_sensitivity_tab()

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
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
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
        
        # Create tabs for different analyses
        self.create_data_preview_tab()
        self.create_feature_selection_tab()
        self.create_model_config_tab()
        self.create_results_tab()
        self.create_predictions_tab()
        self.create_error_analysis_tab()

        # Disable tabs initially
        self.tabs.setEnabled(False)
        
        # Status bar for messages
        self.statusBar().showMessage("Ready")
    def create_error_analysis_tab(self):
        """Create a tab for error analysis visualization with selectable plot types"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Control Panel
        control_panel = QGroupBox("Error Analysis Controls")
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        
        # Plot type selection
        plot_type_layout = QHBoxLayout()
        plot_type_layout.addWidget(QLabel("Select Plot Type:"))
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Absolute Error by Index",
            "Absolute Error Distribution",
            "Signed Error Distribution",
            "Error Boxplot",
            "Error Scatterplot vs. Predicted"
        ])
        self.plot_type_combo.currentIndexChanged.connect(self.plot_error_analysis)
        plot_type_layout.addWidget(self.plot_type_combo)
        plot_type_layout.addStretch()
        
        control_layout.addLayout(plot_type_layout)
        
        # Bin size control
        bin_layout = QHBoxLayout()
        bin_layout.addWidget(QLabel("Histogram Settings:"))
        
        self.bin_size_spin = QDoubleSpinBox()
        self.bin_size_spin.setRange(0.0, 1000.0)
        self.bin_size_spin.setValue(0.0)
        self.bin_size_spin.setSingleStep(0.01)
        self.bin_size_spin.setPrefix("Bin Width: ")
        self.bin_size_spin.valueChanged.connect(self.plot_error_analysis)
        
        self.bin_count_spin = QSpinBox()
        self.bin_count_spin.setRange(5, 100)
        self.bin_count_spin.setValue(20)
        self.bin_count_spin.setPrefix("Bin Count: ")
        self.bin_count_spin.valueChanged.connect(self.plot_error_analysis)
        
        bin_layout.addWidget(self.bin_size_spin)
        bin_layout.addWidget(self.bin_count_spin)
        bin_layout.addStretch()
        
        control_layout.addLayout(bin_layout)
        
        # Training/Test set visibility options
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Show:"))
        
        self.show_train_check = QCheckBox("Training Set")
        self.show_train_check.setChecked(True)
        self.show_train_check.stateChanged.connect(self.plot_error_analysis)
        
        self.show_test_check = QCheckBox("Test Set")
        self.show_test_check.setChecked(True)
        self.show_test_check.stateChanged.connect(self.plot_error_analysis)
        
        dataset_layout.addWidget(self.show_train_check)
        dataset_layout.addWidget(self.show_test_check)
        dataset_layout.addStretch()
        
        control_layout.addLayout(dataset_layout)
        
        # Add control panel to main layout
        layout.addWidget(control_panel)
        
        # Plot button
        self.plot_error_btn = QPushButton("Generate Error Analysis Plot")
        self.plot_error_btn.clicked.connect(self.plot_error_analysis)
        layout.addWidget(self.plot_error_btn)
        
        # Error statistics summary
        self.error_stats_text = QTextEdit()
        self.error_stats_text.setReadOnly(True)
        self.error_stats_text.setMaximumHeight(100)
        layout.addWidget(QLabel("Error Statistics:"))
        layout.addWidget(self.error_stats_text)
        
        # Matplotlib canvas for the plot
        self.error_canvas = MatplotlibCanvas(self, width=7, height=5)
        self.error_toolbar = NavigationToolbar(self.error_canvas, self)
        
        layout.addWidget(self.error_toolbar)
        layout.addWidget(self.error_canvas)
        
        self.tabs.addTab(tab, "Error Analysis")

    def plot_error_analysis(self):
        """Plot different error analysis visualizations based on selected plot type"""
        if not self.model_results:
            return
            
        # Get data
        y_train = self.model_results['y_train']
        y_test = self.model_results['y_test']
        y_train_pred = self.model_results['predictions']['y_train_pred']
        y_test_pred = self.model_results['predictions']['y_test_pred']
        
        # Calculate errors
        abs_train_errors = np.abs(y_train - y_train_pred)
        abs_test_errors = np.abs(y_test - y_test_pred)
        signed_train_errors = y_train - y_train_pred
        signed_test_errors = y_test - y_test_pred
        
        # Update error statistics
        self.update_error_statistics(
            abs_train_errors, abs_test_errors,
            signed_train_errors, signed_test_errors
        )
        
        # Check visibility options
        show_train = self.show_train_check.isChecked()
        show_test = self.show_test_check.isChecked()
        
        if not show_train and not show_test:
            # If neither is checked, default to showing both
            show_train = True
            show_test = True
            self.show_train_check.setChecked(True)
            self.show_test_check.setChecked(True)
        
        # Get plot type
        plot_type = self.plot_type_combo.currentText()
        
        # Clear the canvas
        self.error_canvas.fig.clear()
        ax = self.error_canvas.fig.add_subplot(111)
        
        # Get bin settings
        bin_size = self.bin_size_spin.value()
        bin_count = self.bin_count_spin.value()
        
        # Determine bins based on settings
        if bin_size > 0:
            if "Absolute Error" in plot_type:
                max_val = max(abs_train_errors.max() if show_train else 0, 
                            abs_test_errors.max() if show_test else 0)
                bins = np.arange(0, max_val + bin_size, bin_size)
            else:  # Signed error
                max_abs = max(
                    abs(signed_train_errors.min()) if show_train else 0,
                    abs(signed_train_errors.max()) if show_train else 0,
                    abs(signed_test_errors.min()) if show_test else 0,
                    abs(signed_test_errors.max()) if show_test else 0
                )
                bins = np.arange(-max_abs - bin_size, max_abs + bin_size, bin_size)
        else:
            bins = bin_count
        
        # Create the selected plot
        if plot_type == "Absolute Error by Index":
            # Plot absolute errors vs index
            if show_train:
                train_indices = range(len(abs_train_errors))
                ax.scatter(train_indices, abs_train_errors, label='Training Set', alpha=0.6)
            
            if show_test:
                test_start_idx = len(abs_train_errors) if show_train else 0
                test_indices = range(test_start_idx, test_start_idx + len(abs_test_errors))
                ax.scatter(test_indices, abs_test_errors, label='Test Set', alpha=0.6, color='orange')
            
            ax.set_title("Absolute Error by Index")
            ax.set_xlabel("Index")
            ax.set_ylabel("Absolute Error")
            
        elif plot_type == "Absolute Error Distribution":
            # Histogram of absolute errors
            if show_train:
                sns.histplot(abs_train_errors, kde=True, label='Training Set', 
                            color='blue', ax=ax, stat='density', bins=bins)
            
            if show_test:
                sns.histplot(abs_test_errors, kde=True, label='Test Set', 
                            color='orange', ax=ax, stat='density', bins=bins)
            
            # Add mean lines
            if show_train:
                mean_train = abs_train_errors.mean()
                ax.axvline(mean_train, color='blue', linestyle='--', 
                        label=f'Mean Train: {mean_train:.4f}')
            
            if show_test:
                mean_test = abs_test_errors.mean()
                ax.axvline(mean_test, color='orange', linestyle='--', 
                        label=f'Mean Test: {mean_test:.4f}')
            
            ax.set_title("Distribution of Absolute Errors")
            ax.set_xlabel("Absolute Error")
            ax.set_ylabel("Density")
            
        elif plot_type == "Signed Error Distribution":
            # Histogram of signed errors
            if show_train:
                sns.histplot(signed_train_errors, kde=True, label='Training Set', 
                            color='blue', ax=ax, stat='density', bins=bins)
            
            if show_test:
                sns.histplot(signed_test_errors, kde=True, label='Test Set', 
                            color='orange', ax=ax, stat='density', bins=bins)
            
            # Add zero line and mean lines
            ax.axvline(0, color='black', linestyle='--', label='Zero Error')
            
            if show_train:
                mean_train = signed_train_errors.mean()
                ax.axvline(mean_train, color='blue', linestyle='--', 
                        label=f'Mean Train: {mean_train:.4f}')
            
            if show_test:
                mean_test = signed_test_errors.mean()
                ax.axvline(mean_test, color='orange', linestyle='--', 
                        label=f'Mean Test: {mean_test:.4f}')
            
            ax.set_title("Distribution of Signed Errors")
            ax.set_xlabel("Signed Error (Actual - Predicted)")
            ax.set_ylabel("Density")
            
        elif plot_type == "Error Boxplot":
            # Boxplot of errors
            boxplot_data = []
            boxplot_labels = []
            
            if show_train:
                boxplot_data.append(abs_train_errors)
                boxplot_labels.append("Train")
                boxplot_data.append(signed_train_errors)
                boxplot_labels.append("Train (Signed)")
            
            if show_test:
                boxplot_data.append(abs_test_errors)
                boxplot_labels.append("Test")
                boxplot_data.append(signed_test_errors)
                boxplot_labels.append("Test (Signed)")
            
            sns.boxplot(data=boxplot_data, ax=ax)
            ax.set_xticklabels(boxplot_labels)
            ax.set_title("Boxplot of Errors")
            ax.set_ylabel("Error Value")
            
        elif plot_type == "Error Scatterplot vs. Predicted":
            # Scatterplot of errors vs predicted values
            if show_train:
                ax.scatter(y_train_pred, signed_train_errors, 
                        label='Training Set', alpha=0.6)
            
            if show_test:
                ax.scatter(y_test_pred, signed_test_errors, 
                        label='Test Set', alpha=0.6, color='orange')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', label='Zero Error')
            
            ax.set_title("Error vs Predicted Value")
            ax.set_xlabel("Predicted Value")
            ax.set_ylabel("Error (Actual - Predicted)")
        
        # Common plot elements
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Draw the plot
        self.error_canvas.fig.tight_layout()
        self.error_canvas.draw()

    def update_error_statistics(self, abs_train_errors, abs_test_errors, 
                           signed_train_errors, signed_test_errors):
        """Update error statistics text box with summary statistics"""
        self.error_stats_text.clear()
        
        # Training set statistics
        train_mae = abs_train_errors.mean()
        train_rmse = np.sqrt(np.mean(signed_train_errors**2))
        train_median_ae = np.median(abs_train_errors)
        train_max_ae = abs_train_errors.max()
        train_mean_error = signed_train_errors.mean()
        
        # Test set statistics
        test_mae = abs_test_errors.mean()
        test_rmse = np.sqrt(np.mean(signed_test_errors**2))
        test_median_ae = np.median(abs_test_errors)
        test_max_ae = abs_test_errors.max()
        test_mean_error = signed_test_errors.mean()
        
        # Format and display statistics
        stats_text = f"Training Set: MAE = {train_mae:.4f}, RMSE = {train_rmse:.4f}, " \
                    f"Median AE = {train_median_ae:.4f}, Max AE = {train_max_ae:.4f}, " \
                    f"Mean Error = {train_mean_error:.4f}\n\n" \
                    f"Test Set: MAE = {test_mae:.4f}, RMSE = {test_rmse:.4f}, " \
                    f"Median AE = {test_median_ae:.4f}, Max AE = {test_max_ae:.4f}, " \
                    f"Mean Error = {test_mean_error:.4f}"
        
        self.error_stats_text.setText(stats_text)



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
        
        # Add the feature correlation with target option
        self.show_correlations_btn = QPushButton("Show Feature Correlations with Target")
        self.show_correlations_btn.clicked.connect(self.show_feature_correlations)
        right_layout.addWidget(self.show_correlations_btn)
        
        # Add left and right groups to horizontal layout
        h_layout.addWidget(left_group, 60)
        h_layout.addWidget(right_group, 40)
        
        layout.addLayout(h_layout)
        
        self.tabs.addTab(tab, "Feature Selection")
        
        # Connect target variable change to handler
        self.target_var.currentTextChanged.connect(self.target_variable_changed)
    
    def create_model_config_tab(self):
        """Create tab for model configuration and training"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Main configuration form
        form_layout = QFormLayout()
        
        # Model type selection
        self.model_type = QComboBox()
        self.model_type.addItems([
            "LinearRegression", 
            "Ridge", 
            "Lasso", 
            "ElasticNet", 
            "SVR", 
            "RandomForest", 
            "GradientBoosting"
        ])
        form_layout.addRow("Model Type:", self.model_type)
        
        # Test size
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.1, 0.5)
        self.test_size.setValue(0.2)
        self.test_size.setSingleStep(0.05)
        form_layout.addRow("Test Set Size:", self.test_size)
        
        # Random state
        self.random_state = QSpinBox()
        self.random_state.setRange(0, 1000)
        self.random_state.setValue(42)
        form_layout.addRow("Random Seed:", self.random_state)
        
        # Data scaling
        self.scaling_method = QComboBox()
        self.scaling_method.addItems(["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])
        form_layout.addRow("Data Scaling:", self.scaling_method)
        
        # Cross-validation
        self.use_cross_val = QCheckBox("Use cross-validation")
        self.use_cross_val.setChecked(True)
        form_layout.addRow("", self.use_cross_val)
        
        self.cv_folds = QSpinBox()
        self.cv_folds.setRange(2, 10)
        self.cv_folds.setValue(5)
        form_layout.addRow("CV Folds:", self.cv_folds)
        
        # Feature selection
        self.do_feature_selection = QCheckBox("Perform Feature Selection")
        form_layout.addRow("", self.do_feature_selection)
        
        self.feature_selection_method = QComboBox()
        self.feature_selection_method.addItems(["SelectKBest", "RFE"])
        form_layout.addRow("Feature Selection Method:", self.feature_selection_method)
        
        self.num_features = QSpinBox()
        self.num_features.setRange(1, 100)
        self.num_features.setValue(5)
        form_layout.addRow("Number of Features to Select:", self.num_features)
        
        # Add form to layout
        layout.addLayout(form_layout)
        
        # Hyperparameters section
        self.hyperparams_group = QGroupBox("Model Hyperparameters")
        self.hyperparams_layout = QFormLayout()
        self.hyperparams_group.setLayout(self.hyperparams_layout)
        
        layout.addWidget(self.hyperparams_group)
        
        # Training button
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        layout.addWidget(self.train_btn)
        
        self.tabs.addTab(tab, "Model Configuration")
        
        # Connect model type change to hyperparameter UI update
        self.model_type.currentTextChanged.connect(self.update_hyperparameter_ui)
        self.update_hyperparameter_ui()  # Initialize the UI
    
    def create_results_tab(self):
        """Create tab for model results display"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Results summary
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier New", 10))
        
        # Plots section
        plots_layout = QHBoxLayout()
        
        # Actual vs Predicted plot
        left_plot_group = QGroupBox("Actual vs Predicted")
        left_plot_layout = QVBoxLayout()
        self.actual_vs_pred_canvas = MatplotlibCanvas(self, width=4, height=4)
        self.actual_vs_pred_toolbar = NavigationToolbar(self.actual_vs_pred_canvas, self)
        left_plot_layout.addWidget(self.actual_vs_pred_toolbar)
        left_plot_layout.addWidget(self.actual_vs_pred_canvas)
        left_plot_group.setLayout(left_plot_layout)
        
        # Residuals plot
        right_plot_group = QGroupBox("Residuals Analysis")
        right_plot_layout = QVBoxLayout()
        self.residuals_canvas = MatplotlibCanvas(self, width=4, height=4)
        self.residuals_toolbar = NavigationToolbar(self.residuals_canvas, self)
        right_plot_layout.addWidget(self.residuals_toolbar)
        right_plot_layout.addWidget(self.residuals_canvas)
        right_plot_group.setLayout(right_plot_layout)
        
        plots_layout.addWidget(left_plot_group)
        plots_layout.addWidget(right_plot_group)
        
        # Create a splitter for results text and plots
        splitter = QSplitter(Qt.Vertical)
        
        # Add results text to a container
        text_container = QWidget()
        text_layout = QVBoxLayout()
        text_layout.addWidget(QLabel("Model Results Summary:"))
        text_layout.addWidget(self.results_text)
        text_container.setLayout(text_layout)
        
        # Add plots to a container
        plots_container = QWidget()
        plots_container.setLayout(plots_layout)
        
        splitter.addWidget(text_container)
        splitter.addWidget(plots_container)
        
        layout.addWidget(splitter)
        
        # Feature importance plot
        feature_importance_group = QGroupBox("Feature Importance")
        feature_importance_layout = QVBoxLayout()
        self.feature_importance_canvas = MatplotlibCanvas(self, width=5, height=3)
        self.feature_importance_toolbar = NavigationToolbar(self.feature_importance_canvas, self)
        feature_importance_layout.addWidget(self.feature_importance_toolbar)
        feature_importance_layout.addWidget(self.feature_importance_canvas)
        feature_importance_group.setLayout(feature_importance_layout)
        
        layout.addWidget(feature_importance_group)
        
        # Save results buttons
        btn_layout = QHBoxLayout()
        
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        btn_layout.addWidget(self.save_model_btn)
        
        self.save_plots_btn = QPushButton("Save Plots")
        self.save_plots_btn.clicked.connect(self.save_results_plots)
        btn_layout.addWidget(self.save_plots_btn)
        
        layout.addLayout(btn_layout)
        
        self.tabs.addTab(tab, "Model Results")
    
    def create_predictions_tab(self):
        """Create tab for making predictions with the trained model"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Instructions
        layout.addWidget(QLabel("Enter values for each feature to make predictions:"))
        
        # Input form for feature values
        self.prediction_form = QFormLayout()
        self.prediction_inputs = {}  # Will store input widgets
        
        # Create a scrollable area for the form
        form_widget = QWidget()
        form_widget.setLayout(self.prediction_form)
        
        scroll_area = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.addWidget(form_widget)
        scroll_layout.addStretch()
        scroll_area.setLayout(scroll_layout)
        
        layout.addWidget(scroll_area)
        
        # Prediction output
        result_group = QGroupBox("Prediction Result")
        result_layout = QVBoxLayout()
        self.prediction_result = QLabel("No prediction yet")
        self.prediction_result.setFont(QFont("Arial", 12, QFont.Bold))
        self.prediction_result.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.prediction_result)
        
        # Add visualization for the prediction
        self.prediction_canvas = MatplotlibCanvas(self, width=5, height=3)
        result_layout.addWidget(self.prediction_canvas)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.predict_btn = QPushButton("Make Prediction")
        self.predict_btn.clicked.connect(self.make_prediction)
        controls_layout.addWidget(self.predict_btn)
        
        self.clear_inputs_btn = QPushButton("Clear Inputs")
        self.clear_inputs_btn.clicked.connect(self.clear_prediction_inputs)
        controls_layout.addWidget(self.clear_inputs_btn)
        
        layout.addLayout(controls_layout)
        
        self.tabs.addTab(tab, "Predictions")
    
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
        
        # Enable tabs and buttons
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
    
    def show_feature_correlations(self):
        """Show correlations between features and target variable"""
        target_var = self.target_var.currentText()
        
        if not target_var or self.df is None:
            QMessageBox.warning(self, "Error", "No target variable selected")
            return
        
        if not pd.api.types.is_numeric_dtype(self.df[target_var]):
            QMessageBox.warning(self, "Error", "Target variable must be numeric for correlation analysis")
            return
        
        # Get numeric columns only
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if target_var not in numeric_df.columns:
            QMessageBox.warning(self, "Error", "Target variable must be numeric for correlation analysis")
            return
        
        # Create a dialog with correlation analysis
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Feature Correlations with {target_var}")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        
        # Create a matplotlib canvas for the correlation plot
        canvas = MatplotlibCanvas(width=8, height=6)
        toolbar = NavigationToolbar(canvas, dialog)
        
        # Create the correlation plot
        canvas.fig.clear()
        
        # Calculate correlations with target
        correlations = numeric_df.corr()[target_var].sort_values(ascending=False)
        correlations = correlations.drop(target_var)  # Remove self-correlation
        
        # Plot correlations
        ax = canvas.fig.add_subplot(111)
        bars = correlations.plot(kind='barh', ax=ax, color=correlations.map(lambda x: 'green' if x > 0 else 'red'))
        
        # Customize plot
        ax.set_title(f"Feature Correlations with {target_var}")
        ax.set_xlabel("Correlation Coefficient")
        ax.set_ylabel("Features")
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add correlation values as text
        for i, v in enumerate(correlations):
            ax.text(v + (0.01 if v >= 0 else -0.01), 
                    i, 
                    f"{v:.3f}", 
                    va='center', 
                    ha='left' if v >= 0 else 'right',
                    fontweight='bold')
        
        canvas.fig.tight_layout()
        canvas.draw()
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        # Add a text area with the correlation values
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setFont(QFont("Courier New", 10))
        text_area.setMaximumHeight(200)
        
        text_area.append(f"Correlations with {target_var}:\n")
        for feature, corr in correlations.items():
            text_area.append(f"{feature}: {corr:.6f}")
        
        layout.addWidget(text_area)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def update_hyperparameter_ui(self):
        """Update hyperparameter UI based on selected model type"""
        # Clear existing hyperparameters
        for i in reversed(range(self.hyperparams_layout.count())): 
            self.hyperparams_layout.itemAt(i).widget().setParent(None)
        
        model_type = self.model_type.currentText()
        
        if model_type == "LinearRegression":
            # Linear Regression has no hyperparameters to tune
            self.hyperparams_layout.addRow(QLabel("Linear Regression has no hyperparameters to tune."))
        
        elif model_type == "Ridge":
            # Ridge regression hyperparameters
            alpha_spin = QDoubleSpinBox()
            alpha_spin.setRange(0.0001, 100.0)
            alpha_spin.setValue(1.0)
            alpha_spin.setSingleStep(0.1)
            alpha_spin.setDecimals(4)
            self.hyperparams_layout.addRow("Alpha:", alpha_spin)
        
        elif model_type == "Lasso":
            # Lasso regression hyperparameters
            alpha_spin = QDoubleSpinBox()
            alpha_spin.setRange(0.0001, 100.0)
            alpha_spin.setValue(1.0)
            alpha_spin.setSingleStep(0.1)
            alpha_spin.setDecimals(4)
            self.hyperparams_layout.addRow("Alpha:", alpha_spin)
        
        elif model_type == "ElasticNet":
            # ElasticNet hyperparameters
            alpha_spin = QDoubleSpinBox()
            alpha_spin.setRange(0.0001, 100.0)
            alpha_spin.setValue(1.0)
            alpha_spin.setSingleStep(0.1)
            alpha_spin.setDecimals(4)
            self.hyperparams_layout.addRow("Alpha:", alpha_spin)
            
            l1_ratio_spin = QDoubleSpinBox()
            l1_ratio_spin.setRange(0.0, 1.0)
            l1_ratio_spin.setValue(0.5)
            l1_ratio_spin.setSingleStep(0.1)
            self.hyperparams_layout.addRow("L1 Ratio:", l1_ratio_spin)
        
        elif model_type == "SVR":
            # SVR hyperparameters
            C_spin = QDoubleSpinBox()
            C_spin.setRange(0.1, 100.0)
            C_spin.setValue(1.0)
            C_spin.setSingleStep(0.1)
            self.hyperparams_layout.addRow("C:", C_spin)
            
            epsilon_spin = QDoubleSpinBox()
            epsilon_spin.setRange(0.01, 1.0)
            epsilon_spin.setValue(0.1)
            epsilon_spin.setSingleStep(0.01)
            self.hyperparams_layout.addRow("Epsilon:", epsilon_spin)
            
            kernel_combo = QComboBox()
            kernel_combo.addItems(["linear", "poly", "rbf", "sigmoid"])
            kernel_combo.setCurrentText("rbf")
            self.hyperparams_layout.addRow("Kernel:", kernel_combo)
        
        elif model_type == "RandomForest":
            # Random Forest hyperparameters
            n_estimators_spin = QSpinBox()
            n_estimators_spin.setRange(10, 500)
            n_estimators_spin.setValue(100)
            n_estimators_spin.setSingleStep(10)
            self.hyperparams_layout.addRow("Number of Trees:", n_estimators_spin)
            
            max_depth_spin = QSpinBox()
            max_depth_spin.setRange(1, 50)
            max_depth_spin.setValue(10)
            max_depth_spin.setSingleStep(1)
            self.hyperparams_layout.addRow("Max Depth:", max_depth_spin)
            
            min_samples_split_spin = QSpinBox()
            min_samples_split_spin.setRange(2, 20)
            min_samples_split_spin.setValue(2)
            min_samples_split_spin.setSingleStep(1)
            self.hyperparams_layout.addRow("Min Samples Split:", min_samples_split_spin)
        
        elif model_type == "GradientBoosting":
            # Gradient Boosting hyperparameters
            n_estimators_spin = QSpinBox()
            n_estimators_spin.setRange(10, 500)
            n_estimators_spin.setValue(100)
            n_estimators_spin.setSingleStep(10)
            self.hyperparams_layout.addRow("Number of Trees:", n_estimators_spin)
            
            learning_rate_spin = QDoubleSpinBox()
            learning_rate_spin.setRange(0.001, 1.0)
            learning_rate_spin.setValue(0.1)
            learning_rate_spin.setSingleStep(0.01)
            learning_rate_spin.setDecimals(3)
            self.hyperparams_layout.addRow("Learning Rate:", learning_rate_spin)
            
            max_depth_spin = QSpinBox()
            max_depth_spin.setRange(1, 20)
            max_depth_spin.setValue(3)
            max_depth_spin.setSingleStep(1)
            self.hyperparams_layout.addRow("Max Depth:", max_depth_spin)
    
    def get_hyperparameters(self):
        """Extract hyperparameters from the UI based on the selected model"""
        model_type = self.model_type.currentText()
        hyperparams = {}
        
        # Get all widgets in the hyperparameters layout
        for i in range(self.hyperparams_layout.rowCount()):
            label_item = self.hyperparams_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = self.hyperparams_layout.itemAt(i, QFormLayout.FieldRole)
            
            if label_item is not None and field_item is not None:
                label_widget = label_item.widget()
                field_widget = field_item.widget()
                
                if isinstance(label_widget, QLabel) and label_widget.text() != "Linear Regression has no hyperparameters to tune.":
                    param_name = label_widget.text().replace(":", "").strip().lower()
                    
                    if isinstance(field_widget, QSpinBox) or isinstance(field_widget, QDoubleSpinBox):
                        param_value = field_widget.value()
                    elif isinstance(field_widget, QComboBox):
                        param_value = field_widget.currentText()
                    else:
                        continue
                    
                    # Map UI parameter names to sklearn parameter names
                    if param_name == "number of trees":
                        hyperparams["n_estimators"] = param_value
                    elif param_name == "learning rate":
                        hyperparams["learning_rate"] = param_value
                    elif param_name == "max depth":
                        if param_value <= 0:  # Handle "None" case
                            hyperparams["max_depth"] = None
                        else:
                            hyperparams["max_depth"] = param_value
                    elif param_name == "min samples split":
                        hyperparams["min_samples_split"] = param_value
                    elif param_name == "l1 ratio":
                        hyperparams["l1_ratio"] = param_value
                    else:
                        # For other parameters, use the name directly
                        hyperparams[param_name] = param_value
        
        return hyperparams
    
    def get_selected_features(self):
        """Get the list of selected input features"""
        selected_features = []
        
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_features.append(item.text())
        
        return selected_features
    
    def train_model(self):
        """Train the selected machine learning model"""
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
        
        # Get model parameters
        model_type = self.model_type.currentText()
        test_size = self.test_size.value()
        random_state = self.random_state.value()
        scaling_method = self.scaling_method.currentText()
        use_cross_val = self.use_cross_val.isChecked()
        cv_folds = self.cv_folds.value()
        do_feature_selection = self.do_feature_selection.isChecked()
        feature_selection_method = self.feature_selection_method.currentText()
        num_features = self.num_features.value()
        
        # Get hyperparameters
        hyperparameters = self.get_hyperparameters()
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Disable UI during training
        self.tabs.setEnabled(False)
        self.train_btn.setEnabled(False)
        
        # Start model training thread
        self.model_trainer = ModelTrainer(
            self.df, input_features, target_variable, model_type, 
            test_size, random_state, scaling_method, hyperparameters, 
            use_cross_val, cv_folds, do_feature_selection, feature_selection_method, 
            num_features
        )
        self.model_trainer.progress_update.connect(self.update_progress)
        self.model_trainer.status_update.connect(self.update_status)
        self.model_trainer.error_raised.connect(self.show_error)
        self.model_trainer.training_finished.connect(self.model_trained)
        self.model_trainer.start()

    def create_test_size_sensitivity_tab(self):
        """Create tab for analyzing effect of test size on model performance"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Button to start the analysis
        self.run_sensitivity_btn = QPushButton("Run Test Size Sensitivity Analysis")
        self.run_sensitivity_btn.clicked.connect(self.run_test_size_sensitivity)
        layout.addWidget(self.run_sensitivity_btn)
        
        # Matplotlib canvas for plot
        self.test_size_canvas = MatplotlibCanvas(self, width=6, height=5)
        self.test_size_toolbar = NavigationToolbar(self.test_size_canvas, self)
        layout.addWidget(self.test_size_toolbar)
        layout.addWidget(self.test_size_canvas)
        
        # Text box to show optimal test size info
        self.test_size_info = QTextEdit()
        self.test_size_info.setReadOnly(True)
        self.test_size_info.setMaximumHeight(120)
        layout.addWidget(self.test_size_info)
        
        self.tabs.addTab(tab, "Test Size Sensitivity")
    def run_test_size_sensitivity(self):
        """Run test size sensitivity analysis"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        input_features = self.get_selected_features()
        target_variable = self.target_var.currentText()
        
        if not input_features or not target_variable:
            QMessageBox.warning(self, "Error", "Please select input features and target variable first")
            return
        
        if target_variable in input_features:
            QMessageBox.warning(self, "Error", "Target variable cannot be one of the input features")
            return
        
        # Prepare data
        X = self.df[input_features].fillna(self.df[input_features].mean())
        y = self.df[target_variable].fillna(self.df[target_variable].mean())
        
        scaling = self.scaling_method.currentText()
        scaler = None
        if scaling == "StandardScaler":
            scaler = StandardScaler()
        elif scaling == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaling == "RobustScaler":
            scaler = RobustScaler()
        
        if scaler:
            X = scaler.fit_transform(X)
        else:
            X = X.values
        
        # Ranges of test sizes
        test_sizes = np.arange(0.1, 0.91, 0.02)
        train_r2_scores = []
        test_r2_scores = []
        
        model_type = self.model_type.currentText()
        
        # outside the loop
        if model_type == "LinearRegression":
            model_template = LinearRegression()
        elif model_type == "Ridge":
            model_template = Ridge()
        elif model_type == "Lasso":
            model_template = Lasso()
        elif model_type == "ElasticNet":
            model_template = ElasticNet()
        elif model_type == "SVR":
            model_template = SVR()
        elif model_type == "RandomForest":
            model_template = RandomForestRegressor()
        elif model_type == "GradientBoosting":
            model_template = GradientBoostingRegressor()
        else:
            model_template = LinearRegression()

        # inside the loop
        for ts in test_sizes:
            model = clone(model_template)  # <- fast copy
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=self.random_state.value())
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            train_r2_scores.append(train_r2)
            test_r2_scores.append(test_r2)
        
        # Find optimal test size (smallest gap between train and test r2)
        gaps = np.abs(np.array(train_r2_scores) - np.array(test_r2_scores))
        optimal_idx = np.argmin(gaps)
        optimal_test_size = test_sizes[optimal_idx]
        optimal_train_r2 = train_r2_scores[optimal_idx]
        optimal_test_r2 = test_r2_scores[optimal_idx]
        
        # Plot
        self.test_size_canvas.fig.clear()
        ax = self.test_size_canvas.fig.add_subplot(111)
        ax.plot(test_sizes, train_r2_scores, label="Train R²", marker='o')
        ax.plot(test_sizes, test_r2_scores, label="Test R²", marker='o')
        
        # Mark optimal point
        ax.axvline(x=optimal_test_size, color='red', linestyle='--', label=f"Optimal Test Size = {optimal_test_size:.2f}")
        ax.scatter(optimal_test_size, optimal_train_r2, color='green', s=50, label=f"Train R²: {optimal_train_r2:.3f}")
        ax.scatter(optimal_test_size, optimal_test_r2, color='blue', s=50, label=f"Test R²: {optimal_test_r2:.3f}")
        
        ax.set_xlabel("Test Size")
        ax.set_ylabel("R² Score")
        ax.set_title("Train/Test R² vs Test Size")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        self.test_size_canvas.fig.tight_layout()
        self.test_size_canvas.draw()
        
        # Update info text
        self.test_size_info.setPlainText(
            f"Optimal Test Size: {optimal_test_size:.3f}\n"
            f"Train R² at Optimal: {optimal_train_r2:.4f}\n"
            f"Test R² at Optimal: {optimal_test_r2:.4f}\n"
            f"Gap: {gaps[optimal_idx]:.4f}"
        )

    def model_trained(self, results):
        """Called when model training is complete"""
        self.model_results = results
        self.progress_bar.setVisible(False)
        
        # Enable UI
        self.tabs.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.save_model_btn.setEnabled(True)
        self.save_plots_btn.setEnabled(True)
        
        # Switch to results tab
        self.tabs.setCurrentIndex(3)  # Results tab
        
        # Update results display
        self.update_results_display()
        
        # Update prediction tab
        self.update_prediction_inputs()
        
        self.statusBar().showMessage(f"Model training complete! Model type: {results['model_type']}")
    
    def update_results_display(self):
        """Update the results display with model performance metrics"""
        if not self.model_results:
            return
        
        # Update text summary
        self.results_text.clear()
        self.results_text.append(f"Model Training Results\n{'='*50}\n")
        
        # Model info
        self.results_text.append(f"Model Type: {self.model_results['model_type']}")
        
        # Selected features
        self.results_text.append(f"\nSelected Features ({len(self.model_results['selected_features'])}):")
        for feature in self.model_results['selected_features']:
            self.results_text.append(f"- {feature}")
        
        # Performance metrics
        self.results_text.append("\nTraining Set Metrics:")
        train_metrics = self.model_results['metrics']['train']
        self.results_text.append(f"- MSE: {train_metrics['mse']:.6f}")
        self.results_text.append(f"- RMSE: {train_metrics['rmse']:.6f}")
        self.results_text.append(f"- MAE: {train_metrics['mae']:.6f}")
        self.results_text.append(f"- R²: {train_metrics['r2']:.6f}")
        self.results_text.append(f"- Explained Variance: {train_metrics['explained_variance']:.6f}")
        self.results_text.append(f"- Total Signed Train Error: {train_metrics['total_signed_error']:.6f}")

        self.results_text.append("\nTest Set Metrics:")
        test_metrics = self.model_results['metrics']['test']
        self.results_text.append(f"- MSE: {test_metrics['mse']:.6f}")
        self.results_text.append(f"- RMSE: {test_metrics['rmse']:.6f}")
        self.results_text.append(f"- MAE: {test_metrics['mae']:.6f}")
        self.results_text.append(f"- R²: {test_metrics['r2']:.6f}")
        self.results_text.append(f"- Explained Variance: {test_metrics['explained_variance']:.6f}")
        self.results_text.append(f"- Total Signed Test Error: {test_metrics['total_signed_error']:.6f}")

        # Cross-validation results if available
        if 'cv_scores' in self.model_results:
            cv_scores = self.model_results['cv_scores']
            self.results_text.append(f"\nCross-Validation R² Scores ({len(cv_scores['scores'])} folds):")
            for i, score in enumerate(cv_scores['scores'], 1):
                self.results_text.append(f"- Fold {i}: {score:.6f}")
            self.results_text.append(f"- Mean R²: {cv_scores['mean']:.6f}")
            self.results_text.append(f"- Std Dev: {cv_scores['std']:.6f}")
        
        # Update plots
        self.update_results_plots()
    
    def update_results_plots(self):
        """Update visualization plots for model results"""
        if not self.model_results:
            return
        
        # Get data
        y_train = self.model_results['y_train']
        y_train_pred = self.model_results['predictions']['y_train_pred']
        y_test = self.model_results['y_test']
        y_test_pred = self.model_results['predictions']['y_test_pred']
        
        # Actual vs Predicted plot
        self.actual_vs_pred_canvas.fig.clear()
        ax1 = self.actual_vs_pred_canvas.fig.add_subplot(111)
        
        # Plot training and test data points
        ax1.scatter(y_train, y_train_pred, alpha=0.5, label='Training data')
        ax1.scatter(y_test, y_test_pred, alpha=0.5, label='Test data')
        
        # Add perfect prediction line
        min_val = min(min(y_train), min(y_test))
        max_val = max(max(y_train), max(y_test))
        margin = (max_val - min_val) * 0.05  # 5% margin
        ax1.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
                'k--', label='Perfect prediction')
        
        # Add R² annotation
        train_r2 = self.model_results['metrics']['train']['r2']
        test_r2 = self.model_results['metrics']['test']['r2']
        ax1.text(0.05, 0.95, f'Training R²: {train_r2:.4f}\nTest R²: {test_r2:.4f}', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Customize plot
        ax1.set_title('Actual vs Predicted Values')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.legend(loc='lower right')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Equal aspect ratio
        ax1.set_aspect('equal', adjustable='box')
        
        self.actual_vs_pred_canvas.fig.tight_layout()
        self.actual_vs_pred_canvas.draw()
        
        # Residuals plot
        self.residuals_canvas.fig.clear()
        
        # Create 2x1 subplot grid for residuals
        ax2a = self.residuals_canvas.fig.add_subplot(211)
        ax2b = self.residuals_canvas.fig.add_subplot(212)
        
        # Calculate residuals
        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred
        
        # Scatter plot of residuals
        ax2a.scatter(y_train_pred, train_residuals, alpha=0.5, label='Training')
        ax2a.scatter(y_test_pred, test_residuals, alpha=0.5, label='Test')
        ax2a.axhline(y=0, color='k', linestyle='--')
        ax2a.set_title('Residuals vs Predicted Values')
        ax2a.set_xlabel('Predicted Values')
        ax2a.set_ylabel('Residuals')
        ax2a.legend(loc='best')
        ax2a.grid(True, linestyle='--', alpha=0.7)
        
        # Histogram of residuals
        combined_residuals = np.concatenate([train_residuals, test_residuals])
        ax2b.hist(train_residuals, bins=20, alpha=0.5, label='Training')
        ax2b.hist(test_residuals, bins=20, alpha=0.5, label='Test')
        ax2b.axvline(x=0, color='k', linestyle='--')
        ax2b.set_title('Distribution of Residuals')
        ax2b.set_xlabel('Residuals')
        ax2b.set_ylabel('Frequency')
        ax2b.legend(loc='best')
        
        self.residuals_canvas.fig.tight_layout()
        self.residuals_canvas.draw()
        
        # Feature importance plot
        self.feature_importance_canvas.fig.clear()
        ax3 = self.feature_importance_canvas.fig.add_subplot(111)

        if 'feature_importances' in self.model_results:
            fi = self.model_results['feature_importances']
            
            # Check format
            if isinstance(fi, dict):
                features = fi['features']
                importances = fi['importances']
            else:
                features = self.model_results['selected_features']
                importances = fi
            
            # Determine if it's coefficients or feature importances
            is_coefficients = False
            if self.model_results['model_type'] in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
                is_coefficients = True

            # Sort
            sorted_idx = np.argsort(importances)
            features_sorted = [features[i] for i in sorted_idx]
            importances_sorted = importances[sorted_idx]
            
            y_pos = np.arange(len(features_sorted))
            ax3.barh(y_pos, importances_sorted, align='center')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(features_sorted)
            
            # Set dynamic title and label
            if is_coefficients:
                ax3.set_title('Feature Coefficients')
                ax3.set_xlabel('Coefficient Value')
            else:
                ax3.set_title('Feature Importances')
                ax3.set_xlabel('Importance Value')
            
            # Add value labels
            for i, v in enumerate(importances_sorted):
                ax3.text(v + (0.01 if v > 0 else -0.01), i, f"{v:.4f}", 
                        va='center', ha='left' if v > 0 else 'right')
        else:
            ax3.text(0.5, 0.5, 'Feature importance not available for this model', 
                    ha='center', va='center', transform=ax3.transAxes)

        self.feature_importance_canvas.fig.tight_layout()
        self.feature_importance_canvas.draw()

        
    def update_prediction_inputs(self):
        """Update the prediction form with input fields for each feature"""
        if not self.model_results:
            return
        
        # Clear existing form
        for i in reversed(range(self.prediction_form.count())): 
            self.prediction_form.itemAt(i).widget().setParent(None)
        
        self.prediction_inputs = {}
        
        # Add input field for each feature
        for feature in self.model_results['selected_features']:
            # Create input field
            input_field = QDoubleSpinBox()
            input_field.setRange(-1000000, 1000000)
            input_field.setDecimals(4)
            
            # Set default value from data mean if available
            if self.df is not None and feature in self.df.columns:
                mean_val = self.df[feature].mean()
                input_field.setValue(mean_val)
                
                # Add tooltip with feature statistics
                stats = self.df[feature].describe()
                tooltip = f"Min: {stats['min']:.4f}\n"
                tooltip += f"Mean: {stats['mean']:.4f}\n"
                tooltip += f"Max: {stats['max']:.4f}"
                input_field.setToolTip(tooltip)
            
            self.prediction_inputs[feature] = input_field
            self.prediction_form.addRow(f"{feature}:", input_field)
    
    def make_prediction(self):
        """Make a prediction using the trained model"""
        if not self.model_results or 'model' not in self.model_results:
            QMessageBox.warning(self, "Error", "No trained model available")
            return
        
        try:
            # Get input values
            input_data = {}
            for feature, input_field in self.prediction_inputs.items():
                input_data[feature] = input_field.value()
            
            # Create a DataFrame with the input values
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            model = self.model_results['model']
            
            # Apply the same preprocessing as during training
            X = input_df.values
            
            # Display the prediction
            prediction = model.predict(X)[0]
            
            self.prediction_result.setText(f"Prediction: {prediction:.4f}")
            
            # Update the prediction visualization
            self.update_prediction_visualization(prediction, input_data)
            
        except Exception as e:
            QMessageBox.warning(self, "Prediction Error", f"Error making prediction: {str(e)}")
    
    def update_prediction_visualization(self, prediction, input_data):
        """Update the prediction visualization"""
        if not self.model_results:
            return
        
        # Clear the canvas
        self.prediction_canvas.fig.clear()
        ax = self.prediction_canvas.fig.add_subplot(111)
        
        # Get training and test values to show context
        y_train = self.model_results['y_train']
        y_test = self.model_results['y_test']
        
        # Create histogram of target values
        all_values = pd.concat([y_train, y_test])
        
        # Create histogram
        sns.histplot(all_values, kde=True, ax=ax)
        
        # Add vertical line for prediction
        ax.axvline(x=prediction, color='red', linestyle='--', linewidth=2, 
                   label=f'Prediction: {prediction:.4f}')
        
        # Add percentile annotation
        percentile = stats.percentileofscore(all_values, prediction)
        ax.text(0.98, 0.95, f'Percentile: {percentile:.1f}%', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Customize plot
        ax.set_title('Prediction in Context of Target Distribution')
        ax.set_xlabel('Target Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        self.prediction_canvas.fig.tight_layout()
        self.prediction_canvas.draw()
    
    def clear_prediction_inputs(self):
        """Clear all prediction input fields"""
        for input_field in self.prediction_inputs.values():
            input_field.clear()
        
        self.prediction_result.setText("No prediction yet")
        
        # Clear the canvas
        self.prediction_canvas.fig.clear()
        self.prediction_canvas.draw()
    
    def save_model(self):
        """Save the trained model to a file"""
        if not self.model_results or 'model' not in self.model_results:
            QMessageBox.warning(self, "Error", "No trained model available")
            return
        
        try:
            # Create a save dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Model", "", 
                "Pickle Files (*.pkl);;All Files (*)"
            )
            
            if not file_path:
                return
            
            if not file_path.endswith('.pkl'):
                file_path += '.pkl'
            
            # Save the model using pickle
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(self.model_results['model'], f)
            
            QMessageBox.information(self, "Success", f"Model saved to {file_path}")
            
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Error saving model: {str(e)}")
    
    def save_results_plots(self):
        """Save the result plots to files"""
        if not self.model_results:
            QMessageBox.warning(self, "Error", "No model results available")
            return
        
        try:
            # Get directory path
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Plots"
            )
            
            if not dir_path:
                return
                
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save actual vs predicted plot
            actual_pred_path = os.path.join(dir_path, f"actual_vs_pred_{timestamp}.png")
            self.actual_vs_pred_canvas.fig.savefig(actual_pred_path, dpi=300, bbox_inches='tight')
            
            # Save residuals plot
            residuals_path = os.path.join(dir_path, f"residuals_{timestamp}.png")
            self.residuals_canvas.fig.savefig(residuals_path, dpi=300, bbox_inches='tight')
            
            # Save feature importance plot
            importance_path = os.path.join(dir_path, f"feature_importance_{timestamp}.png")
            self.feature_importance_canvas.fig.savefig(importance_path, dpi=300, bbox_inches='tight')
            
            QMessageBox.information(
                self, "Success", 
                f"Plots saved to:\n- {actual_pred_path}\n- {residuals_path}\n- {importance_path}"
            )
            
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Error saving plots: {str(e)}")
    
    def save_results(self):
        """Save comprehensive results to a directory"""
        if not self.model_results:
            QMessageBox.warning(self, "Error", "No model results available")
            return
        
        try:
            # Get directory path
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Results"
            )
            
            if not dir_path:
                return
                
            # Create timestamped folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join(dir_path, f"ML_Results_{timestamp}")
            
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Save model
            model_path = os.path.join(results_dir, "model.pkl")
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(self.model_results['model'], f)
            
            # Save metrics as CSV
            metrics_path = os.path.join(results_dir, "metrics.csv")
            
            # Combine metrics into a DataFrame
            train_metrics = self.model_results['metrics']['train']
            test_metrics = self.model_results['metrics']['test']
            
            metrics_df = pd.DataFrame({
                'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'Explained Variance'],
                'Training': [
                    train_metrics['mse'],
                    train_metrics['rmse'],
                    train_metrics['mae'],
                    train_metrics['r2'],
                    train_metrics['explained_variance']
                ],
                'Test': [
                    test_metrics['mse'],
                    test_metrics['rmse'],
                    test_metrics['mae'],
                    test_metrics['r2'],
                    test_metrics['explained_variance']
                ]
            })
            
            metrics_df.to_csv(metrics_path, index=False)
            
            # Save cross-validation results if available
            if 'cv_scores' in self.model_results:
                cv_path = os.path.join(results_dir, "cross_validation.csv")
                cv_scores = self.model_results['cv_scores']['scores']
                
                cv_df = pd.DataFrame({
                    'Fold': range(1, len(cv_scores) + 1),
                    'R²': cv_scores
                })
                
                cv_df.to_csv(cv_path, index=False)
            
            # Save feature importance if available
            if 'feature_importances' in self.model_results:
                fi_path = os.path.join(results_dir, "feature_importance.csv")
                
                fi = self.model_results['feature_importances']
                
                # Check format of feature importances
                if isinstance(fi, dict):
                    features = fi['features']
                    importances = fi['importances']
                else:
                    features = self.model_results['selected_features']
                    importances = fi
                
                fi_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                })
                
                fi_df.to_csv(fi_path, index=False)
            
            # Save plots
            plots_dir = os.path.join(results_dir, "plots")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
                
            # Save actual vs predicted plot
            self.actual_vs_pred_canvas.fig.savefig(
                os.path.join(plots_dir, "actual_vs_pred.png"), 
                dpi=300, bbox_inches='tight'
            )
            
            # Save residuals plot
            self.residuals_canvas.fig.savefig(
                os.path.join(plots_dir, "residuals.png"), 
                dpi=300, bbox_inches='tight'
            )
            
            # Save feature importance plot
            self.feature_importance_canvas.fig.savefig(
                os.path.join(plots_dir, "feature_importance.png"), 
                dpi=300, bbox_inches='tight'
            )
            
            # Save full report
            report_path = os.path.join(results_dir, "model_report.txt")
            with open(report_path, 'w') as f:
                f.write(self.results_text.toPlainText())
            
            QMessageBox.information(self, "Success", f"All results saved to {results_dir}")
            
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Error saving results: {str(e)}")
            

def main():
    app = QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()