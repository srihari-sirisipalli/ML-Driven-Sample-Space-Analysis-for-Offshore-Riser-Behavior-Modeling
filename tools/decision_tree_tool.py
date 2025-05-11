"""
Decision Tree Analysis Tool - A comprehensive application for analyzing decision trees
with enhanced visualization and error analysis capabilities.

The application includes:
- Data loading and preprocessing
- Feature selection and target analysis
- Decision tree training and configuration
- Interactive tree visualization
- Detailed error analysis
- Node-by-node analysis
- Tree pruning capabilities
- Export options for saving results
"""

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
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.base import clone

# GUI Libraries
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QTabWidget, 
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QTableWidget, QTableWidgetItem, QComboBox, 
    QSplitter, QTextEdit, QMessageBox, QProgressBar,
    QSizePolicy, QListWidget, QAbstractItemView,
    QCheckBox, QGroupBox, QRadioButton, QSpinBox, 
    QDoubleSpinBox, QFormLayout, QGridLayout, QFrame,
    QListWidgetItem, QDialog, QHeaderView, QSlider,
    QScrollArea, QToolButton, QMenu, QAction, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor, QBrush, QPalette, QIcon

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# PDF Export (optional)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, ListFlowable, ListItem
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from io import BytesIO
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

#------------------------------------------------------------------------------
# Worker Thread Classes
#------------------------------------------------------------------------------

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
                    if df_test.columns.str.contains('Unnamed').any() or df_test.shape[1] < 5:
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
            
            # Identify features and potential target variables
            self.progress_update.emit(70)
            self.status_update.emit("Processing complete!")
            self.progress_update.emit(100)
            
            # Emit result
            self.loading_finished.emit(self.df)
            
        except Exception as e:
            self.error_raised.emit(f"Error processing file: {str(e)}")


class TreeTrainer(QThread):
    """Worker thread to handle decision tree training and analysis"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    error_raised = pyqtSignal(str)
    training_finished = pyqtSignal(object)
    
    def __init__(self, df, input_features, target_variable, task_type, test_size, random_state, 
                 tree_params, scaling_method, cross_validation, cv_folds, max_important_nodes=20, parent=None):
        super().__init__(parent)
        self.df = df
        self.input_features = input_features
        self.target_variable = target_variable
        self.task_type = task_type  # 'classification' or 'regression'
        self.test_size = test_size
        self.random_state = random_state
        self.tree_params = tree_params
        self.scaling_method = scaling_method
        self.cross_validation = cross_validation
        self.cv_folds = cv_folds
        self.max_important_nodes = max_important_nodes
        
    def run(self):
        try:
            results = {}
            
            self.status_update.emit("Preparing data...")
            self.progress_update.emit(10)
            
            # Prepare X and y
            X = self.df[self.input_features].copy()
            y = self.df[self.target_variable].copy()
            
            # Handle missing values
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
            if self.task_type == 'regression':
                y = y.fillna(y.mean())
            else:
                # For classification, fill with mode
                y = y.fillna(y.mode()[0])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state,
                stratify=y if self.task_type == 'classification' else None
            )
            
            results['raw_data'] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            self.status_update.emit("Scaling features...")
            self.progress_update.emit(25)
            
            # Feature scaling
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
            
            results['scaled_data'] = {
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled
            }
            
            # Initialize decision tree model
            self.status_update.emit("Training decision tree...")
            self.progress_update.emit(40)
            
            if self.task_type == 'regression':
                tree_model = DecisionTreeRegressor(**self.tree_params)
            else:
                tree_model = DecisionTreeClassifier(**self.tree_params)
            
            # Train the model
            tree_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            self.status_update.emit("Making predictions...")
            self.progress_update.emit(60)
            
            y_train_pred = tree_model.predict(X_train_scaled)
            y_test_pred = tree_model.predict(X_test_scaled)
            
            # Cross-validation if requested
            if self.cross_validation:
                self.status_update.emit("Performing cross-validation...")
                self.progress_update.emit(75)
                
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                if self.task_type == 'regression':
                    cv_scores = cross_val_score(tree_model, X_train_scaled, y_train, cv=kf, scoring='r2')
                else:
                    cv_scores = cross_val_score(tree_model, X_train_scaled, y_train, cv=kf, scoring='f1_weighted')
                
                results['cv_results'] = {
                    'scores': cv_scores,
                    'mean': np.mean(cv_scores),
                    'std': np.std(cv_scores)
                }
            
            # Calculate metrics
            self.status_update.emit("Calculating metrics...")
            self.progress_update.emit(85)
            
            results['predictions'] = {
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred
            }
            
            if self.task_type == 'regression':
                metrics = self._calculate_regression_metrics(y_train, y_train_pred, y_test, y_test_pred)
            else:
                metrics = self._calculate_classification_metrics(y_train, y_train_pred, y_test, y_test_pred)
            
            results['metrics'] = metrics
            
            # Extract tree information
            self.status_update.emit("Extracting tree information...")
            self.progress_update.emit(95)
            
            # Get feature importances
            results['feature_importance'] = {
                'features': self.input_features,
                'importance': tree_model.feature_importances_
            }
            
            # Get tree structure
            results['tree_structure'] = {
                'model': tree_model,
                'feature_names': self.input_features,
                'target_name': self.target_variable,
                'num_nodes': tree_model.tree_.node_count,
                'max_depth': tree_model.tree_.max_depth,
                'num_leaves': tree_model.get_n_leaves(),
                'tree_text': export_text(tree_model, feature_names=list(self.input_features))
            }
            
            # Save task type and other parameters
            results['parameters'] = {
                'task_type': self.task_type,
                'test_size': self.test_size,
                'random_state': self.random_state,
                'tree_params': self.tree_params,
                'scaling_method': self.scaling_method,
                'max_important_nodes': self.max_important_nodes
            }

            # Generate pruned trees (Cost Complexity Pruning)
            self.status_update.emit("Generating pruned trees...")
            results['pruning'] = self._generate_pruned_trees(X_train_scaled, y_train, X_test_scaled, y_test)
            
            # Identify key nodes
            self.status_update.emit(f"Identifying important nodes (max {self.max_important_nodes})...")
            results['important_nodes'] = self._identify_important_nodes(tree_model, self.max_important_nodes)
            
            self.status_update.emit("Training and analysis complete!")
            self.progress_update.emit(100)
            
            # Emit results
            self.training_finished.emit(results)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_raised.emit(f"Error during tree training: {str(e)}")


    def _calculate_regression_metrics(self, y_train, y_train_pred, y_test, y_test_pred):
        """Calculate enhanced metrics for regression tasks"""
        # Calculate errors
        train_error = y_train - y_train_pred
        test_error = y_test - y_test_pred
        
        # Calculate absolute errors
        train_abs_error = np.abs(train_error)
        test_abs_error = np.abs(test_error)
        
        metrics = {
            'train': {
                'MSE': mean_squared_error(y_train, y_train_pred),
                'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'MAE': mean_absolute_error(y_train, y_train_pred),
                'R²': r2_score(y_train, y_train_pred),
                'Explained Variance': explained_variance_score(y_train, y_train_pred),
                'Max Error': np.max(np.abs(y_train - y_train_pred)),
                'Min Error': np.min(np.abs(y_train - y_train_pred)),
                'Sum of Errors': np.sum(y_train - y_train_pred),
                'Sum of Absolute Errors': np.sum(np.abs(y_train - y_train_pred)),
                'Mean Percentage Error': np.mean((y_train - y_train_pred) / y_train) * 100 if np.all(y_train != 0) else np.nan,
                'Mean Absolute Percentage Error': np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100 if np.all(y_train != 0) else np.nan,
                'Median Absolute Error': np.median(train_abs_error),
                'Median Error': np.median(train_error),
                'Error Standard Deviation': np.std(train_error),
                'Error Variance': np.var(train_error),
                '90th Percentile Error': np.percentile(train_abs_error, 90),
                '95th Percentile Error': np.percentile(train_abs_error, 95),
                '99th Percentile Error': np.percentile(train_abs_error, 99),
            },
            'test': {
                'MSE': mean_squared_error(y_test, y_test_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'MAE': mean_absolute_error(y_test, y_test_pred),
                'R²': r2_score(y_test, y_test_pred),
                'Explained Variance': explained_variance_score(y_test, y_test_pred),
                'Max Error': np.max(np.abs(y_test - y_test_pred)),
                'Min Error': np.min(np.abs(y_test - y_test_pred)),
                'Sum of Errors': np.sum(y_test - y_test_pred),
                'Sum of Absolute Errors': np.sum(np.abs(y_test - y_test_pred)),
                'Mean Percentage Error': np.mean((y_test - y_test_pred) / y_test) * 100 if np.all(y_test != 0) else np.nan,
                'Mean Absolute Percentage Error': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100 if np.all(y_test != 0) else np.nan,
                'Median Absolute Error': np.median(test_abs_error),
                'Median Error': np.median(test_error),
                'Error Standard Deviation': np.std(test_error),
                'Error Variance': np.var(test_error),
                '90th Percentile Error': np.percentile(test_abs_error, 90),
                '95th Percentile Error': np.percentile(test_abs_error, 95),
                '99th Percentile Error': np.percentile(test_abs_error, 99),
            }
        }
        
        # Store raw errors for further analysis
        metrics['raw_errors'] = {
            'train_error': train_error,
            'test_error': test_error,
            'train_abs_error': train_abs_error,
            'test_abs_error': test_abs_error
        }
        
        return metrics
    
    def _calculate_classification_metrics(self, y_train, y_train_pred, y_test, y_test_pred):
        """Calculate enhanced metrics for classification tasks"""
        # Handle multi-class vs binary
        average_method = 'binary' if len(np.unique(y_train)) == 2 else 'weighted'
        
        metrics = {
            'train': {
                'Accuracy': accuracy_score(y_train, y_train_pred),
                'Precision': precision_score(y_train, y_train_pred, average=average_method, zero_division=0),
                'Recall': recall_score(y_train, y_train_pred, average=average_method, zero_division=0),
                'F1 Score': f1_score(y_train, y_train_pred, average=average_method, zero_division=0),
                'Confusion Matrix': confusion_matrix(y_train, y_train_pred).tolist(),
                'Classification Report': classification_report(y_train, y_train_pred, output_dict=True),
                'Balanced Accuracy': np.mean([precision_score(y_train, y_train_pred, average=average_method, zero_division=0),
                                           recall_score(y_train, y_train_pred, average=average_method, zero_division=0)]),
                'Error Rate': 1 - accuracy_score(y_train, y_train_pred)
            },
            'test': {
                'Accuracy': accuracy_score(y_test, y_test_pred),
                'Precision': precision_score(y_test, y_test_pred, average=average_method, zero_division=0),
                'Recall': recall_score(y_test, y_test_pred, average=average_method, zero_division=0),
                'F1 Score': f1_score(y_test, y_test_pred, average=average_method, zero_division=0),
                'Confusion Matrix': confusion_matrix(y_test, y_test_pred).tolist(),
                'Classification Report': classification_report(y_test, y_test_pred, output_dict=True),
                'Balanced Accuracy': np.mean([precision_score(y_test, y_test_pred, average=average_method, zero_division=0),
                                           recall_score(y_test, y_test_pred, average=average_method, zero_division=0)]),
                'Error Rate': 1 - accuracy_score(y_test, y_test_pred)
            }
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y_train)) == 2:
            try:
                metrics['train']['ROC AUC'] = roc_auc_score(y_train, y_train_pred)
                metrics['test']['ROC AUC'] = roc_auc_score(y_test, y_test_pred)
            except:
                # If ROC AUC can't be calculated
                metrics['train']['ROC AUC'] = np.nan
                metrics['test']['ROC AUC'] = np.nan
        
        return metrics
        
    def _generate_pruned_trees(self, X_train, y_train, X_test, y_test):
        """Generate pruned trees using cost complexity pruning"""
        try:
            # Create a copy of tree parameters without ccp_alpha
            tree_params = {k: v for k, v in self.tree_params.items() if k != 'ccp_alpha'}
            
            if self.task_type == 'regression':
                base_model = DecisionTreeRegressor(**tree_params)
            else:
                base_model = DecisionTreeClassifier(**tree_params)
                
            # Fit the base model to get alphas
            base_model.fit(X_train, y_train)
            
            # Get the cost complexity pruning path
            path = base_model.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
            
            # Skip the last alpha that would result in a single node tree
            if len(ccp_alphas) > 1:
                ccp_alphas = ccp_alphas[:-1]
            
            # If there are too many alphas, select a subset
            if len(ccp_alphas) > 10:
                indices = np.linspace(0, len(ccp_alphas) - 1, 10, dtype=int)
                ccp_alphas = ccp_alphas[indices]
            
            pruned_trees = []
            for alpha in ccp_alphas:
                if self.task_type == 'regression':
                    pruned_model = DecisionTreeRegressor(**tree_params, ccp_alpha=alpha)
                else:
                    pruned_model = DecisionTreeClassifier(**tree_params, ccp_alpha=alpha)
                
                pruned_model.fit(X_train, y_train)
                
                # Calculate scores
                train_score = pruned_model.score(X_train, y_train)
                test_score = pruned_model.score(X_test, y_test)
                
                pruned_trees.append({
                    'alpha': alpha,
                    'train_score': train_score,
                    'test_score': test_score,
                    'num_nodes': pruned_model.tree_.node_count,
                    'model': pruned_model
                })
            
            return {
                'alphas': ccp_alphas,
                'trees': pruned_trees
            }
        except Exception as e:
            print(f"Error in pruning: {str(e)}")
            return None
    def _identify_important_nodes(self, tree_model, max_nodes=20):
        """Identify key nodes in the decision tree"""
        try:
            tree = tree_model.tree_
            
            # Get node importances based on samples and impurity reduction
            n_nodes = tree.node_count
            node_importance = np.zeros(n_nodes)
            
            for i in range(n_nodes):
                if tree.children_left[i] != tree.children_right[i]:  # Not a leaf
                    # Weight by number of samples and impurity reduction
                    node_importance[i] = (
                        tree.n_node_samples[i] / tree.n_node_samples[0] *  # relative node sample size
                        (tree.impurity[i] - (tree.n_node_samples[tree.children_left[i]] / tree.n_node_samples[i]) * tree.impurity[tree.children_left[i]] -
                        (tree.n_node_samples[tree.children_right[i]] / tree.n_node_samples[i]) * tree.impurity[tree.children_right[i]])
                    )
            
            # Get top important nodes
            max_nodes = min(max_nodes, n_nodes) if max_nodes > 0 else min(5, n_nodes)
            top_nodes = np.argsort(node_importance)[-max_nodes:][::-1] if len(node_importance) >= max_nodes else np.argsort(node_importance)[::-1]
            
            # Get info for each important node
            important_nodes = []
            for node_id in top_nodes:
                if node_importance[node_id] > 0:  # Only include nodes with positive importance
                    node_info = {
                        'id': node_id,
                        'importance': node_importance[node_id],
                        'samples': tree.n_node_samples[node_id],
                        'impurity': tree.impurity[node_id],
                        'feature': tree.feature[node_id] if tree.children_left[node_id] != tree.children_right[node_id] else None,
                        'threshold': tree.threshold[node_id] if tree.children_left[node_id] != tree.children_right[node_id] else None,
                        'is_leaf': tree.children_left[node_id] == tree.children_right[node_id],
                        'depth': None,  # Will be calculated later
                        'decision_path': None  # Will be calculated later
                    }
                    important_nodes.append(node_info)
            
            # Calculate depth for each node
            def get_node_depth(node_id, tree, current_depth=0):
                if node_id == 0:  # Root node
                    return 0
                
                # Find parent
                for i in range(tree.node_count):
                    if tree.children_left[i] == node_id or tree.children_right[i] == node_id:
                        return get_node_depth(i, tree, current_depth) + 1
                
                return -1  # Should not happen
            
            # Calculate decision path for each important node
            def get_decision_path(node_id, tree, feature_names):
                path = []
                current = node_id
                
                # Work backwards from node to root
                while current != 0:
                    # Find parent node
                    parent = -1
                    left_child = False
                    
                    for i in range(tree.node_count):
                        if tree.children_left[i] == current:
                            parent = i
                            left_child = True
                            break
                        elif tree.children_right[i] == current:
                            parent = i
                            left_child = False
                            break
                    
                    if parent == -1:
                        break  # Should not happen
                    
                    # Add decision to path
                    feature_id = tree.feature[parent]
                    threshold = tree.threshold[parent]
                    
                    feature_name = feature_names[feature_id] if feature_id < len(feature_names) else f"Feature {feature_id}"
                    if feature_id == -2 or feature_id >= len(feature_names):
                        feature_name = "N/A"
                    else:
                        feature_name = feature_names[feature_id]

                    if left_child:
                        decision = f"{feature_name} <= {threshold:.6f}"
                    else:
                        decision = f"{feature_name} > {threshold:.6f}"
                    
                    path.append((parent, decision))
                    
                    # Move to parent
                    current = parent
                
                # Path is in reverse order, flip it
                path.reverse()
                return path
            
            # Update depth and decision path for each important node
            for node in important_nodes:
                node['depth'] = get_node_depth(node['id'], tree)
                node['decision_path'] = get_decision_path(node['id'], tree, self.input_features)
            
            return important_nodes
            
        except Exception as e:
            print(f"Error identifying important nodes: {str(e)}")
            return []

#------------------------------------------------------------------------------
# Custom UI Components
#------------------------------------------------------------------------------

class MatplotlibCanvas(FigureCanvas):
    """Canvas for matplotlib plots"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

class ErrorAnalysisTab(QWidget):
    """Tab for detailed error analysis"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Controls area
        controls_layout = QHBoxLayout()
        
        # Dataset selector
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QHBoxLayout()
        self.train_set_cb = QCheckBox("Training Set")
        self.train_set_cb.setChecked(True)
        self.test_set_cb = QCheckBox("Test Set")
        self.test_set_cb.setChecked(True)
        dataset_layout.addWidget(self.train_set_cb)
        dataset_layout.addWidget(self.test_set_cb)
        dataset_group.setLayout(dataset_layout)
        controls_layout.addWidget(dataset_group)
        
        # Error type selector
        error_type_group = QGroupBox("Error Type")
        error_type_layout = QHBoxLayout()
        self.signed_error_cb = QCheckBox("Signed Error")
        self.signed_error_cb.setChecked(True)
        self.absolute_error_cb = QCheckBox("Absolute Error")
        self.absolute_error_cb.setChecked(True)
        error_type_layout.addWidget(self.signed_error_cb)
        error_type_layout.addWidget(self.absolute_error_cb)
        error_type_group.setLayout(error_type_layout)
        controls_layout.addWidget(error_type_group)
        
        # Histogram bin control
        bin_group = QGroupBox("Histogram Bins")
        bin_layout = QHBoxLayout()
        bin_layout.addWidget(QLabel("Bins:"))
        self.bin_slider = QSlider(Qt.Horizontal)
        self.bin_slider.setRange(5, 100)
        self.bin_slider.setValue(30)
        self.bin_slider.setTickInterval(5)
        self.bin_slider.setTickPosition(QSlider.TicksBelow)
        self.bin_value = QLabel("30")
        self.bin_slider.valueChanged.connect(lambda v: self.bin_value.setText(str(v)))
        bin_layout.addWidget(self.bin_slider)
        bin_layout.addWidget(self.bin_value)
        bin_group.setLayout(bin_layout)
        controls_layout.addWidget(bin_group)
        
        # Apply button
        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.clicked.connect(self.apply_settings)
        controls_layout.addWidget(self.apply_btn)
        
        layout.addLayout(controls_layout)
        # Tabs for different visualizations
        self.viz_tabs = QTabWidget()
        
        # Create individual tabs for each visualization
        self.create_error_dist_tab()
        self.create_actual_vs_pred_tab()
        self.create_error_index_tab()
        self.create_top_errors_tab()
        self.create_residual_plots_tab()
        
        layout.addWidget(self.viz_tabs)
    # Add this new method to handle the apply button click
    def apply_settings(self):
        """Apply the current settings and update all plots"""
        # Always get the latest tree results from main application
        if hasattr(self, 'main_app') and self.main_app and hasattr(self.main_app, 'tree_results'):
            latest_results = self.main_app.tree_results
            if latest_results:
                self.update_plots(latest_results)
            else:
                print("No tree results available in main application")
        else:
            # Fallback to stored results if main app reference is not available
            if self.current_tree_results:
                self.update_plots(self.current_tree_results)
            else:
                print("No tree results available for error analysis")
    def create_error_dist_tab(self):
        """Create tab for error distribution visualization"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Canvas for histogram
        self.error_dist_canvas = MatplotlibCanvas(self, width=8, height=5)
        self.error_dist_toolbar = NavigationToolbar(self.error_dist_canvas, self)
        layout.addWidget(self.error_dist_toolbar)
        layout.addWidget(self.error_dist_canvas)
        
        self.viz_tabs.addTab(tab, "Error Distribution")
        
    def create_actual_vs_pred_tab(self):
        """Create tab for actual vs predicted visualization"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Canvas
        self.actual_pred_canvas = MatplotlibCanvas(self, width=8, height=5)
        self.actual_pred_toolbar = NavigationToolbar(self.actual_pred_canvas, self)
        layout.addWidget(self.actual_pred_toolbar)
        layout.addWidget(self.actual_pred_canvas)
        
        self.viz_tabs.addTab(tab, "Actual vs Predicted")
        
    def create_error_index_tab(self):
        """Create tab for error vs index visualization"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Canvas
        self.error_index_canvas = MatplotlibCanvas(self, width=8, height=5)
        self.error_index_toolbar = NavigationToolbar(self.error_index_canvas, self)
        layout.addWidget(self.error_index_toolbar)
        layout.addWidget(self.error_index_canvas)
        
        self.viz_tabs.addTab(tab, "Error vs Index")
        
    def create_top_errors_tab(self):
        """Create tab for top errors table"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Show Top:"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 100)
        self.top_n_spin.setValue(20)
        controls_layout.addWidget(self.top_n_spin)
        self.update_top_errors_btn = QPushButton("Refresh")
        controls_layout.addWidget(self.update_top_errors_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Table
        self.top_errors_table = QTableWidget()
        layout.addWidget(self.top_errors_table)
        
        self.viz_tabs.addTab(tab, "Top Errors")
        
    def create_residual_plots_tab(self):
        """Create tab for residual analysis plots"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # 2x2 Grid of plots
        self.residual_canvas = MatplotlibCanvas(self, width=10, height=8)
        self.residual_toolbar = NavigationToolbar(self.residual_canvas, self)
        layout.addWidget(self.residual_toolbar)
        layout.addWidget(self.residual_canvas)
        
        self.viz_tabs.addTab(tab, "Residual Analysis")
    
     # Modify the update_plots method to store the tree_results
    def update_plots(self, tree_results):
        """Update all plots with the tree results"""
        if not tree_results or tree_results['parameters']['task_type'] != 'regression':
            return
            
        # Store the tree results for later use
        self.current_tree_results = tree_results
            
        # Extract data
        y_train = tree_results['raw_data']['y_train'].reset_index(drop=True)
        y_train_pred = pd.Series(tree_results['predictions']['y_train_pred'])
        y_test = tree_results['raw_data']['y_test'].reset_index(drop=True)
        y_test_pred = pd.Series(tree_results['predictions']['y_test_pred'])
        
        # Compute errors
        train_error = y_train - y_train_pred
        test_error = y_test - y_test_pred
        train_abs_error = np.abs(train_error)
        test_abs_error = np.abs(test_error)
        
        # Update plots based on selected settings
        show_train = self.train_set_cb.isChecked()
        show_test = self.test_set_cb.isChecked()
        show_signed = self.signed_error_cb.isChecked()
        show_abs = self.absolute_error_cb.isChecked()
        bins = self.bin_slider.value()
        
        # Update error distribution
        self.update_error_distribution(
            train_error, test_error, train_abs_error, test_abs_error,
            show_train, show_test, show_signed, show_abs, bins
        )
        
        # Update actual vs predicted
        self.update_actual_vs_predicted(
            y_train, y_train_pred, y_test, y_test_pred,
            show_train, show_test
        )
        
        # Update error vs index
        self.update_error_vs_index(
            train_error, test_error, train_abs_error, test_abs_error,
            show_train, show_test, show_signed, show_abs
        )
        
        # Update top errors table
        self.update_top_errors_table(
            y_train, y_train_pred, y_test, y_test_pred,
            train_error, test_error, train_abs_error, test_abs_error,
            show_train, show_test, self.top_n_spin.value()
        )
        
        # Update residual plots
        self.update_residual_plots(
            y_train, y_train_pred, y_test, y_test_pred,
            train_error, test_error, show_train, show_test
        )
    def update_error_distribution(self, train_error, test_error, train_abs_error, test_abs_error, 
                                show_train, show_test, show_signed, show_abs, bins):
        """Update error distribution histogram"""
        self.error_dist_canvas.fig.clear()
        ax = self.error_dist_canvas.fig.add_subplot(111)
        
        # Plot histograms based on selections
        plots = 0
        if show_train and show_signed:
            sns.histplot(train_error, bins=bins, kde=True, ax=ax, 
                        color='blue', alpha=0.6, label='Train (Signed)')
            plots += 1
        
        if show_test and show_signed:
            sns.histplot(test_error, bins=bins, kde=True, ax=ax, 
                        color='red', alpha=0.6, label='Test (Signed)')
            plots += 1
            
        if show_train and show_abs:
            sns.histplot(train_abs_error, bins=bins, kde=True, ax=ax, 
                        color='green', alpha=0.6, label='Train (Absolute)')
            plots += 1
            
        if show_test and show_abs:
            sns.histplot(test_abs_error, bins=bins, kde=True, ax=ax, 
                        color='purple', alpha=0.6, label='Test (Absolute)')
            plots += 1
            
        # Add vertical line at zero for signed errors
        if show_signed and (show_train or show_test):
            ax.axvline(0, linestyle='--', color='black', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('Error Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        
        # Add legend if multiple plots
        if plots > 1:
            ax.legend()
            
        self.error_dist_canvas.fig.tight_layout()
        self.error_dist_canvas.draw()
    
    def update_actual_vs_predicted(self, y_train, y_train_pred, y_test, y_test_pred, show_train, show_test):
        """Update actual vs predicted scatter plot"""
        self.actual_pred_canvas.fig.clear()
        ax = self.actual_pred_canvas.fig.add_subplot(111)
        
        # Plot scatter plots based on selections
        if show_train:
            ax.scatter(y_train, y_train_pred, alpha=0.6, 
                    color='blue', label='Training Set')
            
        if show_test:
            ax.scatter(y_test, y_test_pred, alpha=0.6, 
                     color='red', label='Test Set')
            
        # Add perfect prediction line
        if show_train or show_test:
            all_y = pd.concat([y_train, y_test])
            ax.plot([all_y.min(), all_y.max()], [all_y.min(), all_y.max()], 
                  'k--', label='Perfect Prediction')
            
        # Set labels and title
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')
        
        # Add legend if both sets shown
        if show_train and show_test:
            ax.legend()
            
        self.actual_pred_canvas.fig.tight_layout()
        self.actual_pred_canvas.draw()
    
    def update_error_vs_index(self, train_error, test_error, train_abs_error, test_abs_error, 
                           show_train, show_test, show_signed, show_abs):
        """Update error vs index plot"""
        self.error_index_canvas.fig.clear()
        
        # Create subplots based on number of selected options
        num_plots = (show_train and show_signed) + (show_test and show_signed) + \
                   (show_train and show_abs) + (show_test and show_abs)
        
        if num_plots == 0:
            return
            
        # Create figure with appropriate number of subplots
        if num_plots == 1:
            axs = [self.error_index_canvas.fig.add_subplot(111)]
        elif num_plots == 2:
            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            axs = axs.flatten()
            
        plot_idx = 0
        
        # Plot data based on selections
        if show_train and show_signed:
            axs[plot_idx].plot(range(len(train_error)), train_error, 
                           color='blue', alpha=0.7)
            axs[plot_idx].set_title('Training Set - Signed Error')
            axs[plot_idx].set_xlabel('Index')
            axs[plot_idx].set_ylabel('Signed Error')
            axs[plot_idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plot_idx += 1
            
        if show_test and show_signed:
            axs[plot_idx].plot(range(len(test_error)), test_error, 
                           color='red', alpha=0.7)
            axs[plot_idx].set_title('Test Set - Signed Error')
            axs[plot_idx].set_xlabel('Index')
            axs[plot_idx].set_ylabel('Signed Error')
            axs[plot_idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plot_idx += 1
            
        if show_train and show_abs:
            axs[plot_idx].plot(range(len(train_abs_error)), train_abs_error, 
                           color='green', alpha=0.7)
            axs[plot_idx].set_title('Training Set - Absolute Error')
            axs[plot_idx].set_xlabel('Index')
            axs[plot_idx].set_ylabel('Absolute Error')
            plot_idx += 1
            
        if show_test and show_abs:
            axs[plot_idx].plot(range(len(test_abs_error)), test_abs_error, 
                           color='purple', alpha=0.7)
            axs[plot_idx].set_title('Test Set - Absolute Error')
            axs[plot_idx].set_xlabel('Index')
            axs[plot_idx].set_ylabel('Absolute Error')
        
        self.error_index_canvas.fig.tight_layout()
        self.error_index_canvas.draw()
    def update_top_errors_table(self, y_train, y_train_pred, y_test, y_test_pred, 
                             train_error, test_error, train_abs_error, test_abs_error,
                             show_train, show_test, top_n):
        """Update top errors table"""
        # Create DataFrames
        if show_train:
            train_df = pd.DataFrame({
                'Dataset': ['Train'] * len(y_train),
                'Index': y_train.index,
                'Actual': y_train,
                'Predicted': y_train_pred,
                'Error': train_error,
                'AbsError': train_abs_error
            })
        else:
            train_df = pd.DataFrame()
            
        if show_test:
            test_df = pd.DataFrame({
                'Dataset': ['Test'] * len(y_test),
                'Index': y_test.index,
                'Actual': y_test,
                'Predicted': y_test_pred,
                'Error': test_error,
                'AbsError': test_abs_error
            })
        else:
            test_df = pd.DataFrame()
            
        # Combine datasets
        combined_df = pd.concat([train_df, test_df])
        
        if combined_df.empty:
            return
            
        # Sort by absolute error and get top N
        top_errors = combined_df.sort_values('AbsError', ascending=False).head(top_n)
        
        # Update table
        self.top_errors_table.setRowCount(len(top_errors))
        self.top_errors_table.setColumnCount(6)
        self.top_errors_table.setHorizontalHeaderLabels(
            ['Dataset', 'Index', 'Actual', 'Predicted', 'Error', 'Absolute Error']
        )
        
        # Fill table
        for i, (_, row) in enumerate(top_errors.iterrows()):
            self.top_errors_table.setItem(i, 0, QTableWidgetItem(row['Dataset']))
            self.top_errors_table.setItem(i, 1, QTableWidgetItem(str(row['Index'])))
            self.top_errors_table.setItem(i, 2, QTableWidgetItem(f"{row['Actual']:.6f}"))
            self.top_errors_table.setItem(i, 3, QTableWidgetItem(f"{row['Predicted']:.6f}"))
            self.top_errors_table.setItem(i, 4, QTableWidgetItem(f"{row['Error']:.6f}"))
            self.top_errors_table.setItem(i, 5, QTableWidgetItem(f"{row['AbsError']:.6f}"))
            
            # Highlight high errors
            if row['AbsError'] > combined_df['AbsError'].mean() + 2 * combined_df['AbsError'].std():
                for j in range(6):
                    self.top_errors_table.item(i, j).setBackground(QBrush(QColor(255, 200, 200)))
        
        self.top_errors_table.resizeColumnsToContents()
    
    def update_residual_plots(self, y_train, y_train_pred, y_test, y_test_pred,
                           train_error, test_error, show_train, show_test):
        """Update residual analysis plots"""
        self.residual_canvas.fig.clear()
        
        # Create 2x2 grid of plots
        axs = self.residual_canvas.fig.subplots(2, 2)
        
        # 1. Residuals vs Predicted
        ax1 = axs[0, 0]
        if show_train:
            ax1.scatter(y_train_pred, train_error, alpha=0.6, color='blue', label='Train')
        if show_test:
            ax1.scatter(y_test_pred, test_error, alpha=0.6, color='red', label='Test')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        if show_train and show_test:
            ax1.legend()
            
        # 2. Q-Q Plot
        ax2 = axs[0, 1]
        if show_train:
            stats.probplot(train_error, plot=ax2)
            ax2.set_title('Q-Q Plot (Train)')
        elif show_test:
            stats.probplot(test_error, plot=ax2)
            ax2.set_title('Q-Q Plot (Test)')
            
        # 3. Residual Histogram
        ax3 = axs[1, 0]
        if show_train:
            sns.histplot(train_error, kde=True, ax=ax3, color='blue', alpha=0.6, label='Train')
        if show_test:
            sns.histplot(test_error, kde=True, ax=ax3, color='red', alpha=0.6, label='Test')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residual Distribution')
        if show_train and show_test:
            ax3.legend()
            
        # 4. Residual Lag Plot
        ax4 = axs[1, 1]
        if show_train and len(train_error) > 1:
            ax4.scatter(train_error[:-1], train_error[1:], alpha=0.6, color='blue', label='Train')
            ax4.set_title('Residual Lag Plot (Train)')
        elif show_test and len(test_error) > 1:
            ax4.scatter(test_error[:-1], test_error[1:], alpha=0.6, color='red', label='Test')
            ax4.set_title('Residual Lag Plot (Test)')
        ax4.set_xlabel('Residual(t)')
        ax4.set_ylabel('Residual(t+1)')
            
        self.residual_canvas.fig.tight_layout()
        self.residual_canvas.draw()

#------------------------------------------------------------------------------
# Dialogs
#------------------------------------------------------------------------------

class TreePruningDialog(QDialog):
    """Dialog for tree pruning and optimization"""
    def __init__(self, tree_results, parent=None):
        super().__init__(parent)
        self.tree_results = tree_results
        self.selected_alpha = None
        self.selected_tree = None
        
        self.setWindowTitle("Decision Tree Pruning")
        self.setMinimumSize(800, 600)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI elements"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        if not self.tree_results or 'pruning' not in self.tree_results or not self.tree_results['pruning']:
            layout.addWidget(QLabel("Pruning results not available."))
            return
        
        # Pruning performance plot
        plot_group = QGroupBox("Pruning Performance")
        plot_layout = QVBoxLayout()
        self.pruning_canvas = MatplotlibCanvas(self, width=8, height=5)
        self.pruning_toolbar = NavigationToolbar(self.pruning_canvas, self)
        plot_layout.addWidget(self.pruning_toolbar)
        plot_layout.addWidget(self.pruning_canvas)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
        # Alpha selector
        selector_group = QGroupBox("Select Optimal Tree")
        selector_layout = QVBoxLayout()
        
        self.alpha_table = QTableWidget()
        self.alpha_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.alpha_table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        selector_layout.addWidget(self.alpha_table)
        selector_group.setLayout(selector_layout)
        layout.addWidget(selector_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.select_btn = QPushButton("Select Tree")
        self.select_btn.clicked.connect(self.select_tree)
        button_layout.addWidget(self.select_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Update plots and table
        self.update_pruning_plot()
        self.update_alpha_table()
        
    def update_pruning_plot(self):
        """Update the pruning performance plot"""
        pruning_data = self.tree_results['pruning']
        trees = pruning_data['trees']
        
        if not trees:
            return
            
        # Extract data for plotting
        alphas = [tree['alpha'] for tree in trees]
        train_scores = [tree['train_score'] for tree in trees]
        test_scores = [tree['test_score'] for tree in trees]
        num_nodes = [tree['num_nodes'] for tree in trees]
        
        # Clear canvas
        self.pruning_canvas.fig.clear()
        
        # Create two subplots
        ax1 = self.pruning_canvas.fig.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Plot scores on first y-axis
        ax1.plot(alphas, train_scores, 'b-', marker='o', label='Train Score')
        ax1.plot(alphas, test_scores, 'r-', marker='s', label='Test Score')
        ax1.set_xscale('log')
        ax1.set_xlabel('alpha')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot number of nodes on second y-axis
        ax2.plot(alphas, num_nodes, 'g--', marker='^', label='Num Nodes')
        ax2.set_ylabel('Number of Nodes')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        ax1.set_title('Pruning Performance vs Complexity')
        self.pruning_canvas.fig.tight_layout()
        self.pruning_canvas.draw()
    def update_alpha_table(self):
        """Update alpha selection table"""
        pruning_data = self.tree_results['pruning']
        trees = pruning_data['trees']
        
        if not trees:
            return
            
        # Setup table
        self.alpha_table.setRowCount(len(trees))
        self.alpha_table.setColumnCount(4)
        self.alpha_table.setHorizontalHeaderLabels(['Alpha', 'Train Score', 'Test Score', 'Num Nodes'])
        
        # Find best test score for highlighting
        best_test_idx = np.argmax([t['test_score'] for t in trees])
        
        # Fill table
        for i, tree in enumerate(trees):
            self.alpha_table.setItem(i, 0, QTableWidgetItem(f"{tree['alpha']:.8f}"))
            self.alpha_table.setItem(i, 1, QTableWidgetItem(f"{tree['train_score']:.6f}"))
            self.alpha_table.setItem(i, 2, QTableWidgetItem(f"{tree['test_score']:.6f}"))
            self.alpha_table.setItem(i, 3, QTableWidgetItem(str(tree['num_nodes'])))
            
            # Highlight best test score
            if i == best_test_idx:
                for j in range(4):
                    self.alpha_table.item(i, j).setBackground(QBrush(QColor(200, 255, 200)))
        
        self.alpha_table.resizeColumnsToContents()
        
        # Select best row by default
        self.alpha_table.selectRow(best_test_idx)
        
    def select_tree(self):
        """Select the chosen pruned tree"""
        selected_rows = self.alpha_table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row_idx = selected_rows[0].row()
        self.selected_alpha = self.tree_results['pruning']['trees'][row_idx]['alpha']
        self.selected_tree = self.tree_results['pruning']['trees'][row_idx]['model']
        
        self.accept()


class NodeDetailsDialog(QDialog):
    """Dialog for detailed node analysis"""
    def __init__(self, tree_results, node_id, parent=None):
        super().__init__(parent)
        self.tree_results = tree_results
        self.node_id = node_id
        
        self.setWindowTitle(f"Node {node_id} Analysis")
        self.setMinimumSize(800, 600)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI elements"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        if not self.tree_results or 'tree_structure' not in self.tree_results:
            layout.addWidget(QLabel("Tree results not available."))
            return
            
        # Node details
        details_group = QGroupBox("Node Details")
        details_layout = QVBoxLayout()
        self.node_details = QTextEdit()
        self.node_details.setReadOnly(True)
        details_layout.addWidget(self.node_details)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # Sample distribution plot
        dist_group = QGroupBox("Sample Distribution")
        dist_layout = QVBoxLayout()
        self.dist_canvas = MatplotlibCanvas(self, width=8, height=4)
        dist_layout.addWidget(self.dist_canvas)
        dist_group.setLayout(dist_layout)
        layout.addWidget(dist_group)
        
        # Decision path
        path_group = QGroupBox("Decision Path")
        path_layout = QVBoxLayout()
        self.path_text = QTextEdit()
        self.path_text.setReadOnly(True)
        self.path_text.setFont(QFont("Courier New", 10))
        path_layout.addWidget(self.path_text)
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)
        
        # Update content
        self.update_node_details()
        self.update_distribution_plot()
        self.update_decision_path()
        
    def update_node_details(self):
        """Update node details text"""
        if not self.tree_results:
            return
            
        tree = self.tree_results['tree_structure']['model'].tree_
        feature_names = self.tree_results['tree_structure']['feature_names']
        task_type = self.tree_results['parameters']['task_type']
        
        # Basic node info
        is_leaf = tree.children_left[self.node_id] == tree.children_right[self.node_id]
        node_type = "Leaf Node" if is_leaf else "Internal Node"
        
        details = f"Node {self.node_id} Information\n{'='*30}\n\n"
        details += f"Type: {node_type}\n"
        
        if not is_leaf:
            # Feature and threshold for split
            feature_id = tree.feature[self.node_id]
            threshold = tree.threshold[self.node_id]
            
            feature_name = feature_names[feature_id] if feature_id < len(feature_names) else f"Feature {feature_id}"
            
            details += f"Split Feature: {feature_name}\n"
            details += f"Split Threshold: {threshold:.6f}\n"
            
            # Children
            left_child = tree.children_left[self.node_id]
            right_child = tree.children_right[self.node_id]
            
            details += f"Left Child Node: {left_child}\n"
            details += f"Right Child Node: {right_child}\n"
        
        # Samples
        n_samples = tree.n_node_samples[self.node_id]
        details += f"Number of Samples: {n_samples}\n"
        details += f"Percentage of Total: {n_samples/tree.n_node_samples[0]*100:.2f}%\n"
        
        # Node value interpretation
        if task_type == 'regression':
            value = tree.value[self.node_id][0][0]
            details += f"Predicted Value: {value:.6f}\n"
        else:
            # Classification - show class distribution
            class_counts = tree.value[self.node_id][0]
            details += f"Class Distribution: {class_counts}\n"
            
            # Get class names if available
            try:
                target_name = self.tree_results['tree_structure']['target_name']
                class_names = sorted(list(set(self.tree_results['raw_data']['y_train'].tolist())))
                
                # Display distribution with class names
                details += f"\nDetailed Class Distribution:\n"
                total_samples = sum(class_counts)
                for i, (count, class_name) in enumerate(zip(class_counts, class_names)):
                    percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                    details += f"{class_name}: {count} ({percentage:.2f}%)\n"
            except:
                pass
        
        # Node impurity and depth
        impurity = tree.impurity[self.node_id]
        impurity_name = "Gini" if task_type == "classification" else "MSE"
        details += f"Node {impurity_name}: {impurity:.6f}\n"
        
        # Calculate node depth
        depth = 0
        current = self.node_id
        while current != 0:  # Not root
            # Find parent
            for i in range(tree.node_count):
                if tree.children_left[i] == current or tree.children_right[i] == current:
                    current = i
                    depth += 1
                    break
        
        details += f"Node Depth: {depth}\n"
        
        self.node_details.setText(details)
    
    def update_distribution_plot(self):
        """Update sample distribution plot"""
        if not self.tree_results:
            return
            
        tree = self.tree_results['tree_structure']['model'].tree_
        task_type = self.tree_results['parameters']['task_type']
        
        # Clear the canvas
        self.dist_canvas.fig.clear()
        ax = self.dist_canvas.fig.add_subplot(111)
        
        if task_type == 'classification':
            # Classification - bar chart of class distribution
            class_counts = tree.value[self.node_id][0]
            
            # Get class names if available
            try:
                target_name = self.tree_results['tree_structure']['target_name']
                class_names = sorted(list(set(self.tree_results['raw_data']['y_train'].tolist())))
            except:
                class_names = [f"Class {i}" for i in range(len(class_counts))]
            
            # Create bar chart
            ax.bar(class_names, class_counts)
            ax.set_ylabel('Number of Samples')
            ax.set_title(f'Class Distribution in Node {self.node_id}')
            
            # Add counts on top of bars
            for i, count in enumerate(class_counts):
                ax.text(i, count + 0.1, str(int(count)), ha='center')
            
            # Rotate x labels if many classes
            if len(class_names) > 4:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            # Regression - visualization
            node_value = tree.value[self.node_id][0][0]
            
            # Get target variable and its distribution
            target_name = self.tree_results['tree_structure']['target_name']
            target_values = pd.concat([
                self.tree_results['raw_data']['y_train'],
                self.tree_results['raw_data']['y_test']
            ])
            
            # Create histogram of overall distribution
            sns.histplot(target_values, kde=True, ax=ax, alpha=0.5, color='blue', label='Overall Distribution')
            
            # Add vertical line for node prediction
            ax.axvline(node_value, color='red', linestyle='--', 
                      label=f'Node Prediction: {node_value:.4f}')
            
            ax.set_xlabel(target_name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Target Distribution with Node {self.node_id} Prediction')
            ax.legend()
        
        # Update the canvas
        self.dist_canvas.fig.tight_layout()
        self.dist_canvas.draw()
    
    def update_decision_path(self):
        """Update decision path text"""
        if not self.tree_results:
            return
            
        tree = self.tree_results['tree_structure']['model'].tree_
        feature_names = self.tree_results['tree_structure']['feature_names']
        
        # Clear existing text
        self.path_text.clear()
        
        # If it's the root node, there's no path
        if self.node_id == 0:
            self.path_text.setText("This is the root node (no decision path).")
            return
        
        # Trace path from root to node
        path = []
        current = self.node_id
        
        # Work backwards from node to root
        while current != 0:
            # Find parent node
            parent = -1
            left_child = False
            
            for i in range(tree.node_count):
                if tree.children_left[i] == current:
                    parent = i
                    left_child = True
                    break
                elif tree.children_right[i] == current:
                    parent = i
                    left_child = False
                    break
            
            if parent == -1:
                # Couldn't find parent, something went wrong
                break
            
            # Add decision to path
            feature_id = tree.feature[parent]
            threshold = tree.threshold[parent]
            
            feature_name = feature_names[feature_id] if feature_id < len(feature_names) else f"Feature {feature_id}"
            
            if left_child:
                decision = f"{feature_name} <= {threshold:.6f}"
            else:
                decision = f"{feature_name} > {threshold:.6f}"
            
            path.append((parent, decision))
            
            # Move to parent
            current = parent
        
        # Path is in reverse order, flip it
        path.reverse()
        
        # Display path
        self.path_text.append(f"Decision Path to Node {self.node_id}\n{'='*30}\n")
        
        for i, (parent, decision) in enumerate(path, 1):
            self.path_text.append(f"{i}. Node {parent}: {decision}")


class AdvancedTreeVisualizationDialog(QDialog):
    """Dialog for advanced tree visualization with range selection"""
    def __init__(self, tree_results, parent=None):
        super().__init__(parent)
        self.tree_results = tree_results
        
        self.setWindowTitle("Advanced Tree Visualization")
        self.setMinimumSize(1000, 800)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI elements"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        if not self.tree_results or 'tree_structure' not in self.tree_results:
            layout.addWidget(QLabel("Tree results not available."))
            return
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Depth range selection
        depth_group = QGroupBox("Depth Range")
        depth_layout = QHBoxLayout()
        
        max_depth = self.tree_results['tree_structure']['max_depth']
        
        depth_layout.addWidget(QLabel("From:"))
        self.min_depth_spin = QSpinBox()
        self.min_depth_spin.setRange(0, max_depth)
        self.min_depth_spin.setValue(0)
        depth_layout.addWidget(self.min_depth_spin)
        
        depth_layout.addWidget(QLabel("To:"))
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(0, max_depth)
        self.max_depth_spin.setValue(min(max_depth, 3))
        depth_layout.addWidget(self.max_depth_spin)
        
        depth_group.setLayout(depth_layout)
        controls_layout.addWidget(depth_group)
        
        # Visual options
        options_group = QGroupBox("Visualization Options")
        options_layout = QVBoxLayout()
        
        self.show_values_cb = QCheckBox("Show Node Values")
        self.show_values_cb.setChecked(True)
        options_layout.addWidget(self.show_values_cb)
        
        self.show_features_cb = QCheckBox("Show Feature Names")
        self.show_features_cb.setChecked(True)
        options_layout.addWidget(self.show_features_cb)
        
        self.filled_cb = QCheckBox("Filled (Color Nodes)")
        self.filled_cb.setChecked(True)
        options_layout.addWidget(self.filled_cb)
        
        options_group.setLayout(options_layout)
        controls_layout.addWidget(options_group)
        
        # Font size
        font_group = QGroupBox("Font Size")
        font_layout = QHBoxLayout()
        
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 14)
        self.font_size_spin.setValue(10)
        font_layout.addWidget(self.font_size_spin)
        
        font_group.setLayout(font_layout)
        controls_layout.addWidget(font_group)
        
        # Update button
        self.update_btn = QPushButton("Update Visualization")
        self.update_btn.clicked.connect(self.update_visualization)
        controls_layout.addWidget(self.update_btn)
        
        layout.addLayout(controls_layout)
        
        # Tree visualization in scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        tree_widget = QWidget()
        tree_layout = QVBoxLayout()
        tree_widget.setLayout(tree_layout)
        
        self.tree_canvas = MatplotlibCanvas(self, width=12, height=8)
        self.tree_toolbar = NavigationToolbar(self.tree_canvas, self)
        
        tree_layout.addWidget(self.tree_toolbar)
        tree_layout.addWidget(self.tree_canvas)
        
        scroll_area.setWidget(tree_widget)
        layout.addWidget(scroll_area)
        
        # Initial visualization
        self.update_visualization()
    
    def update_visualization(self):
        """Update tree visualization based on settings"""
        if not self.tree_results:
            return
            
        # Get visualization parameters
        min_depth = self.min_depth_spin.value()
        max_depth = self.max_depth_spin.value()
        show_values = self.show_values_cb.isChecked()
        show_features = self.show_features_cb.isChecked()
        filled = self.filled_cb.isChecked()
        fontsize = self.font_size_spin.value()
        
        # Validate depth range
        if min_depth > max_depth:
            QMessageBox.warning(self, "Invalid Range", "Minimum depth cannot be greater than maximum depth.")
            return
            
        # Get tree model and features
        tree_model = self.tree_results['tree_structure']['model']
        feature_names = list(self.tree_results['tree_structure']['feature_names']) if show_features else None
        
        # Get class names for classification
        class_names = None
        task_type = self.tree_results['parameters']['task_type']
        
        if task_type == 'classification':
            try:
                target_name = self.tree_results['tree_structure']['target_name']
                class_names = sorted(list(set(self.tree_results['raw_data']['y_train'].tolist())))
            except:
                class_names = None
                
        # Calculate figure size based on tree complexity
        nodes_count = tree_model.tree_.node_count
        width = 10 + (max_depth - min_depth) * 2
        height = 6 + min(20, (nodes_count // 10))
        
        # Clear canvas and set new size
        self.tree_canvas.fig.clear()
        self.tree_canvas.fig.set_size_inches(width, height)
        ax = self.tree_canvas.fig.subplots()
        
        # Plot tree with specified range
        try:
            plot_tree(
                tree_model,
                ax=ax,
                max_depth=max_depth if max_depth > 0 else None,
                feature_names=feature_names,
                class_names=class_names,
                filled=filled,
                rounded=True,
                precision=3,
                proportion=show_values,
                fontsize=fontsize
            )
            
            # Set title
            if min_depth == 0 and (max_depth == 0 or max_depth >= self.tree_results['tree_structure']['max_depth']):
                title = f"Full Decision Tree ({task_type.capitalize()})"
            else:
                title = f"Decision Tree - Depth {min_depth} to {max_depth} ({task_type.capitalize()})"
                
            ax.set_title(title)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error rendering tree: {str(e)}",
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
        
        # Update canvas
        self.tree_canvas.fig.tight_layout()
        self.tree_canvas.draw()


class ImportantNodesDialog(QDialog):
    """Dialog for important nodes analysis with node count selection"""
    def __init__(self, tree_results, parent=None):
        super().__init__(parent)
        self.tree_results = tree_results
        
        self.setWindowTitle("Important Nodes Analysis")
        self.setMinimumSize(800, 600)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI elements"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        if not self.tree_results or 'important_nodes' not in self.tree_results or not self.tree_results['important_nodes']:
            layout.addWidget(QLabel("Important nodes information not available."))
            return
            
        # Controls area
        controls_layout = QHBoxLayout()
        
        # Number of nodes selector
        controls_layout.addWidget(QLabel("Number of Important Nodes:"))
        self.node_count_spinner = QSpinBox()
        self.node_count_spinner.setRange(1, len(self.tree_results['important_nodes']))
        self.node_count_spinner.setValue(5)  # Default to 5 nodes
        self.node_count_spinner.valueChanged.connect(self.update_nodes_table)
        controls_layout.addWidget(self.node_count_spinner)
        
        # Add export button
        self.export_btn = QPushButton("Export Nodes Data")
        self.export_btn.clicked.connect(self.export_nodes_data)
        controls_layout.addWidget(self.export_btn)
        
        # Spacer to push controls to the left
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
            
        # Important nodes list
        nodes_group = QGroupBox("Important Decision Nodes")
        nodes_layout = QVBoxLayout()
        
        self.nodes_table = QTableWidget()
        self.nodes_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.nodes_table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        nodes_layout.addWidget(self.nodes_table)
        nodes_group.setLayout(nodes_layout)
        layout.addWidget(nodes_group)
        
        # Node details area
        details_group = QGroupBox("Node Details")
        details_layout = QVBoxLayout()
        
        self.node_details = QTextEdit()
        self.node_details.setReadOnly(True)
        
        details_layout.addWidget(self.node_details)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.view_detailed_btn = QPushButton("View Detailed Node Analysis")
        self.view_detailed_btn.clicked.connect(self.view_detailed_analysis)
        button_layout.addWidget(self.view_detailed_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Update table
        self.update_nodes_table()
        
        # Connect selection change
        self.nodes_table.itemSelectionChanged.connect(self.node_selected)
    
    def update_nodes_table(self):
        """Update important nodes table based on selected count"""
        # Get the number of nodes to display
        num_nodes = self.node_count_spinner.value()
        
        # Get the important nodes (limited by the selected count)
        important_nodes = self.tree_results['important_nodes'][:num_nodes]
        
        # Setup table
        self.nodes_table.setRowCount(len(important_nodes))
        self.nodes_table.setColumnCount(5)
        self.nodes_table.setHorizontalHeaderLabels(['Node ID', 'Importance', 'Depth', 'Samples', 'Feature'])
        
        # Fill table
        for i, node in enumerate(important_nodes):
            self.nodes_table.setItem(i, 0, QTableWidgetItem(str(node['id'])))
            self.nodes_table.setItem(i, 1, QTableWidgetItem(f"{node['importance']:.6f}"))
            self.nodes_table.setItem(i, 2, QTableWidgetItem(str(node['depth'])))
            self.nodes_table.setItem(i, 3, QTableWidgetItem(str(node['samples'])))
            
            feature = "Leaf" if node['is_leaf'] else (
                node['feature'] if isinstance(node['feature'], str) else 
                self.tree_results['tree_structure']['feature_names'][node['feature']] 
                if node['feature'] is not None and node['feature'] < len(self.tree_results['tree_structure']['feature_names']) 
                else f"Feature {node['feature']}")
            
            self.nodes_table.setItem(i, 4, QTableWidgetItem(feature))
        
        self.nodes_table.resizeColumnsToContents()
        
        # Select first row
        if self.nodes_table.rowCount() > 0:
            self.nodes_table.selectRow(0)
    
    def export_nodes_data(self):
        """Export important nodes data to file"""
        # Get the number of nodes to export
        num_nodes = self.node_count_spinner.value()
        
        # Get the important nodes (limited by the selected count)
        important_nodes = self.tree_results['important_nodes'][:num_nodes]
        
        # Ask for save location
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Export Important Nodes Data", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx);;Text Files (*.txt)")
        
        if not file_path:
            return  # User canceled
        
        try:
            # Determine file type and export accordingly
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == '.csv':
                self.export_to_csv(file_path, important_nodes)
            elif ext.lower() == '.xlsx':
                self.export_to_excel(file_path, important_nodes)
            elif ext.lower() == '.txt':
                self.export_to_text(file_path, important_nodes)
            else:
                # Default to CSV if extension not recognized
                if not file_path.endswith('.csv'):
                    file_path += '.csv'
                self.export_to_csv(file_path, important_nodes)
            
            QMessageBox.information(self, "Export Successful", 
                                   f"Important nodes data successfully exported to {file_path}")
        
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                                f"Error exporting important nodes data: {str(e)}")
    
    def export_to_csv(self, file_path, important_nodes):
        """Export important nodes data to CSV file"""
        import pandas as pd
        
        # Convert nodes to DataFrame
        nodes_data = []
        
        for node in important_nodes:
            # Get feature name
            feature = "Leaf" if node['is_leaf'] else (
                node['feature'] if isinstance(node['feature'], str) else 
                self.tree_results['tree_structure']['feature_names'][node['feature']] 
                if node['feature'] is not None and node['feature'] < len(self.tree_results['tree_structure']['feature_names']) 
                else f"Feature {node['feature']}")
            
            # Create decision path text
            decision_path = ""
            if node['decision_path']:
                for i, (parent, decision) in enumerate(node['decision_path'], 1):
                    decision_path += f"{i}. Node {parent}: {decision}\n"
            
            # Add node data
            nodes_data.append({
                'Node ID': node['id'],
                'Importance': node['importance'],
                'Depth': node['depth'],
                'Samples': node['samples'],
                'Feature': feature,
                'Is Leaf': node['is_leaf'],
                'Threshold': node['threshold'] if not node['is_leaf'] else "N/A",
                'Decision Path': decision_path
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(nodes_data)
        df.to_csv(file_path, index=False)
    
    def export_to_excel(self, file_path, important_nodes):
        """Export important nodes data to Excel file"""
        import pandas as pd
        
        # Create pandas ExcelWriter
        with pd.ExcelWriter(file_path) as writer:
            # Export nodes summary to first sheet
            nodes_summary = []
            
            for node in important_nodes:
                # Get feature name
                feature = "Leaf" if node['is_leaf'] else (
                    node['feature'] if isinstance(node['feature'], str) else 
                    self.tree_results['tree_structure']['feature_names'][node['feature']] 
                    if node['feature'] is not None and node['feature'] < len(self.tree_results['tree_structure']['feature_names']) 
                    else f"Feature {node['feature']}")
                
                # Add node summary
                nodes_summary.append({
                    'Node ID': node['id'],
                    'Importance': node['importance'],
                    'Depth': node['depth'],
                    'Samples': node['samples'],
                    'Feature': feature,
                    'Is Leaf': node['is_leaf'],
                    'Threshold': node['threshold'] if not node['is_leaf'] else "N/A"
                })
            
            # Create DataFrame and save to Excel
            summary_df = pd.DataFrame(nodes_summary)
            summary_df.to_excel(writer, sheet_name='Nodes Summary', index=False)
            
            # Create separate sheet for detailed path information
            paths_data = []
            
            for node in important_nodes:
                if node['decision_path']:
                    for i, (parent, decision) in enumerate(node['decision_path']):
                        paths_data.append({
                            'Node ID': node['id'],
                            'Step': i + 1,
                            'Parent Node': parent,
                            'Decision': decision
                        })
            
            if paths_data:
                paths_df = pd.DataFrame(paths_data)
                paths_df.to_excel(writer, sheet_name='Decision Paths', index=False)
            
            # Add tree parameters sheet
            params_data = []
            
            params_data.append({'Parameter': 'Task Type', 'Value': self.tree_results['parameters']['task_type']})
            params_data.append({'Parameter': 'Test Size', 'Value': self.tree_results['parameters']['test_size']})
            params_data.append({'Parameter': 'Random State', 'Value': self.tree_results['parameters']['random_state']})
            
            for param, value in self.tree_results['parameters']['tree_params'].items():
                params_data.append({'Parameter': param, 'Value': str(value)})
            
            params_df = pd.DataFrame(params_data)
            params_df.to_excel(writer, sheet_name='Tree Parameters', index=False)
    
    def export_to_text(self, file_path, important_nodes):
        """Export important nodes data to text file"""
        with open(file_path, 'w') as f:
            f.write(f"Important Nodes Analysis\n{'='*30}\n\n")
            
            # Write tree parameters
            f.write(f"Tree Parameters\n{'-'*30}\n")
            f.write(f"Task Type: {self.tree_results['parameters']['task_type']}\n")
            f.write(f"Test Size: {self.tree_results['parameters']['test_size']}\n")
            f.write(f"Random State: {self.tree_results['parameters']['random_state']}\n\n")
            
            # Write important nodes information
            f.write(f"Important Nodes\n{'-'*30}\n\n")
            
            for i, node in enumerate(important_nodes, 1):
                # Get feature name
                feature = "Leaf" if node['is_leaf'] else (
                    node['feature'] if isinstance(node['feature'], str) else 
                    self.tree_results['tree_structure']['feature_names'][node['feature']] 
                    if node['feature'] is not None and node['feature'] < len(self.tree_results['tree_structure']['feature_names']) 
                    else f"Feature {node['feature']}")
                
                f.write(f"Node {i} (ID: {node['id']})\n")
                f.write(f"  Importance: {node['importance']:.6f}\n")
                f.write(f"  Depth: {node['depth']}\n")
                f.write(f"  Samples: {node['samples']}\n")
                
                if node['is_leaf']:
                    f.write("  Type: Leaf Node\n")
                else:
                    f.write("  Type: Internal Node\n")
                    f.write(f"  Split Feature: {feature}\n")
                    f.write(f"  Split Threshold: {node['threshold']:.6f}\n")
                
                # Write decision path
                if node['decision_path']:
                    f.write("  Decision Path:\n")
                    for j, (parent, decision) in enumerate(node['decision_path'], 1):
                        f.write(f"    {j}. Node {parent}: {decision}\n")
                
                f.write("\n")
    
    def node_selected(self):
        """Handle node selection in table"""
        selected_rows = self.nodes_table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row_idx = selected_rows[0].row()
        
        # Make sure we don't go out of bounds
        if row_idx >= len(self.tree_results['important_nodes']):
            return
            
        node = self.tree_results['important_nodes'][row_idx]
        
        # Update details
        details = f"Node {node['id']} Details\n{'='*30}\n\n"
        
        details += f"Importance Score: {node['importance']:.6f}\n"
        details += f"Depth in Tree: {node['depth']}\n"
        details += f"Number of Samples: {node['samples']} ({node['samples']/self.tree_results['tree_structure']['model'].tree_.n_node_samples[0]*100:.2f}% of total)\n"
        
        if node['is_leaf']:
            details += "Type: Leaf Node\n"
            
            # Add value info
            task_type = self.tree_results['parameters']['task_type']
            tree = self.tree_results['tree_structure']['model'].tree_
            
            if task_type == 'regression':
                value = tree.value[node['id']][0][0]
                details += f"Predicted Value: {value:.6f}\n"
            else:
                # Classification - show class distribution
                class_counts = tree.value[node['id']][0]
                details += f"Class Distribution: {class_counts}\n"
        else:
            details += "Type: Decision Node\n"
            
            feature_name = self.tree_results['tree_structure']['feature_names'][node['feature']] if node['feature'] < len(self.tree_results['tree_structure']['feature_names']) else f"Feature {node['feature']}"
            details += f"Split Feature: {feature_name}\n"
            details += f"Split Threshold: {node['threshold']:.6f}\n"
        
        # Add decision path
        details += f"\nDecision Path:\n{'-'*20}\n"
        
        if not node['decision_path']:
            details += "This is the root node (no decision path).\n"
        else:
            for i, (parent, decision) in enumerate(node['decision_path'], 1):
                details += f"{i}. Node {parent}: {decision}\n"
        
        self.node_details.setText(details)
    
    def view_detailed_analysis(self):
        """Open detailed node analysis dialog for selected node"""
        selected_rows = self.nodes_table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row_idx = selected_rows[0].row()
        
        # Make sure we don't go out of bounds
        if row_idx >= len(self.tree_results['important_nodes']):
            return
            
        node_id = self.tree_results['important_nodes'][row_idx]['id']
        
        dialog = NodeDetailsDialog(self.tree_results, node_id, self)
        dialog.exec_()

#------------------------------------------------------------------------------
# Main Application Class
#------------------------------------------------------------------------------

class DecisionTreeAnalysisApp(QMainWindow):
    """Main application window for Decision Tree Analysis Tool"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.setWindowTitle("Decision Tree Analysis Tool")
        self.setGeometry(100, 100, 1280, 800)
        
        # Data storage
        self.df = None
        self.file_path = None
        self.tree_results = None
        
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
        
        # Advanced visualization button
        self.adv_viz_btn = QPushButton("Advanced Visualization")
        self.adv_viz_btn.clicked.connect(self.show_advanced_visualization)
        self.adv_viz_btn.setEnabled(False)
        toolbar_layout.addWidget(self.adv_viz_btn)
        
        # Important nodes button
        self.important_nodes_btn = QPushButton("Important Nodes")
        self.important_nodes_btn.clicked.connect(self.show_important_nodes)
        self.important_nodes_btn.setEnabled(False)
        toolbar_layout.addWidget(self.important_nodes_btn)
        
        # Tree pruning button
        self.pruning_btn = QPushButton("Tree Pruning")
        self.pruning_btn.clicked.connect(self.show_tree_pruning)
        self.pruning_btn.setEnabled(False)
        toolbar_layout.addWidget(self.pruning_btn)
        
        # Save results button
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        toolbar_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(toolbar_layout)
        
        # Progress bar for loading and training
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_data_preview_tab()
        self.create_feature_selection_tab()
        self.create_tree_config_tab()
        self.create_tree_visualization_tab()
        self.create_metrics_tab()
        self.create_node_analysis_tab()
        
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
        
        # Data summary button
        self.data_summary_btn = QPushButton("Data Summary")
        self.data_summary_btn.clicked.connect(self.show_data_summary)
        self.data_summary_btn.setEnabled(False)
        controls_layout.addWidget(self.data_summary_btn)
        
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
        
        # Search box for features
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.feature_search = QLineEdit()
        self.feature_search.textChanged.connect(self.filter_features)
        search_layout.addWidget(self.feature_search)
        left_layout.addLayout(search_layout)
        
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
        
        # Task type selection
        task_layout = QHBoxLayout()
        self.task_regression = QRadioButton("Regression")
        self.task_regression.setChecked(True)
        self.task_classification = QRadioButton("Classification")
        task_layout.addWidget(QLabel("Task Type:"))
        task_layout.addWidget(self.task_regression)
        task_layout.addWidget(self.task_classification)
        right_layout.addLayout(task_layout)
        
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
    def filter_features(self):
        """Filter feature list based on search text"""
        search_text = self.feature_search.text().lower()
        
        # Show all items if search text is empty
        if not search_text:
            for i in range(self.feature_list.count()):
                self.feature_list.item(i).setHidden(False)
            return
        
        # Hide items that don't match the search text
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if search_text not in item.text().lower():
                item.setHidden(True)
            else:
                item.setHidden(False)

    def create_tree_config_tab(self):
        """Create tab for decision tree configuration"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Split layout horizontally
        h_layout = QHBoxLayout()
        
        # Left side - Tree parameters
        left_group = QGroupBox("Decision Tree Parameters")
        left_layout = QFormLayout()
        left_group.setLayout(left_layout)
        
        # Max depth
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 50)
        self.max_depth.setValue(5)
        self.max_depth.setSpecialValueText("None")  # 1 means None (unlimited)
        left_layout.addRow("Max Depth:", self.max_depth)
        
        # Min samples split
        self.min_samples_split = QSpinBox()
        self.min_samples_split.setRange(2, 100)
        self.min_samples_split.setValue(2)
        left_layout.addRow("Min Samples Split:", self.min_samples_split)
        
        # Min samples leaf
        self.min_samples_leaf = QSpinBox()
        self.min_samples_leaf.setRange(1, 100)
        self.min_samples_leaf.setValue(1)
        left_layout.addRow("Min Samples Leaf:", self.min_samples_leaf)
        
        # Max features
        self.max_features = QComboBox()
        self.max_features.addItems(["None","auto", "sqrt", "log2", ])
        left_layout.addRow("Max Features:", self.max_features)
        
        # Criterion
        self.criterion = QComboBox()
        # Default to regression criteria, will update based on task type
        self.criterion.addItems(["squared_error", "friedman_mse", "absolute_error", "poisson"])
        left_layout.addRow("Criterion:", self.criterion)
        
        # CCP Alpha (for pruning)
        self.ccp_alpha = QDoubleSpinBox()
        self.ccp_alpha.setRange(0.0, 1.0)
        self.ccp_alpha.setValue(0.0)
        self.ccp_alpha.setDecimals(6)
        self.ccp_alpha.setSingleStep(0.00001)
        left_layout.addRow("Pruning Alpha:", self.ccp_alpha)
        
        # Right side - Training settings
        right_group = QGroupBox("Training Settings")
        right_layout = QFormLayout()
        right_group.setLayout(right_layout)
        
        # Test size
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.1, 0.5)
        self.test_size.setValue(0.2)
        self.test_size.setSingleStep(0.05)
        right_layout.addRow("Test Size:", self.test_size)
        
        # Random state
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
        
        # Add max important nodes option
        self.max_important_nodes = QSpinBox()
        self.max_important_nodes.setRange(1, 100)
        self.max_important_nodes.setValue(20)
        right_layout.addRow("Max Important Nodes:", self.max_important_nodes)
        
        # Add left and right groups to horizontal layout
        h_layout.addWidget(left_group, 50)
        h_layout.addWidget(right_group, 50)
        
        layout.addLayout(h_layout)
        
        # Train tree button
        self.train_btn = QPushButton("Train Decision Tree")
        self.train_btn.clicked.connect(self.train_decision_tree)
        layout.addWidget(self.train_btn)
        
        self.tabs.addTab(tab, "Tree Configuration")
        
        # Connect task type radios to criterion update
        self.task_regression.toggled.connect(self.update_criterion_options)
        self.task_classification.toggled.connect(self.update_criterion_options)
    def create_tree_visualization_tab(self):
        """Create tab for decision tree visualization"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Visualization settings
        settings_group = QGroupBox("Visualization Settings")
        settings_layout = QFormLayout()
        settings_group.setLayout(settings_layout)
        
        # Max depth to display
        self.vis_max_depth = QSpinBox()
        self.vis_max_depth.setRange(1, 20)
        self.vis_max_depth.setValue(5)
        settings_layout.addRow("Max Depth to Display:", self.vis_max_depth)
        
        # Fill colors
        self.filled_tree = QCheckBox("Filled (Color Nodes)")
        self.filled_tree.setChecked(True)
        settings_layout.addRow("", self.filled_tree)
        
        # Show feature names
        self.show_feature_names = QCheckBox("Show Feature Names")
        self.show_feature_names.setChecked(True)
        settings_layout.addRow("", self.show_feature_names)
        
        # Show values
        self.show_values = QCheckBox("Show Node Values")
        self.show_values.setChecked(True)
        settings_layout.addRow("", self.show_values)
        
        # Font size
        self.tree_fontsize = QSpinBox()
        self.tree_fontsize.setRange(6, 20)
        self.tree_fontsize.setValue(10)
        settings_layout.addRow("Font Size:", self.tree_fontsize)
        
        controls_layout.addWidget(settings_group)
        
        # Tree info
        info_group = QGroupBox("Tree Information")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)
        
        self.tree_info_text = QTextEdit()
        self.tree_info_text.setReadOnly(True)
        info_layout.addWidget(self.tree_info_text)
        
        controls_layout.addWidget(info_group)
        
        # Update button
        self.update_tree_viz_btn = QPushButton("Update Visualization")
        self.update_tree_viz_btn.clicked.connect(self.update_tree_visualization)
        controls_layout.addWidget(self.update_tree_viz_btn)
        
        layout.addLayout(controls_layout)
        
        # Tree visualization
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        tree_widget = QWidget()
        tree_layout = QVBoxLayout()
        
        self.tree_canvas = MatplotlibCanvas(self, width=12, height=8)
        self.tree_toolbar = NavigationToolbar(self.tree_canvas, self)
        tree_layout.addWidget(self.tree_toolbar)
        tree_layout.addWidget(self.tree_canvas)
        
        tree_widget.setLayout(tree_layout)
        scroll_area.setWidget(tree_widget)
        layout.addWidget(scroll_area)
        
        # Text representation
        text_group = QGroupBox("Tree Text Representation")
        text_layout = QVBoxLayout()
        text_group.setLayout(text_layout)
        
        self.tree_text = QTextEdit()
        self.tree_text.setReadOnly(True)
        self.tree_text.setFont(QFont("Courier New", 10))
        text_layout.addWidget(self.tree_text)
        
        layout.addWidget(text_group)
        
        self.tabs.addTab(tab, "Tree Visualization")
    def create_metrics_tab(self):
        """Create tab for metrics and performance evaluation"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Dataset selector for metrics
        selector_layout = QHBoxLayout()
        
        selector_layout.addWidget(QLabel("Show Metrics for:"))
        self.metrics_dataset = QComboBox()
        self.metrics_dataset.addItems(["Both", "Training Set", "Test Set"])
        self.metrics_dataset.currentIndexChanged.connect(self.update_metrics_display)
        selector_layout.addWidget(self.metrics_dataset)
        
        selector_layout.addStretch()
        
        layout.addLayout(selector_layout)
        
        # Error metrics table
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()
        metrics_group.setLayout(metrics_layout)
        
        self.metrics_table = QTableWidget()
        metrics_layout.addWidget(self.metrics_table)
        
        layout.addWidget(metrics_group)
        
        # Feature importance
        importance_group = QGroupBox("Feature Importance")
        importance_layout = QVBoxLayout()
        importance_group.setLayout(importance_layout)
        
        # Add importance threshold control
        imp_controls = QHBoxLayout()
        imp_controls.addWidget(QLabel("Show Features with Importance >"))
        self.importance_threshold = QDoubleSpinBox()
        self.importance_threshold.setRange(0.0, 1.0)
        self.importance_threshold.setValue(0.01)
        self.importance_threshold.setSingleStep(0.01)
        self.importance_threshold.valueChanged.connect(self.update_feature_importance)
        imp_controls.addWidget(self.importance_threshold)
        imp_controls.addStretch()
        
        importance_layout.addLayout(imp_controls)
        
        self.feature_importance_canvas = MatplotlibCanvas(self, width=10, height=5)
        self.feature_importance_toolbar = NavigationToolbar(self.feature_importance_canvas, self)
        importance_layout.addWidget(self.feature_importance_toolbar)
        importance_layout.addWidget(self.feature_importance_canvas)
        
        layout.addWidget(importance_group)
        
        # Cross-validation results
        cv_group = QGroupBox("Cross-Validation Results")
        cv_layout = QVBoxLayout()
        cv_group.setLayout(cv_layout)
        
        self.cv_text = QTextEdit()
        self.cv_text.setReadOnly(True)
        cv_layout.addWidget(self.cv_text)
        
        layout.addWidget(cv_group)
        
        self.tabs.addTab(tab, "Performance Metrics")
    
    def create_node_analysis_tab(self):
        """Create tab for node-by-node analysis"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Node navigation
        nav_group = QGroupBox("Node Navigation")
        nav_layout = QHBoxLayout()
        nav_group.setLayout(nav_layout)
        
        # Node selector
        nav_layout.addWidget(QLabel("Node ID:"))
        self.node_selector = QSpinBox()
        self.node_selector.setMinimum(0)
        self.node_selector.setMaximum(0)  # Will update when tree is trained
        self.node_selector.valueChanged.connect(self.node_selected)
        nav_layout.addWidget(self.node_selector)
        
        # Previous/Next buttons
        self.prev_node_btn = QPushButton("←")
        self.prev_node_btn.clicked.connect(lambda: self.node_selector.setValue(self.node_selector.value() - 1))
        self.prev_node_btn.setToolTip("Go to previous node")
        self.prev_node_btn.setFixedWidth(40)
        
        self.next_node_btn = QPushButton("→")
        self.next_node_btn.clicked.connect(lambda: self.node_selector.setValue(self.node_selector.value() + 1))
        self.next_node_btn.setToolTip("Go to next node")
        self.next_node_btn.setFixedWidth(40)
        
        nav_layout.addWidget(self.prev_node_btn)
        nav_layout.addWidget(self.next_node_btn)
        
        # Navigation buttons for tree traversal
        self.parent_node_btn = QPushButton("Parent")
        self.parent_node_btn.clicked.connect(self.go_to_parent_node)
        self.parent_node_btn.setToolTip("Go to parent node")
        
        self.left_child_btn = QPushButton("Left Child")
        self.left_child_btn.clicked.connect(self.go_to_left_child)
        self.left_child_btn.setToolTip("Go to left child node")
        
        self.right_child_btn = QPushButton("Right Child")
        self.right_child_btn.clicked.connect(self.go_to_right_child)
        self.right_child_btn.setToolTip("Go to right child node")
        
        nav_layout.addWidget(self.parent_node_btn)
        nav_layout.addWidget(self.left_child_btn)
        nav_layout.addWidget(self.right_child_btn)
        
        controls_layout.addWidget(nav_group)
        
        # Detailed analysis button
        self.node_details_btn = QPushButton("Detailed Node Analysis")
        self.node_details_btn.clicked.connect(self.show_node_details)
        controls_layout.addWidget(self.node_details_btn)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Split view for node info and visualization
        splitter = QSplitter(Qt.Horizontal)
        
        # Node info panel
        info_panel = QWidget()
        info_layout = QVBoxLayout()
        info_panel.setLayout(info_layout)
        
        # Node information
        node_info_group = QGroupBox("Node Information")
        node_info_layout = QVBoxLayout()
        node_info_group.setLayout(node_info_layout)
        
        self.node_info_text = QTextEdit()
        self.node_info_text.setReadOnly(True)
        node_info_layout.addWidget(self.node_info_text)
        
        info_layout.addWidget(node_info_group)
        
        # Decision path
        path_group = QGroupBox("Decision Path")
        path_layout = QVBoxLayout()
        path_group.setLayout(path_layout)
        
        self.decision_path_text = QTextEdit()
        self.decision_path_text.setReadOnly(True)
        self.decision_path_text.setFont(QFont("Courier New", 10))
        path_layout.addWidget(self.decision_path_text)
        
        info_layout.addWidget(path_group)
        
        # Add info panel to splitter
        splitter.addWidget(info_panel)
        
        # Visualization panel
        viz_panel = QWidget()
        viz_layout = QVBoxLayout()
        viz_panel.setLayout(viz_layout)
        
        # Sample distribution in node
        distribution_group = QGroupBox("Sample Distribution in Node")
        distribution_layout = QVBoxLayout()
        distribution_group.setLayout(distribution_layout)
        
        self.node_dist_canvas = MatplotlibCanvas(self, width=8, height=5)
        self.node_dist_toolbar = NavigationToolbar(self.node_dist_canvas, self)
        distribution_layout.addWidget(self.node_dist_toolbar)
        distribution_layout.addWidget(self.node_dist_canvas)
        
        viz_layout.addWidget(distribution_group)
        
        # Node position in tree visualization
        node_pos_group = QGroupBox("Node Position in Tree")
        node_pos_layout = QVBoxLayout()
        node_pos_group.setLayout(node_pos_layout)
        
        self.node_pos_canvas = MatplotlibCanvas(self, width=8, height=5)
        self.node_pos_toolbar = NavigationToolbar(self.node_pos_canvas, self)
        node_pos_layout.addWidget(self.node_pos_toolbar)
        node_pos_layout.addWidget(self.node_pos_canvas)
        
        viz_layout.addWidget(node_pos_group)
        
        # Add visualization panel to splitter
        splitter.addWidget(viz_panel)
        
        # Set initial sizes
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
        self.tabs.addTab(tab, "Node Analysis")
    
    def go_to_parent_node(self):
        """Navigate to parent of current node"""
        if not self.tree_results:
            return
            
        current_node = self.node_selector.value()
        if current_node == 0:  # Root has no parent
            return
            
        tree = self.tree_results['tree_structure']['model'].tree_
        
        # Find parent
        for i in range(tree.node_count):
            if tree.children_left[i] == current_node or tree.children_right[i] == current_node:
                self.node_selector.setValue(i)
                break
        
    def go_to_left_child(self):
        """Navigate to left child of current node"""
        if not self.tree_results:
            return
            
        current_node = self.node_selector.value()
        tree = self.tree_results['tree_structure']['model'].tree_
        
        left_child = tree.children_left[current_node]
        if left_child != tree.children_right[current_node]:  # Not a leaf
            self.node_selector.setValue(left_child)
    
    def go_to_right_child(self):
        """Navigate to right child of current node"""
        if not self.tree_results:
            return
            
        current_node = self.node_selector.value()
        tree = self.tree_results['tree_structure']['model'].tree_
        
        right_child = tree.children_right[current_node]
        if right_child != tree.children_left[current_node]:  # Not a leaf
            self.node_selector.setValue(right_child)
    
    def show_node_details(self):
        """Show detailed node analysis dialog"""
        if not self.tree_results:
            return
            
        current_node = self.node_selector.value()
        dialog = NodeDetailsDialog(self.tree_results, current_node, self)
        dialog.exec_()
    #--------------------------------------------------------------------------
    # Data handling methods
    #--------------------------------------------------------------------------
    
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
        
        # Enable tabs and data summary button
        self.tabs.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.data_summary_btn.setEnabled(True)
        
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
            if not item.isHidden():  # Only select visible items if search is active
                item.setCheckState(Qt.Checked)
    
    def deselect_all_features(self):
        """Deselect all features in the feature list"""
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            item.setCheckState(Qt.Unchecked)
    
    def update_criterion_options(self):
        """Update criterion options based on selected task type"""
        current_text = self.criterion.currentText()
        self.criterion.clear()
        
        if self.task_regression.isChecked():
            self.criterion.addItems(["squared_error", "friedman_mse", "absolute_error", "poisson"])
        else:
            self.criterion.addItems(["gini", "entropy", "log_loss"])
        
        # Try to restore the previous selection if it's valid for the current task
        index = self.criterion.findText(current_text)
        if index >= 0:
            self.criterion.setCurrentIndex(index)
    
    def target_variable_changed(self, target_var):
        """Handle target variable change"""
        if not target_var or self.df is None:
            self.target_info_label.setText("No target variable selected")
            return
        
        # Update target info
        self.target_info_label.setText(f"Selected Target: {target_var}")
        
        # Analyze basic statistics
        if target_var in self.df.columns:
            # Determine if target is likely categorical
            unique_values = self.df[target_var].nunique()
            if unique_values <= 10 or not pd.api.types.is_numeric_dtype(self.df[target_var]):
                # Likely categorical
                self.task_classification.setChecked(True)
                self.update_criterion_options()
                
                # Display value counts
                value_counts = self.df[target_var].value_counts()
                stats_text = "Value Counts:\n"
                for val, count in value_counts.items():
                    stats_text += f"{val}: {count} ({count/len(self.df)*100:.1f}%)\n"
                
                self.target_info_label.setText(f"Selected Target: {target_var}\n\n{stats_text}")
            else:
                # Likely numeric/continuous
                self.task_regression.setChecked(True)
                self.update_criterion_options()
                
                # Display numeric statistics
                stats = self.df[target_var].describe()
                stats_text = "\n".join([f"{idx}: {val:.4f}" for idx, val in stats.items()])
                self.target_info_label.setText(f"Selected Target: {target_var}\n\nStatistics:\n{stats_text}")
        
        # Create distribution plot
        self.plot_target_distribution(target_var)
    
    def plot_target_distribution(self, target_var):
        """Plot distribution of the target variable"""
        if self.df is None or target_var not in self.df.columns:
            return
        
        # Clear the canvas
        self.target_dist_canvas.fig.clear()
        ax = self.target_dist_canvas.fig.add_subplot(111)
        
        # Check if target appears categorical
        unique_values = self.df[target_var].nunique()
        if unique_values <= 10 or not pd.api.types.is_numeric_dtype(self.df[target_var]):
            # For categorical variables, create a count plot
            value_counts = self.df[target_var].value_counts()
            ax.bar(value_counts.index.astype(str), value_counts.values)
            ax.set_title(f"Value Counts for {target_var}")
            ax.set_xlabel(target_var)
            ax.set_ylabel("Count")
            
            # Rotate labels if there are several categories
            if unique_values > 4:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            # For numeric variables, create histogram with density plot
            sns.histplot(self.df[target_var].dropna(), kde=True, ax=ax)
            
            # Add mean and median lines
            mean_val = self.df[target_var].mean()
            median_val = self.df[target_var].median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
            ax.legend()
            
            ax.set_title(f"Distribution of {target_var}")
            ax.set_xlabel(target_var)
            ax.set_ylabel("Frequency")
        
        # Draw the plot
        self.target_dist_canvas.fig.tight_layout()
        self.target_dist_canvas.draw()
    #--------------------------------------------------------------------------
    # Analysis methods
    #--------------------------------------------------------------------------
    def analyze_target_variable(self):
        """Create a detailed analysis of the target variable"""
        target_var = self.target_var.currentText()
        
        if not target_var or self.df is None:
            QMessageBox.warning(self, "Error", "No target variable selected")
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
        
        # Create correlation tab
        corr_tab = QWidget()
        corr_layout = QVBoxLayout()
        corr_canvas = MatplotlibCanvas(width=6, height=4)
        corr_toolbar = NavigationToolbar(corr_canvas, dialog)
        corr_layout.addWidget(corr_toolbar)
        corr_layout.addWidget(corr_canvas)
        corr_tab.setLayout(corr_layout)
        
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
        tabs.addTab(corr_tab, "Feature Correlations")
        tabs.addTab(stats_tab, "Statistics")
        
        layout.addWidget(tabs)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        
        # Create plots and statistics
        unique_values = self.df[target_var].nunique()
        
        # Distribution plot
        dist_canvas.fig.clear()
        ax1 = dist_canvas.fig.add_subplot(111)
        
        if unique_values <= 10 or not pd.api.types.is_numeric_dtype(self.df[target_var]):
            # Categorical target
            value_counts = self.df[target_var].value_counts()
            ax1.bar(value_counts.index.astype(str), value_counts.values)
            ax1.set_title(f"Value Counts for {target_var}")
            ax1.set_xlabel(target_var)
            ax1.set_ylabel("Count")
            
            # Rotate labels if there are several categories
            if unique_values > 4:
                plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
                
            # Add percentage labels
            total = value_counts.sum()
            for i, (cat, count) in enumerate(value_counts.items()):
                percentage = count / total * 100
                ax1.text(i, count, f"{percentage:.1f}%", ha='center', va='bottom')
        else:
            # Numeric target
            sns.histplot(self.df[target_var].dropna(), kde=True, ax=ax1, bins=30)
            
            # Add mean and median lines
            mean_val = self.df[target_var].mean()
            median_val = self.df[target_var].median()
            ax1.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax1.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
            ax1.legend()
            
            ax1.set_title(f"Distribution of {target_var}")
            ax1.set_xlabel(target_var)
            ax1.set_ylabel("Frequency")
        
        dist_canvas.fig.tight_layout()
        dist_canvas.draw()
        
        # Correlation with features
        corr_canvas.fig.clear()
        ax2 = corr_canvas.fig.add_subplot(111)
        
        # Get numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_var in numeric_cols:
            # Calculate correlations with target
            correlations = []
            for col in numeric_cols:
                if col != target_var:
                    corr = self.df[[col, target_var]].corr().iloc[0, 1]
                    correlations.append((col, corr))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            if correlations:
                # Take top 15 features
                features, corr_values = zip(*correlations[:15])
                
                # Create horizontal bar chart
                y_pos = np.arange(len(features))
                bars = ax2.barh(y_pos, corr_values)
                
                # Color bars based on correlation value
                for i, bar in enumerate(bars):
                    if corr_values[i] > 0:
                        bar.set_color('green')
                    else:
                        bar.set_color('red')
                
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(features)
                ax2.set_xlabel("Correlation with Target")
                ax2.set_title(f"Top Feature Correlations with {target_var}")
                
                # Add correlation values
                for i, v in enumerate(corr_values):
                    ax2.text(v + (0.01 if v >= 0 else -0.01), 
                        i, 
                        f"{v:.3f}", 
                        va='center',
                        ha='left' if v >= 0 else 'right',
                        fontweight='bold')
            else:
                ax2.text(0.5, 0.5, "No features to correlate with target",
                    ha='center', va='center', fontsize=12)
        else:
            ax2.text(0.5, 0.5, "Correlation requires numeric target",
                ha='center', va='center', fontsize=12)
        
        corr_canvas.fig.tight_layout()
        corr_canvas.draw()
    
        # Statistics
        if pd.api.types.is_numeric_dtype(self.df[target_var]):
            # Numeric statistics
            data = self.df[target_var]
            basic_stats = data.describe()
            
            # Additional statistics
            skewness = data.skew()
            kurtosis = data.kurtosis()
            missing = data.isnull().sum()
            missing_pct = missing / len(data) * 100
            
            # Shapiro-Wilk test for normality
            if len(data.dropna()) >= 3 and len(data.dropna()) <= 5000:
                try:
                    shapiro_test = stats.shapiro(data.dropna())
                    shapiro_pvalue = shapiro_test[1]
                    normality_test = f"Shapiro-Wilk Test p-value: {shapiro_pvalue:.6f}"
                    if shapiro_pvalue < 0.05:
                        normality_test += " (Data is not normally distributed)"
                    else:
                        normality_test += " (Data is normally distributed)"
                except:
                    normality_test = "Shapiro-Wilk Test: Could not be performed"
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
            
            # Add quartiles and percentiles
            stats_text.append("\nPercentiles:")
            percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            for p in percentiles:
                val = data.quantile(p)
                stats_text.append(f"  {p*100:3.0f}th: {val:.6f}")
        else:
            # Categorical statistics
            value_counts = self.df[target_var].value_counts()
            
            # Format statistics text
            stats_text.append(f"Statistical Analysis for {target_var}\n")
            stats_text.append("=" * 50 + "\n")
            stats_text.append("Value Counts:")
            for val, count in value_counts.items():
                stats_text.append(f"  {val}: {count} ({count/len(self.df)*100:.2f}%)")
            
            missing = self.df[target_var].isnull().sum()
            missing_pct = missing / len(self.df) * 100
            stats_text.append(f"\nMissing Values: {missing} ({missing_pct:.2f}%)")
            stats_text.append(f"\nUnique Values: {unique_values}")
            
            # Add chi-square tests for association with other categorical variables
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(cat_cols) > 1:
                stats_text.append("\nChi-Square Tests for Association:")
                for col in cat_cols:
                    if col != target_var:
                        try:
                            contingency = pd.crosstab(self.df[target_var], self.df[col])
                            chi2, p, dof, expected = stats.chi2_contingency(contingency)
                            stats_text.append(f"  {col}: chi2={chi2:.2f}, p={p:.6f}")
                        except:
                            pass
        
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
    
    def get_tree_parameters(self):
        """Get decision tree parameters from UI"""
        params = {}
        
        # Get max depth (None if value is 1)
        max_depth = self.max_depth.value()
        params['max_depth'] = None if max_depth == 1 else max_depth
        
        # Get other parameters
        params['min_samples_split'] = self.min_samples_split.value()
        params['min_samples_leaf'] = self.min_samples_leaf.value()
        
        # Max features
        max_features_text = self.max_features.currentText()
        if max_features_text == "None":
            params['max_features'] = None
        else:
            params['max_features'] = max_features_text
        
        # Criterion
        params['criterion'] = self.criterion.currentText()
        
        # Random state
        params['random_state'] = self.random_state.value()
        
        # CCP alpha for pruning
        params['ccp_alpha'] = self.ccp_alpha.value()
        
        return params
    
    def train_decision_tree(self):
        """Train decision tree with selected settings"""
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
        
        # Get task type
        task_type = "classification" if self.task_classification.isChecked() else "regression"
        
        # Check if target is compatible with task type
        if task_type == "regression" and not pd.api.types.is_numeric_dtype(self.df[target_variable]):
            QMessageBox.warning(self, "Error", "Target variable must be numeric for regression")
            return
        
        # Get tree parameters
        tree_params = self.get_tree_parameters()
        
        # Get other training settings
        test_size = self.test_size.value()
        random_state = self.random_state.value()
        scaling_method = self.scaling_method.currentText()
        use_cross_val = self.use_cross_val.isChecked()
        cv_folds = self.cv_folds.value()
        max_important_nodes = self.max_important_nodes.value()
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Disable UI during training
        self.tabs.setEnabled(False)
        self.train_btn.setEnabled(False)
        
        # Start tree training thread
        self.tree_trainer = TreeTrainer(
            self.df, input_features, target_variable, task_type, test_size, random_state,
            tree_params, scaling_method, use_cross_val, cv_folds, max_important_nodes
        )
        self.tree_trainer.progress_update.connect(self.update_progress)
        self.tree_trainer.status_update.connect(self.update_status)
        self.tree_trainer.error_raised.connect(self.show_error)
        self.tree_trainer.training_finished.connect(self.tree_trained)
        self.tree_trainer.start()

    def tree_trained(self, results):
        """Handle when tree training is complete"""
        self.tree_results = results
        self.progress_bar.setVisible(False)
        
        # Enable UI
        self.tabs.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.adv_viz_btn.setEnabled(True)
        self.important_nodes_btn.setEnabled(True)
        self.pruning_btn.setEnabled(True)
        
        # Update all visualizations and metrics
        self.update_tree_info()
        self.update_tree_visualization()
        self.update_metrics_display()
        self.update_node_selector()
        self.update_error_analysis()
        
        # Switch to tree visualization tab
        self.tabs.setCurrentIndex(3)  # Tree visualization tab
        
        self.statusBar().showMessage("Decision tree trained successfully!")
    
    def update_tree_info(self):
        """Update tree information display"""
        if not self.tree_results:
            return
        
        tree_structure = self.tree_results['tree_structure']
        tree_params = self.tree_results['parameters']['tree_params']
        
        # Clear existing info
        self.tree_info_text.clear()
        
        # Display tree information
        self.tree_info_text.append(f"Decision Tree Information\n{'='*30}\n")
        self.tree_info_text.append(f"Task Type: {self.tree_results['parameters']['task_type'].capitalize()}")
        self.tree_info_text.append(f"Number of Nodes: {tree_structure['num_nodes']}")
        self.tree_info_text.append(f"Number of Leaves: {tree_structure['num_leaves']}")
        self.tree_info_text.append(f"Tree Depth: {tree_structure['max_depth']}")
        
        # Parameters
        self.tree_info_text.append(f"\nTree Parameters:\n{'-'*20}")
        for param, value in tree_params.items():
            self.tree_info_text.append(f"{param}: {value}")
        
        # Update tree text representation
        self.tree_text.setText(tree_structure['tree_text'])
    
    def update_tree_visualization(self):
        """Update the tree visualization"""
        if not self.tree_results:
            return
        
        tree_structure = self.tree_results['tree_structure']
        task_type = self.tree_results['parameters']['task_type']
        
        # Get visualization parameters
        max_depth = self.vis_max_depth.value()
        filled = self.filled_tree.isChecked()
        feature_names = self.show_feature_names.isChecked()
        show_values = self.show_values.isChecked()
        fontsize = self.tree_fontsize.value()
        
        # Clear the canvas
        self.tree_canvas.fig.clear()
        ax = self.tree_canvas.fig.add_subplot(111)
        
        # Get feature and class names
        feature_names_list = list(tree_structure['feature_names']) if feature_names else None
        class_names = None  # Only for classification
        
        if task_type == 'classification':
            # Try to get unique class labels
            try:
                target_name = tree_structure['target_name']
                class_names = sorted(list(set(self.tree_results['raw_data']['y_train'].tolist())))
            except:
                class_names = None
        
        # Plot the tree
        try:
            plot_tree(
                tree_structure['model'],
                ax=ax,
                max_depth=max_depth,
                filled=filled,
                feature_names=feature_names_list,
                class_names=class_names if task_type == 'classification' else None,
                rounded=True,
                precision=3,
                proportion=show_values,
                fontsize=fontsize
            )
            ax.set_title(f"Decision Tree ({task_type.capitalize()})")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error rendering tree: {str(e)}",
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
        
        # Update the canvas
        self.tree_canvas.fig.tight_layout()
        self.tree_canvas.draw()
    
    #--------------------------------------------------------------------------
    # Metrics and visualization methods
    #--------------------------------------------------------------------------
    
    def update_metrics_display(self):
        """Update the metrics display"""
        if not self.tree_results:
            return
        
        metrics = self.tree_results['metrics']
        task_type = self.tree_results['parameters']['task_type']
        
        # Get selected display option
        display_option = self.metrics_dataset.currentText()
        
        # Clear the table
        self.metrics_table.clear()
        
        if display_option == "Both":
            # Setup side-by-side comparison
            self.metrics_table.setColumnCount(3)
            self.metrics_table.setHorizontalHeaderLabels(["Metric", "Training Set", "Test Set"])
            
            # Get common metrics for both sets
            metrics_to_show = []
            
            if task_type == 'regression':
                for metric in metrics['train'].keys():
                    if metric in metrics['test'] and isinstance(metrics['train'][metric], (int, float, np.number)) and not np.isnan(metrics['train'][metric]):
                        metrics_to_show.append(metric)
            else:
                for metric in metrics['train'].keys():
                    if metric not in ['Confusion Matrix', 'Classification Report'] and metric in metrics['test'] and isinstance(metrics['train'][metric], (int, float, np.number)):
                        metrics_to_show.append(metric)
            
            # Sort alphabetically
            metrics_to_show.sort()
            
            # Setup rows
            self.metrics_table.setRowCount(len(metrics_to_show))
            
            # Fill table
            for i, metric in enumerate(metrics_to_show):
                self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
                self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{metrics['train'][metric]:.6f}"))
                self.metrics_table.setItem(i, 2, QTableWidgetItem(f"{metrics['test'][metric]:.6f}"))
                
                # Add visual indicators
                if metric in ['R²', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Explained Variance']:
                    # Higher is better
                    if metrics['train'][metric] > 0.8:
                        self.metrics_table.item(i, 1).setBackground(QBrush(QColor(200, 255, 200)))  # Light green
                    if metrics['test'][metric] > 0.8:
                        self.metrics_table.item(i, 2).setBackground(QBrush(QColor(200, 255, 200)))  # Light green
                    
                    if metrics['train'][metric] - metrics['test'][metric] > 0.1:
                        # Potential overfitting
                        self.metrics_table.item(i, 1).setBackground(QBrush(QColor(255, 255, 200)))  # Light yellow
                        self.metrics_table.item(i, 2).setBackground(QBrush(QColor(255, 200, 200)))  # Light red
                        
                elif 'Error' in metric or 'MSE' in metric or 'RMSE' in metric or 'MAE' in metric:
                    # Lower is better
                    # Highlight if test error is significantly higher than train error
                    if metrics['test'][metric] / max(metrics['train'][metric], 0.0001) > 1.2:
                        self.metrics_table.item(i, 1).setBackground(QBrush(QColor(255, 255, 200)))  # Light yellow
                        self.metrics_table.item(i, 2).setBackground(QBrush(QColor(255, 200, 200)))  # Light red
            
        else:
            # Show metrics for just one dataset
            dataset = "train" if display_option == "Training Set" else "test"
            
            # Setup columns
            self.metrics_table.setColumnCount(2)
            self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
            
            # Get metrics to display
            metrics_to_show = []
            
            if task_type == 'regression':
                for metric, value in metrics[dataset].items():
                    if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                        metrics_to_show.append((metric, value))
            else:
                for metric, value in metrics[dataset].items():
                    if metric not in ['Confusion Matrix', 'Classification Report'] and isinstance(value, (int, float, np.number)):
                        metrics_to_show.append((metric, value))
            
            # Sort alphabetically
            metrics_to_show.sort(key=lambda x: x[0])
            
            # Setup rows
            self.metrics_table.setRowCount(len(metrics_to_show))
            
            # Fill table
            for i, (metric, value) in enumerate(metrics_to_show):
                self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
                self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{value:.6f}"))
                
                # Add visual indicators
                if metric in ['R²', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Explained Variance']:
                    # Higher is better
                    if value > 0.8:
                        self.metrics_table.item(i, 1).setBackground(QBrush(QColor(200, 255, 200)))  # Light green
                elif 'Error' in metric or 'MSE' in metric or 'RMSE' in metric or 'MAE' in metric:
                    # Lower is better
                    # Simple color scale
                    if dataset == 'test' and task_type == 'regression':
                        benchmark = metrics['train'].get(metric, 0)
                        if value / max(benchmark, 0.0001) > 1.5:
                            self.metrics_table.item(i, 1).setBackground(QBrush(QColor(255, 200, 200)))  # Light red
        
        # Resize columns
        self.metrics_table.resizeColumnsToContents()
        
        # Update feature importance visualization
        self.update_feature_importance()
        
        # Update cross-validation results
        self.display_cv_results()
    

    def update_feature_importance(self):
        """Display feature importance visualization"""
        if not self.tree_results or 'feature_importance' not in self.tree_results:
            return
        
        feature_importance = self.tree_results['feature_importance']
        features = feature_importance['features']
        importance = feature_importance['importance']
        
        # Get threshold
        threshold = self.importance_threshold.value()
        
        # Create feature importance pairs
        importance_pairs = [(feature, imp) for feature, imp in zip(features, importance)]
        
        # Filter by threshold and sort
        importance_pairs = [(f, i) for f, i in importance_pairs if i >= threshold]
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Clear the canvas
        self.feature_importance_canvas.fig.clear()
        ax = self.feature_importance_canvas.fig.add_subplot(111)
        
        if not importance_pairs:
            ax.text(0.5, 0.5, f"No features with importance >= {threshold}",
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            self.feature_importance_canvas.draw()
            return
        
        # Unpack the top features
        feature_names, importances = zip(*importance_pairs)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, importances, align='center')
        
        # Color code bars
        for i, bar in enumerate(bars):
            # Color gradient by importance
            color_intensity = min(importances[i] * 5, 0.9)  # Scale for better visibility
            bar.set_color((0, 0.5, color_intensity))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        
        # Add importance values to bars
        for i, v in enumerate(importances):
            ax.text(v + 0.01, i, f"{v:.4f}", va='center')
        
        # Add threshold line
        if threshold > 0:
            ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.7,
                     label=f'Threshold: {threshold:.2f}')
            ax.legend()
        
        # Update the canvas
        self.feature_importance_canvas.fig.tight_layout()
        self.feature_importance_canvas.draw()
    
    def display_cv_results(self):
        """Display cross-validation results"""
        if not self.tree_results or 'cv_results' not in self.tree_results:
            self.cv_text.setText("Cross-validation results not available.")
            return
        
        cv_results = self.tree_results['cv_results']
        task_type = self.tree_results['parameters']['task_type']
        
        # Clear existing text
        self.cv_text.clear()
        
        # Display CV results
        self.cv_text.append(f"Cross-Validation Results\n{'='*30}\n")
        
        # Metric name depends on task type
        metric_name = "R²" if task_type == 'regression' else "F1 Score"
        
        # Mean and std
        self.cv_text.append(f"<b>Mean {metric_name}:</b> {cv_results['mean']:.6f}")
        self.cv_text.append(f"<b>Standard Deviation:</b> {cv_results['std']:.6f}")
        
        # Individual fold scores
        self.cv_text.append(f"\n<b>Scores by Fold:</b>")
        for i, score in enumerate(cv_results['scores'], 1):
            self.cv_text.append(f"Fold {i}: {score:.6f}")
            
        # Add interpretation
        self.cv_text.append("\n<b>Interpretation:</b>")
        
        if task_type == 'regression':
            if cv_results['mean'] > 0.7:
                self.cv_text.append("The model shows good predictive performance across folds.")
            elif cv_results['mean'] > 0.5:
                self.cv_text.append("The model shows moderate predictive performance.")
            else:
                self.cv_text.append("The model shows weak predictive performance.")
                
            if cv_results['std'] > 0.1:
                self.cv_text.append("High variation across folds indicates potential instability.")
            else:
                self.cv_text.append("Low variation across folds indicates stable performance.")
        else:
            if cv_results['mean'] > 0.8:
                self.cv_text.append("The model shows good classification performance across folds.")
            elif cv_results['mean'] > 0.6:
                self.cv_text.append("The model shows moderate classification performance.")
            else:
                self.cv_text.append("The model shows weak classification performance.")
                
            if cv_results['std'] > 0.1:
                self.cv_text.append("High variation across folds indicates potential instability.")
            else:
                self.cv_text.append("Low variation across folds indicates stable performance.")

    

    def update_node_selector(self):
        """Update the node selector range based on trained tree"""
        if not self.tree_results:
            return
        
        # Get number of nodes
        num_nodes = self.tree_results['tree_structure']['num_nodes']
        
        # Update node selector range
        self.node_selector.setMaximum(num_nodes - 1)  # 0-indexed
        
        # Select root node (0)
        self.node_selector.setValue(0)
        
        # Trigger node selection update
        self.node_selected(0)


    def node_selected(self, node_id):
        """Handle node selection change"""
        if not self.tree_results or not hasattr(self.tree_results['tree_structure']['model'], 'tree_'):
            return
        
        # Get tree model
        tree = self.tree_results['tree_structure']['model'].tree_
        
        # Check if node ID is valid
        if node_id < 0 or node_id >= tree.node_count:
            self.node_info_text.setText(f"Invalid node ID: {node_id}")
            return
        
        # Update node info
        self.display_node_info(node_id, tree)
        
        # Update node distribution visualization
        self.visualize_node_distribution(node_id, tree)
        
        # Update node position in tree
        self.visualize_node_position(node_id, tree)
        
        # Update decision path
        self.display_decision_path(node_id, tree)
        
        # Update nav button states
        self.update_node_nav_buttons(node_id, tree)
    

    def update_node_nav_buttons(self, node_id, tree):
        """Update navigation button states based on current node"""
        # Root has no parent
        self.parent_node_btn.setEnabled(node_id != 0)
        
        # Check if node has children
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        has_children = left_child != right_child  # This is a boolean, not numpy.bool
        
        # Convert numpy bool to Python bool if needed
        if hasattr(has_children, 'item'):
            has_children = bool(has_children.item())
        
        self.left_child_btn.setEnabled(has_children)
        self.right_child_btn.setEnabled(has_children)
        
        # Enable prev/next based on position
        self.prev_node_btn.setEnabled(node_id > 0)
        self.next_node_btn.setEnabled(node_id < tree.node_count - 1)
    
    def display_node_info(self, node_id, tree):
        """Display information about the selected node"""
        if not self.tree_results:
            return

        feature_names = self.tree_results['tree_structure']['feature_names']
        task_type = self.tree_results['parameters']['task_type']

        # Clear existing info
        self.node_info_text.clear()

        # Display node information
        self.node_info_text.append(f"<h3>Node {node_id} Information</h3>")

        # Check if it's a leaf node
        is_leaf = tree.children_left[node_id] == tree.children_right[node_id]

        if is_leaf:
            self.node_info_text.append("<b>Type:</b> Leaf Node")
        else:
            self.node_info_text.append("<b>Type:</b> Internal Node")

            feature_id = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_id] if feature_id < len(feature_names) else f"Feature {feature_id}"

            self.node_info_text.append(f"<b>Split Feature:</b> {feature_name}")
            self.node_info_text.append(f"<b>Split Threshold:</b> {threshold:.6f}")

            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            self.node_info_text.append(f"<b>Left Child Node:</b> {left_child}")
            self.node_info_text.append(f"<b>Right Child Node:</b> {right_child}")

            self.node_info_text.append(
                f"<b>Left Branch Samples:</b> {tree.n_node_samples[left_child]} "
                f"({tree.n_node_samples[left_child]/tree.n_node_samples[node_id]*100:.1f}%)"
            )
            self.node_info_text.append(
                f"<b>Right Branch Samples:</b> {tree.n_node_samples[right_child]} "
                f"({tree.n_node_samples[right_child]/tree.n_node_samples[node_id]*100:.1f}%)"
            )

        n_samples = tree.n_node_samples[node_id]
        total_samples = tree.n_node_samples[0]
        self.node_info_text.append(f"<b>Number of Samples:</b> {n_samples} ({n_samples/total_samples*100:.1f}% of total)")

        if task_type == 'regression':
            value = tree.value[node_id][0][0]
            self.node_info_text.append(f"<b>Predicted Value:</b> {value:.6f}")
        else:
            # Classification - show class distribution
            class_counts = tree.value[node_id][0]
            total = sum(class_counts)
            try:
                y_train = self.tree_results['raw_data']['y_train']
                class_names = sorted(list(set(y_train.tolist())))
                
                self.node_info_text.append("<b>Class Distribution:</b>")
                for i, (class_name, count) in enumerate(zip(class_names, class_counts)):
                    if i < len(class_names):  # Make sure we don't go out of bounds
                        percentage = (count / total * 100) if total > 0 else 0
                        self.node_info_text.append(f"&nbsp;&nbsp;&nbsp;{class_name}: {int(count)} ({percentage:.2f}%)")
            except Exception as e:
                self.node_info_text.append(f"<b>Class Distribution:</b> {class_counts}")
        
        # Node impurity (Gini or MSE)
        impurity = tree.impurity[node_id]
        impurity_name = "Gini" if task_type == "classification" else "MSE"
        self.node_info_text.append(f"<b>Node {impurity_name}:</b> {impurity:.6f}")
        
        # Calculate node depth
        depth = 0
        current = node_id
        while current != 0:  # Not root
            # Find parent
            for i in range(tree.node_count):
                if tree.children_left[i] == current or tree.children_right[i] == current:
                    current = i
                    depth += 1
                    break
                    
        self.node_info_text.append(f"<b>Node Depth:</b> {depth}")


    def visualize_node_distribution(self, node_id, tree):
        """Visualize the sample distribution in the selected node"""
        if not self.tree_results:
            return
        
        task_type = self.tree_results['parameters']['task_type']
        
        # Clear the canvas
        self.node_dist_canvas.fig.clear()
        ax = self.node_dist_canvas.fig.add_subplot(111)
        
        if task_type == 'classification':
            # Classification - bar chart of class distribution
            class_counts = tree.value[node_id][0]
            
            # Get class names if available
            try:
                target_name = self.tree_results['tree_structure']['target_name']
                class_names = sorted(list(set(self.tree_results['raw_data']['y_train'].tolist())))
            except:
                class_names = [f"Class {i}" for i in range(len(class_counts))]
            
            # Create bar chart with colorful bars
            bars = ax.bar(class_names, class_counts)
            
            # Use a color map for bars
            cmap = plt.cm.get_cmap('tab10', len(class_names))
            for i, bar in enumerate(bars):
                bar.set_color(cmap(i))
            
            ax.set_ylabel('Number of Samples')
            ax.set_title(f'Class Distribution in Node {node_id}')
            
            # Add counts and percentages on top of bars
            total = sum(class_counts)
            for i, count in enumerate(class_counts):
                if total > 0:
                    percentage = (count / total) * 100
                    ax.text(i, count + 0.1, f"{int(count)}\n({percentage:.1f}%)", 
                           ha='center', va='bottom', fontsize=9)
            
            # Rotate x labels if there are many classes
            if len(class_names) > 4:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            # Regression - show position in overall distribution
            node_value = tree.value[node_id][0][0]
            
            # Get target variable and its distribution
            y_train = self.tree_results['raw_data']['y_train']
            y_test = self.tree_results['raw_data']['y_test']
            all_y = pd.concat([y_train, y_test])
            
            # Create histogram of target distribution
            sns.histplot(all_y, kde=True, ax=ax, color='skyblue', alpha=0.6)
            
            # Add vertical line for node prediction
            ax.axvline(node_value, color='red', linestyle='--', 
                      label=f'Node Prediction: {node_value:.4f}')
            
            ax.set_xlabel('Target Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Target Distribution with Node {node_id} Prediction')
            ax.legend()
        
        # Update the canvas
        self.node_dist_canvas.fig.tight_layout()
        self.node_dist_canvas.draw()
    def visualize_node_position(self, node_id, tree):
        """Visualize where the node is located in the tree"""
        if not self.tree_results:
            return
            
        # Clear canvas
        self.node_pos_canvas.fig.clear()
        ax = self.node_pos_canvas.fig.add_subplot(111)
        
        # Get tree depth
        max_depth = self.tree_results['tree_structure']['max_depth']
        
        # Calculate node depth and path
        depth = 0
        path = []
        current = node_id
        while current != 0:  # Not root
            # Find parent
            for i in range(tree.node_count):
                if tree.children_left[i] == current:
                    path.append((i, current, 'left'))
                    current = i
                    depth += 1
                    break
                elif tree.children_right[i] == current:
                    path.append((i, current, 'right'))
                    current = i
                    depth += 1
                    break
            
        # Add root to path
        if current == 0 and node_id != 0:
            path.append((None, 0, None))
            
        # Reverse path to get root-to-node order
        path.reverse()
        
        # Determine depth range to visualize
        start_depth = max(0, depth - 2)
        end_depth = min(max_depth, depth + 2)
        
        # Create a simplified tree visualization
        try:
            # Get feature names
            feature_names = list(self.tree_results['tree_structure']['feature_names'])
            
            # Get model and render partial tree
            model = self.tree_results['tree_structure']['model']
            
            # Add title
            title = f"Node {node_id} Position in Tree"
            if start_depth > 0 or end_depth < max_depth:
                title += f" (Showing Depth {start_depth}-{end_depth})"
            ax.set_title(title)
            
            # Render tree with the selected node highlighted
            plot_tree(
                model,
                ax=ax,
                max_depth=end_depth,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                precision=2,
                fontsize=10
            )
            
            # Highlight selected node
            # This is a hack - we'll add a text annotation to point out the node
            # Finding exact coordinates is complex, so we'll use a text label
            ax.text(0.5, -0.05, f"Node {node_id} is highlighted in the tree above",
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(facecolor='yellow', alpha=0.5))
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error visualizing node position: {str(e)}",
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
        
        # Update canvas
        self.node_pos_canvas.fig.tight_layout()
        self.node_pos_canvas.draw()
    def display_decision_path(self, node_id, tree):
        """Display the decision path to the selected node"""
        if not self.tree_results:
            return
        
        feature_names = self.tree_results['tree_structure']['feature_names']
        
        # Clear existing text
        self.decision_path_text.clear()
        
        # If it's the root node, there's no path
        if node_id == 0:
            self.decision_path_text.setText("This is the root node (no decision path).")
            return
        
        # Trace path from root to node
        path = []
        current = node_id
        
        # Work backwards from node to root
        while current != 0:
            # Find parent node
            parent = -1
            left_child = False
            
            for i in range(tree.node_count):
                if tree.children_left[i] == current:
                    parent = i
                    left_child = True
                    break
                elif tree.children_right[i] == current:
                    parent = i
                    left_child = False
                    break
            
            if parent == -1:
                # Couldn't find parent, something went wrong
                break
            
            # Add decision to path
            feature_id = tree.feature[parent]
            threshold = tree.threshold[parent]
            
            feature_name = feature_names[feature_id] if feature_id < len(feature_names) else f"Feature {feature_id}"
            
            if left_child:
                decision = f"{feature_name} <= {threshold:.6f}"
            else:
                decision = f"{feature_name} > {threshold:.6f}"
            
            path.append((parent, decision))
            
            # Move to parent
            current = parent
        
        # Path is in reverse order, flip it
        path.reverse()
        
        # Format path as HTML
        self.decision_path_text.setHtml(f"<h3>Decision Path to Node {node_id}</h3>")
        
        for i, (parent, decision) in enumerate(path, 1):
            left_indicator = "└── " if i == len(path) else "├── "
            self.decision_path_text.append(f"{i}. <b>Node {parent}:</b> {left_indicator}{decision}")
    def update_error_analysis(self):
        """Update error analysis tab with current tree results"""
        if not self.tree_results:
            return
            
        # Check if task type is regression (only applicable for regression)
        if self.tree_results['parameters']['task_type'] != 'regression':
            return
        
        # Get the error analysis tab
        error_tab = None
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Error Analysis":
                error_tab = self.tabs.widget(i)
                break
        
        if not error_tab:
            # Create the error tab if it doesn't exist
            self.error_tab = ErrorAnalysisTab(self)
            self.tabs.addTab(self.error_tab, "Error Analysis")
        else:
            self.error_tab = error_tab
        
        # Make sure the error tab has a reference to the main application
        self.error_tab.main_app = self
            
        # Update all plots with current tree results
        self.error_tab.update_plots(self.tree_results)
    def update_top_errors(self):
        """Update top errors table"""
        if not self.tree_results or self.tree_results['parameters']['task_type'] != 'regression':
            return
            
        # Just refresh the table with current settings
        self.error_tab.update_plots(self.tree_results)
    
    #--------------------------------------------------------------------------
    # Dialog launching methods
    #--------------------------------------------------------------------------
    
    def show_data_summary(self):
        """Show data summary statistics"""
        if self.df is None:
            return
            
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Data Summary")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        dialog.setLayout(layout)
        
        # Tab widget for different summary views
        tabs = QTabWidget()
        
        # Basic stats tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        stats_tab.setLayout(stats_layout)
        
        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setFont(QFont("Courier New", 10))
        
        # Calculate summary statistics
        summary = self.df.describe(include='all').transpose()
        summary['missing'] = self.df.isnull().sum()
        summary['missing_pct'] = (self.df.isnull().sum() / len(self.df) * 100).round(2)
        summary['dtype'] = self.df.dtypes
        
        # Format as text
        stats_text.setText(f"Data Summary for {self.file_path}\n{'='*50}\n\n")
        stats_text.append(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}\n\n")
        stats_text.append("Summary Statistics:\n")
        stats_text.append(summary.to_string())
        
        stats_layout.addWidget(stats_text)
        tabs.addTab(stats_tab, "Statistics")
        
        # Correlation matrix tab
        corr_tab = QWidget()
        corr_layout = QVBoxLayout()
        corr_tab.setLayout(corr_layout)
        
        # Get numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            corr_canvas = MatplotlibCanvas(width=8, height=6)
            corr_toolbar = NavigationToolbar(corr_canvas, dialog)
            
            # Calculate correlation matrix
            corr = self.df[numeric_cols].corr()
            
            # Plot heatmap
            corr_canvas.fig.clear()
            ax = corr_canvas.fig.add_subplot(111)
            
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
            ax.set_title('Correlation Matrix')
            
            corr_canvas.fig.tight_layout()
            corr_canvas.draw()
            
            corr_layout.addWidget(corr_toolbar)
            corr_layout.addWidget(corr_canvas)
        else:
            corr_layout.addWidget(QLabel("No numeric columns available for correlation analysis."))
        
        tabs.addTab(corr_tab, "Correlation Matrix")
        
        # Distribution plots tab
        dist_tab = QWidget()
        dist_layout = QVBoxLayout()
        dist_tab.setLayout(dist_layout)
        
        # Create distribution plots for numeric columns
        if numeric_cols:
            dist_scroll = QScrollArea()
            dist_scroll.setWidgetResizable(True)
            
            dist_widget = QWidget()
            dist_grid = QGridLayout()
            dist_widget.setLayout(dist_grid)
            
            # Create histograms for each numeric column
            max_cols = 3
            row, col = 0, 0
            
            for i, column in enumerate(numeric_cols[:9]):  # Limit to 9 columns for performance
                dist_canvas = MatplotlibCanvas(width=4, height=3)
                
                # Plot histogram
                dist_canvas.fig.clear()
                ax = dist_canvas.fig.add_subplot(111)
                
                sns.histplot(self.df[column].dropna(), kde=True, ax=ax)
                ax.set_title(column)
                
                dist_canvas.fig.tight_layout()
                dist_canvas.draw()
                
                dist_grid.addWidget(dist_canvas, row, col)
                
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
            
            dist_scroll.setWidget(dist_widget)
            dist_layout.addWidget(dist_scroll)
        else:
            dist_layout.addWidget(QLabel("No numeric columns available for distribution analysis."))
        
        tabs.addTab(dist_tab, "Distributions")
        
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    def show_advanced_visualization(self):
        """Show advanced tree visualization dialog"""
        if not self.tree_results:
            return
            
        dialog = AdvancedTreeVisualizationDialog(self.tree_results, self)
        dialog.exec_()
    
    def show_important_nodes(self):
        """Show important nodes analysis dialog"""
        if not self.tree_results:
            return
            
        dialog = ImportantNodesDialog(self.tree_results, self)
        dialog.exec_()
    
    def show_tree_pruning(self):
        """Show tree pruning dialog"""
        if not self.tree_results:
            return
            
        dialog = TreePruningDialog(self.tree_results, self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_tree is not None:
            # Update the current tree with the pruned version
            self.tree_results['tree_structure']['model'] = dialog.selected_tree
            self.tree_results['tree_structure']['num_nodes'] = dialog.selected_tree.tree_.node_count
            self.tree_results['tree_structure']['max_depth'] = dialog.selected_tree.tree_.max_depth
            self.tree_results['tree_structure']['num_leaves'] = dialog.selected_tree.get_n_leaves()
            self.tree_results['tree_structure']['tree_text'] = export_text(
                dialog.selected_tree, 
                feature_names=list(self.tree_results['tree_structure']['feature_names'])
            )
            
            # Update UI elements
            self.update_tree_info()
            self.update_tree_visualization()
            self.update_node_selector()
            
            QMessageBox.information(self, "Tree Updated", 
                                  f"Tree has been pruned with alpha={dialog.selected_alpha:.8f}.\n"
                                  f"New node count: {dialog.selected_tree.tree_.node_count}")
    

     #--------------------------------------------------------------------------
    # Export methods
    #--------------------------------------------------------------------------
    
    def save_results(self):
        """Save analysis results to file"""
        if not self.tree_results:
            QMessageBox.warning(self, "Error", "No analysis results to save")
            return
        
        # Create dialog to choose what to save
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Options")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Checkboxes for what to save
        save_metrics = QCheckBox("Performance Metrics")
        save_metrics.setChecked(True)
        layout.addWidget(save_metrics)
        
        save_tree_viz = QCheckBox("Tree Visualization")
        save_tree_viz.setChecked(True)
        layout.addWidget(save_tree_viz)
        
        save_feature_importance = QCheckBox("Feature Importance")
        save_feature_importance.setChecked(True)
        layout.addWidget(save_feature_importance)
        
        save_tree_structure = QCheckBox("Tree Structure (Text)")
        save_tree_structure.setChecked(True)
        layout.addWidget(save_tree_structure)
        
        save_error_analysis = QCheckBox("Error Analysis (Regression Only)")
        save_error_analysis.setChecked(True)
        layout.addWidget(save_error_analysis)
        
        save_important_nodes = QCheckBox("Important Nodes Analysis")
        save_important_nodes.setChecked(True)
        layout.addWidget(save_important_nodes)
        
        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Save Format:"))
        
        save_format = QComboBox()
        save_format.addItems(["Excel (.xlsx)", "CSV (.csv)", "Text (.txt)", "PDF (.pdf)"])
        format_layout.addWidget(save_format)
        
        layout.addLayout(format_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() != QDialog.Accepted:
            return
        
        # Get selected options
        do_save_metrics = save_metrics.isChecked()
        do_save_tree_viz = save_tree_viz.isChecked()
        do_save_feature_importance = save_feature_importance.isChecked()
        do_save_tree_structure = save_tree_structure.isChecked()
        do_save_error_analysis = save_error_analysis.isChecked()
        do_save_important_nodes = save_important_nodes.isChecked()
        
        format_idx = save_format.currentIndex()
        
        # Get save path
        if format_idx == 0:  # Excel
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "", "Excel Files (*.xlsx)")
            if not file_path:
                return
            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'
            
            # Save to Excel
            self.save_results_to_excel(file_path, do_save_metrics, 
                                    do_save_tree_structure, do_save_error_analysis,
                                    do_save_important_nodes)
            
        elif format_idx == 1:  # CSV
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "", "CSV Files (*.csv)")
            if not file_path:
                return
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            
            # Save to CSV
            self.save_results_to_csv(file_path, do_save_metrics, do_save_error_analysis,
                                do_save_important_nodes)
            
        elif format_idx == 2:  # Text
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "", "Text Files (*.txt)")
            if not file_path:
                return
            if not file_path.endswith('.txt'):
                file_path += '.txt'
            
            # Save to text
            self.save_results_to_text(file_path, do_save_metrics,
                                    do_save_tree_structure, do_save_error_analysis,
                                    do_save_important_nodes)
            
        elif format_idx == 3:  # PDF
            if not HAS_REPORTLAB:
                QMessageBox.warning(self, "PDF Export Error", 
                                "PDF export requires the ReportLab library. Please install it with:\n\npip install reportlab")
                return
                
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "", "PDF Files (*.pdf)")
            if not file_path:
                return
            if not file_path.endswith('.pdf'):
                file_path += '.pdf'
            
            # Save to PDF
            try:
                self.save_results_to_pdf(file_path, do_save_metrics, do_save_tree_viz,
                                    do_save_feature_importance, do_save_tree_structure,
                                    do_save_error_analysis, do_save_important_nodes)
            except Exception as e:
                QMessageBox.warning(self, "PDF Export Error",
                                f"Error generating PDF: {str(e)}\nSwitching to text format.")
                self.save_results_to_text(file_path.replace('.pdf', '.txt'),
                                        do_save_metrics, do_save_tree_structure, 
                                        do_save_error_analysis, do_save_important_nodes)
        
        # Save visualizations separately if needed
        if format_idx != 3:  # PDF includes visualizations
            if do_save_tree_viz:
                viz_path = file_path.rsplit('.', 1)[0] + "_tree.png"
                self.tree_canvas.fig.savefig(viz_path, dpi=300, bbox_inches='tight')
            
            if do_save_feature_importance:
                imp_path = file_path.rsplit('.', 1)[0] + "_importance.png"
                self.feature_importance_canvas.fig.savefig(imp_path, dpi=300, bbox_inches='tight')
                
            if do_save_error_analysis and self.tree_results['parameters']['task_type'] == 'regression':
                error_path = file_path.rsplit('.', 1)[0] + "_error_analysis.png"
                
                # Create a composite figure with error plots
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                
                # Extract data for error analysis
                y_test = self.tree_results['raw_data']['y_test'].reset_index(drop=True)
                y_test_pred = pd.Series(self.tree_results['predictions']['y_test_pred'])
                test_error = y_test - y_test_pred
                
                # 1. Actual vs Predicted
                axs[0, 0].scatter(y_test, y_test_pred, alpha=0.6, color='blue')
                axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                axs[0, 0].set_title("Actual vs Predicted")
                axs[0, 0].set_xlabel("Actual")
                axs[0, 0].set_ylabel("Predicted")
                
                # 2. Error Distribution
                sns.histplot(test_error, kde=True, ax=axs[0, 1], color='green')
                axs[0, 1].axvline(0, linestyle='--', color='red')
                axs[0, 1].set_title("Error Distribution")
                
                # 3. Error vs Index
                axs[1, 0].plot(range(len(test_error)), test_error, color='purple')
                axs[1, 0].axhline(y=0, color='red', linestyle='--')
                axs[1, 0].set_title("Error vs Index")
                
                # 4. Error vs Predicted
                axs[1, 1].scatter(y_test_pred, test_error, alpha=0.6, color='orange')
                axs[1, 1].axhline(y=0, color='red', linestyle='--')
                axs[1, 1].set_title("Residuals vs Predicted")
                
                fig.tight_layout()
                fig.savefig(error_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        QMessageBox.information(self, "Save Complete", f"Results saved to {file_path}")
    

    def save_results_to_excel(self, file_path, save_metrics=True, save_tree_structure=True,
                        save_error_analysis=True, save_important_nodes=True):
        """Save results to Excel file"""
        import pandas as pd
        
        with pd.ExcelWriter(file_path) as writer:
            # Save metrics
            if save_metrics:
                task_type = self.tree_results['parameters']['task_type']
                
                # Convert metrics dict to DataFrame
                train_metrics = pd.DataFrame([])
                test_metrics = pd.DataFrame([])
                
                if task_type == 'regression':
                    # Extract metrics
                    train_data = []
                    for metric, value in self.tree_results['metrics']['train'].items():
                        if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                            train_data.append({'Metric': metric, 'Value': value})
                    
                    test_data = []
                    for metric, value in self.tree_results['metrics']['test'].items():
                        if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                            test_data.append({'Metric': metric, 'Value': value})
                    
                    train_metrics = pd.DataFrame(train_data)
                    test_metrics = pd.DataFrame(test_data)
                else:
                    # Classification metrics
                    train_data = []
                    for metric, value in self.tree_results['metrics']['train'].items():
                        if metric not in ['Confusion Matrix', 'Classification Report'] and isinstance(value, (int, float, np.number)):
                            train_data.append({'Metric': metric, 'Value': value})
                    
                    test_data = []
                    for metric, value in self.tree_results['metrics']['test'].items():
                        if metric not in ['Confusion Matrix', 'Classification Report'] and isinstance(value, (int, float, np.number)):
                            test_data.append({'Metric': metric, 'Value': value})
                    
                    train_metrics = pd.DataFrame(train_data)
                    test_metrics = pd.DataFrame(test_data)
                
                # Write metrics to Excel
                train_metrics.to_excel(writer, sheet_name='Train Metrics', index=False)
                test_metrics.to_excel(writer, sheet_name='Test Metrics', index=False)
                
                # Save feature importance
                features = self.tree_results['feature_importance']['features']
                importance = self.tree_results['feature_importance']['importance']
                
                imp_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                })
                imp_df = imp_df.sort_values('Importance', ascending=False)
                
                imp_df.to_excel(writer, sheet_name='Feature Importance', index=False)
                
                # Save cross-validation results if available
                if 'cv_results' in self.tree_results:
                    cv_scores = self.tree_results['cv_results']['scores']
                    cv_df = pd.DataFrame({
                        'Fold': range(1, len(cv_scores) + 1),
                        'Score': cv_scores
                    })
                    
                    # Add mean and std
                    mean_row = pd.DataFrame([{'Fold': 'Mean', 'Score': self.tree_results['cv_results']['mean']}])
                    std_row = pd.DataFrame([{'Fold': 'Std', 'Score': self.tree_results['cv_results']['std']}])
                    cv_df = pd.concat([cv_df, mean_row, std_row], ignore_index=True)
                    
                    cv_df.to_excel(writer, sheet_name='Cross-Validation', index=False)
                
                # For classification, save confusion matrix
                if task_type == 'classification':
                    conf_matrix_train = self.tree_results['metrics']['train'].get('Confusion Matrix')
                    conf_matrix_test = self.tree_results['metrics']['test'].get('Confusion Matrix')
                    
                    # Try to get class names
                    try:
                        target_name = self.tree_results['tree_structure']['target_name']
                        class_names = sorted(list(set(self.tree_results['raw_data']['y_train'].tolist())))
                    except:
                        class_names = [f"Class {i}" for i in range(len(conf_matrix_train))]
                    
                    # Create DataFrames
                    conf_df_train = pd.DataFrame(conf_matrix_train, 
                                            index=[f"True_{c}" for c in class_names],
                                            columns=[f"Pred_{c}" for c in class_names])
                    
                    conf_df_test = pd.DataFrame(conf_matrix_test, 
                                            index=[f"True_{c}" for c in class_names],
                                            columns=[f"Pred_{c}" for c in class_names])
                    
                    conf_df_train.to_excel(writer, sheet_name='Confusion Matrix (Train)')
                    conf_df_test.to_excel(writer, sheet_name='Confusion Matrix (Test)')
            
            # Save tree structure as text
            if save_tree_structure:
                tree_text = self.tree_results['tree_structure']['tree_text']
                tree_df = pd.DataFrame({'Tree Structure': tree_text.split('\n')})
                tree_df.to_excel(writer, sheet_name='Tree Structure', index=False)
            
            # Save error analysis for regression
            if save_error_analysis and self.tree_results['parameters']['task_type'] == 'regression':
                # Extract error data
                y_train = self.tree_results['raw_data']['y_train'].reset_index(drop=True)
                y_train_pred = pd.Series(self.tree_results['predictions']['y_train_pred'])
                y_test = self.tree_results['raw_data']['y_test'].reset_index(drop=True)
                y_test_pred = pd.Series(self.tree_results['predictions']['y_test_pred'])
                
                train_error = y_train - y_train_pred
                test_error = y_test - y_test_pred
                
                # Create error DataFrames
                train_errors = pd.DataFrame({
                    'Index': y_train.index,
                    'Actual': y_train,
                    'Predicted': y_train_pred,
                    'Error': train_error,
                    'AbsError': np.abs(train_error)
                })
                
                test_errors = pd.DataFrame({
                    'Index': y_test.index,
                    'Actual': y_test,
                    'Predicted': y_test_pred,
                    'Error': test_error,
                    'AbsError': np.abs(test_error)
                })
                
                # Save to Excel
                train_errors.to_excel(writer, sheet_name='Train Errors', index=False)
                test_errors.to_excel(writer, sheet_name='Test Errors', index=False)
                
                # Save error statistics
                error_stats = pd.DataFrame([
                    {'Metric': 'Mean Error (Train)', 'Value': train_error.mean()},
                    {'Metric': 'Mean Error (Test)', 'Value': test_error.mean()},
                    {'Metric': 'Mean Abs Error (Train)', 'Value': np.abs(train_error).mean()},
                    {'Metric': 'Mean Abs Error (Test)', 'Value': np.abs(test_error).mean()},
                    {'Metric': 'RMSE (Train)', 'Value': np.sqrt(np.mean(train_error**2))},
                    {'Metric': 'RMSE (Test)', 'Value': np.sqrt(np.mean(test_error**2))},
                    {'Metric': 'Max Abs Error (Train)', 'Value': np.abs(train_error).max()},
                    {'Metric': 'Max Abs Error (Test)', 'Value': np.abs(test_error).max()}
                ])
                
                error_stats.to_excel(writer, sheet_name='Error Statistics', index=False)
            
            # Save important nodes data
            if save_important_nodes and 'important_nodes' in self.tree_results and self.tree_results['important_nodes']:
                # Node summary data
                nodes_summary = []
                
                for node in self.tree_results['important_nodes']:
                    # Get feature name
                    feature = "Leaf" if node['is_leaf'] else (
                        node['feature'] if isinstance(node['feature'], str) else 
                        self.tree_results['tree_structure']['feature_names'][node['feature']] 
                        if node['feature'] is not None and node['feature'] < len(self.tree_results['tree_structure']['feature_names']) 
                        else f"Feature {node['feature']}")
                    
                    # Add node data
                    nodes_summary.append({
                        'Node ID': node['id'],
                        'Importance': node['importance'],
                        'Depth': node['depth'],
                        'Samples': node['samples'],
                        'Feature': feature,
                        'Is Leaf': node['is_leaf'],
                        'Threshold': node['threshold'] if not node['is_leaf'] else "N/A"
                    })
                
                # Create DataFrame
                nodes_df = pd.DataFrame(nodes_summary)
                nodes_df.to_excel(writer, sheet_name='Important Nodes', index=False)
                
                # Create separate sheet for decision paths
                paths_data = []
                
                for node in self.tree_results['important_nodes']:
                    if node['decision_path']:
                        for i, (parent, decision) in enumerate(node['decision_path']):
                            paths_data.append({
                                'Node ID': node['id'],
                                'Step': i + 1,
                                'Parent Node': parent,
                                'Decision': decision
                            })
                
                if paths_data:
                    paths_df = pd.DataFrame(paths_data)
                    paths_df.to_excel(writer, sheet_name='Decision Paths', index=False)
        
    def save_results_to_csv(self, file_path, save_metrics=True, save_error_analysis=True, 
                         save_important_nodes=True):
        """Save results to CSV files (multiple files)"""
        import pandas as pd
        
        base_path = file_path.rsplit('.', 1)[0]
        
        if save_metrics:
            # Create metrics DataFrame
            task_type = self.tree_results['parameters']['task_type']
            
            all_metrics = []
            
            # Add train metrics
            for metric, value in self.tree_results['metrics']['train'].items():
                if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                    all_metrics.append({
                        'Set': 'Train',
                        'Metric': metric,
                        'Value': value
                    })
            
            # Add test metrics
            for metric, value in self.tree_results['metrics']['test'].items():
                if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                    all_metrics.append({
                        'Set': 'Test',
                        'Metric': metric,
                        'Value': value
                    })
            
            # Create DataFrame and save
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(file_path, index=False)
            
            # Save feature importance to separate file
            imp_path = base_path + "_importance.csv"
            
            features = self.tree_results['feature_importance']['features']
            importance = self.tree_results['feature_importance']['importance']
            
            imp_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            })
            imp_df = imp_df.sort_values('Importance', ascending=False)
            
            imp_df.to_csv(imp_path, index=False)
            
            # Save parameters to separate file
            params_path = base_path + "_parameters.csv"
            
            params = []
            params.append({'Parameter': 'Task Type', 'Value': self.tree_results['parameters']['task_type']})
            params.append({'Parameter': 'Test Size', 'Value': self.tree_results['parameters']['test_size']})
            params.append({'Parameter': 'Random State', 'Value': self.tree_results['parameters']['random_state']})
            params.append({'Parameter': 'Scaling Method', 'Value': self.tree_results['parameters']['scaling_method']})
            
            for param, value in self.tree_results['parameters']['tree_params'].items():
                params.append({'Parameter': param, 'Value': str(value)})
            
            params_df = pd.DataFrame(params)
            params_df.to_csv(params_path, index=False)
        
        # Save error analysis data for regression
        if save_error_analysis and self.tree_results['parameters']['task_type'] == 'regression':
            error_path = base_path + "_errors.csv"
            
            # Extract error data
            y_train = self.tree_results['raw_data']['y_train'].reset_index(drop=True)
            y_train_pred = pd.Series(self.tree_results['predictions']['y_train_pred'])
            y_test = self.tree_results['raw_data']['y_test'].reset_index(drop=True)
            y_test_pred = pd.Series(self.tree_results['predictions']['y_test_pred'])
            
            train_error = y_train - y_train_pred
            test_error = y_test - y_test_pred
            
            # Create error DataFrames
            train_errors = pd.DataFrame({
                'Set': ['Train'] * len(y_train),
                'Index': y_train.index,
                'Actual': y_train,
                'Predicted': y_train_pred,
                'Error': train_error,
                'AbsError': np.abs(train_error)
            })
            
            test_errors = pd.DataFrame({
                'Set': ['Test'] * len(y_test),
                'Index': y_test.index,
                'Actual': y_test,
                'Predicted': y_test_pred,
                'Error': test_error,
                'AbsError': np.abs(test_error)
            })
            
            # Combine and save
            all_errors = pd.concat([train_errors, test_errors])
            all_errors.to_csv(error_path, index=False)
        
        # Save important nodes data
        if save_important_nodes and 'important_nodes' in self.tree_results and self.tree_results['important_nodes']:
            nodes_path = base_path + "_important_nodes.csv"
            
            # Node summary data
            nodes_summary = []
            
            for node in self.tree_results['important_nodes']:
                # Get feature name
                feature = "Leaf" if node['is_leaf'] else (
                    node['feature'] if isinstance(node['feature'], str) else 
                    self.tree_results['tree_structure']['feature_names'][node['feature']] 
                    if node['feature'] is not None and node['feature'] < len(self.tree_results['tree_structure']['feature_names']) 
                    else f"Feature {node['feature']}")
                
                # Create decision path text
                decision_path = ""
                if node['decision_path']:
                    for i, (parent, decision) in enumerate(node['decision_path'], 1):
                        decision_path += f"{i}. Node {parent}: {decision}; "
                
                # Add node data
                nodes_summary.append({
                    'Node ID': node['id'],
                    'Importance': node['importance'],
                    'Depth': node['depth'],
                    'Samples': node['samples'],
                    'Feature': feature,
                    'Is Leaf': node['is_leaf'],
                    'Threshold': node['threshold'] if not node['is_leaf'] else "N/A",
                    'Decision Path': decision_path
                })
            
            # Create DataFrame
            nodes_df = pd.DataFrame(nodes_summary)
            nodes_df.to_csv(nodes_path, index=False)
            
            # Save detailed path information in a separate file
            paths_path = base_path + "_decision_paths.csv"
            paths_data = []
            
            for node in self.tree_results['important_nodes']:
                if node['decision_path']:
                    for i, (parent, decision) in enumerate(node['decision_path']):
                        paths_data.append({
                            'Node ID': node['id'],
                            'Step': i + 1,
                            'Parent Node': parent,
                            'Decision': decision
                        })
            
            if paths_data:
                paths_df = pd.DataFrame(paths_data)
                paths_df.to_csv(paths_path, index=False)


    def save_results_to_text(self, file_path, save_metrics=True, save_tree_structure=True, 
                            save_error_analysis=True, save_important_nodes=True):
        """Save results to text file"""
        with open(file_path, 'w') as f:
            # Write header
            f.write(f"Decision Tree Analysis Results\n")
            f.write(f"{'='*50}\n\n")
            
            # Write parameters
            f.write(f"Parameters\n{'-'*30}\n")
            f.write(f"Task Type: {self.tree_results['parameters']['task_type']}\n")
            f.write(f"Test Size: {self.tree_results['parameters']['test_size']}\n")
            f.write(f"Random State: {self.tree_results['parameters']['random_state']}\n")
            f.write(f"Scaling Method: {self.tree_results['parameters']['scaling_method']}\n\n")
            
            f.write("Tree Parameters:\n")
            for param, value in self.tree_results['parameters']['tree_params'].items():
                f.write(f"  {param}: {value}\n")
            
            f.write("\n")
            
            # Write metrics
            if save_metrics:
                f.write(f"Performance Metrics\n{'-'*30}\n")
                
                # Train metrics
                f.write("Training Set Metrics:\n")
                for metric, value in self.tree_results['metrics']['train'].items():
                    if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                        f.write(f"  {metric}: {value:.6f}\n")
                
                f.write("\n")
                
                # Test metrics
                f.write("Test Set Metrics:\n")
                for metric, value in self.tree_results['metrics']['test'].items():
                    if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                        f.write(f"  {metric}: {value:.6f}\n")
                
                f.write("\n")
                
                # Feature importance
                f.write(f"Feature Importance\n{'-'*30}\n")
                features = self.tree_results['feature_importance']['features']
                importance = self.tree_results['feature_importance']['importance']
                
                # Sort by importance
                feature_importance = [(f, i) for f, i in zip(features, importance)]
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                for feature, imp in feature_importance:
                    f.write(f"  {feature}: {imp:.6f}\n")
                
                f.write("\n")
                
                # Cross-validation results if available
                if 'cv_results' in self.tree_results:
                    f.write(f"Cross-Validation Results\n{'-'*30}\n")
                    f.write(f"Mean Score: {self.tree_results['cv_results']['mean']:.6f}\n")
                    f.write(f"Standard Deviation: {self.tree_results['cv_results']['std']:.6f}\n\n")
                    
                    f.write("Scores by Fold:\n")
                    for i, score in enumerate(self.tree_results['cv_results']['scores'], 1):
                        f.write(f"  Fold {i}: {score:.6f}\n")
                    
                    f.write("\n")
            
            # Write tree structure
            if save_tree_structure:
                f.write(f"Tree Structure\n{'-'*30}\n")
                f.write(self.tree_results['tree_structure']['tree_text'])
                f.write("\n\n")
            
            # Write error analysis for regression
            if save_error_analysis and self.tree_results['parameters']['task_type'] == 'regression':
                f.write(f"Error Analysis\n{'-'*30}\n")
                
                # Extract error data
                y_train = self.tree_results['raw_data']['y_train']
                y_train_pred = pd.Series(self.tree_results['predictions']['y_train_pred'])
                y_test = self.tree_results['raw_data']['y_test']
                y_test_pred = pd.Series(self.tree_results['predictions']['y_test_pred'])
                
                train_error = y_train - y_train_pred
                test_error = y_test - y_test_pred
                
                # Write error statistics
                f.write("Error Statistics:\n")
                f.write("  Mean Error (Train): {:.6f}\n".format(train_error.mean()))
                f.write("  Mean Error (Test): {:.6f}\n".format(test_error.mean()))
                f.write("  Mean Absolute Error (Train): {:.6f}\n".format(np.abs(train_error).mean()))
                f.write("  Mean Absolute Error (Test): {:.6f}\n".format(np.abs(test_error).mean()))
                f.write("  RMSE (Train): {:.6f}\n".format(np.sqrt(np.mean(train_error**2))))
                f.write("  RMSE (Test): {:.6f}\n".format(np.sqrt(np.mean(test_error**2))))
                f.write("  Max Absolute Error (Train): {:.6f}\n".format(np.abs(train_error).max()))
                f.write("  Max Absolute Error (Test): {:.6f}\n".format(np.abs(test_error).max()))
                
                f.write("\n")
                
                # Write top 10 highest errors for test set
                test_errors = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_test_pred,
                    'Error': test_error,
                    'AbsError': np.abs(test_error)
                })
                top_errors = test_errors.sort_values('AbsError', ascending=False).head(10)
                
                f.write("Top 10 Highest Errors (Test Set):\n")
                for i, (_, row) in enumerate(top_errors.iterrows(), 1):
                    f.write(f"  {i}. Actual: {row['Actual']:.6f}, Predicted: {row['Predicted']:.6f}, "
                        f"Error: {row['Error']:.6f}, Abs Error: {row['AbsError']:.6f}\n")
                    
                f.write("\n")
            
            # Write important nodes information
            if save_important_nodes and 'important_nodes' in self.tree_results and self.tree_results['important_nodes']:
                f.write(f"Important Nodes\n{'-'*30}\n")
                
                for i, node in enumerate(self.tree_results['important_nodes'], 1):
                    # Get feature name
                    feature = "Leaf" if node['is_leaf'] else (
                        node['feature'] if isinstance(node['feature'], str) else 
                        self.tree_results['tree_structure']['feature_names'][node['feature']] 
                        if node['feature'] is not None and node['feature'] < len(self.tree_results['tree_structure']['feature_names']) 
                        else f"Feature {node['feature']}")
                    
                    f.write(f"Node {i} (ID: {node['id']})\n")
                    f.write(f"  Importance: {node['importance']:.6f}\n")
                    f.write(f"  Depth: {node['depth']}\n")
                    f.write(f"  Samples: {node['samples']} ({node['samples']/self.tree_results['tree_structure']['model'].tree_.n_node_samples[0]*100:.1f}% of total)\n")
                    
                    if node['is_leaf']:
                        f.write("  Type: Leaf Node\n")
                        
                        # Add value info
                        task_type = self.tree_results['parameters']['task_type']
                        tree = self.tree_results['tree_structure']['model'].tree_
                        
                        if task_type == 'regression':
                            value = tree.value[node['id']][0][0]
                            f.write(f"  Predicted Value: {value:.6f}\n")
                        else:
                            # Classification - show class distribution
                            class_counts = tree.value[node['id']][0]
                            f.write(f"  Class Distribution: {class_counts}\n")
                    else:
                        f.write("  Type: Decision Node\n")
                        f.write(f"  Split Feature: {feature}\n")
                        f.write(f"  Split Threshold: {node['threshold']:.6f}\n")
                    
                    # Write decision path
                    if node['decision_path']:
                        f.write("  Decision Path:\n")
                        for j, (parent, decision) in enumerate(node['decision_path'], 1):
                            f.write(f"    {j}. Node {parent}: {decision}\n")
                    else:
                        f.write("  Decision Path: Root node (no path)\n")
                    
                    f.write("\n")

    def save_results_to_pdf(self, file_path, save_metrics=True, save_tree_viz=True, 
                      save_feature_importance=True, save_tree_structure=True,
                      save_error_analysis=True):
        """Save results to PDF file"""
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, ListFlowable, ListItem
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.utils import ImageReader
        from io import BytesIO
        
        # Create PDF document
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        styles.add(ParagraphStyle(name='Title', fontSize=16, spaceAfter=12, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='Heading1', fontSize=14, spaceAfter=10, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='Heading2', fontSize=12, spaceAfter=8, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='Normal', fontSize=10, spaceAfter=6))
        styles.add(ParagraphStyle(name='Code', fontSize=8, fontName='Courier', spaceAfter=6))
        
        # Story elements
        story = []
        
        # Title
        story.append(Paragraph("Decision Tree Analysis Report", styles['Title']))
        story.append(Spacer(1, 0.2*inch))
        
        # Parameters section
        story.append(Paragraph("Parameters", styles['Heading1']))
        
        params_data = [["Parameter", "Value"]]
        params_data.append(["Task Type", self.tree_results['parameters']['task_type']])
        params_data.append(["Test Size", str(self.tree_results['parameters']['test_size'])])
        params_data.append(["Random State", str(self.tree_results['parameters']['random_state'])])
        params_data.append(["Scaling Method", str(self.tree_results['parameters']['scaling_method'])])
        
        for param, value in self.tree_results['parameters']['tree_params'].items():
            params_data.append([param, str(value)])
        
        params_table = Table(params_data, colWidths=[2*inch, 3*inch])
        params_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(params_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Performance metrics section
        if save_metrics:
            story.append(Paragraph("Performance Metrics", styles['Heading1']))
            
            # Create metrics table
            metrics_data = [["Metric", "Training Set", "Test Set"]]
            
            task_type = self.tree_results['parameters']['task_type']
            
            # Get common metrics
            common_metrics = []
            for metric in self.tree_results['metrics']['train'].keys():
                if (metric in self.tree_results['metrics']['test'] and 
                    isinstance(self.tree_results['metrics']['train'][metric], (int, float, np.number)) and 
                    not np.isnan(self.tree_results['metrics']['train'][metric]) and
                    metric not in ['Confusion Matrix', 'Classification Report']):
                    common_metrics.append(metric)
            
            # Sort metrics
            if task_type == 'regression':
                important_metrics = ['R²', 'MSE', 'RMSE', 'MAE', 'Explained Variance']
                sorted_metrics = [m for m in important_metrics if m in common_metrics]
                sorted_metrics += [m for m in common_metrics if m not in important_metrics]
            else:
                important_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
                sorted_metrics = [m for m in important_metrics if m in common_metrics]
                sorted_metrics += [m for m in common_metrics if m not in important_metrics]
            
            # Add metrics to table
            for metric in sorted_metrics:
                metrics_data.append([
                    metric,
                    f"{self.tree_results['metrics']['train'][metric]:.6f}",
                    f"{self.tree_results['metrics']['test'][metric]:.6f}"
                ])
            
            metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metrics_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Add cross-validation results if available
            if 'cv_results' in self.tree_results:
                story.append(Paragraph("Cross-Validation Results", styles['Heading2']))
                
                cv_text = f"Mean Score: {self.tree_results['cv_results']['mean']:.6f}\n"
                cv_text += f"Standard Deviation: {self.tree_results['cv_results']['std']:.6f}\n\n"
                
                story.append(Paragraph(cv_text, styles['Normal']))
                
                # Create CV scores table
                cv_data = [["Fold", "Score"]]
                
                for i, score in enumerate(self.tree_results['cv_results']['scores'], 1):
                    cv_data.append([f"Fold {i}", f"{score:.6f}"])
                
                cv_table = Table(cv_data, colWidths=[2*inch, 3*inch])
                cv_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(cv_table)
                story.append(Spacer(1, 0.2*inch))
        
        # Tree visualization
        if save_tree_viz:
            story.append(Paragraph("Decision Tree Visualization", styles['Heading1']))
            
            # Save tree visualization to a BytesIO object
            img_io = BytesIO()
            max_depth = min(3, self.tree_results['tree_structure']['max_depth'])  # Limit depth for legibility
            
            # Create figure
            fig, ax = plt.subplots(figsize=(7, 5))
            
            # Get feature names
            feature_names = list(self.tree_results['tree_structure']['feature_names'])
            
            # Get class names for classification
            class_names = None
            if self.tree_results['parameters']['task_type'] == 'classification':
                try:
                    class_names = sorted(list(set(self.tree_results['raw_data']['y_train'].tolist())))
                except:
                    pass
            
            # Plot tree
            plot_tree(
                self.tree_results['tree_structure']['model'],
                ax=ax,
                max_depth=max_depth,
                filled=True,
                feature_names=feature_names,
                class_names=class_names,
                rounded=True,
                precision=2,
                fontsize=8
            )
            
            ax.set_title(f"Decision Tree (Max Depth: {max_depth})")
            plt.tight_layout()
            
            # Save to BytesIO
            plt.savefig(img_io, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Reset the pointer to the start of BytesIO object
            img_io.seek(0)
            
            # Add image to PDF
            img = Image(ImageReader(img_io))
            img.drawHeight = 4*inch
            img.drawWidth = 6*inch
            
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
            
            # Add note about tree depth
            if self.tree_results['tree_structure']['max_depth'] > max_depth:
                note = f"Note: This visualization shows only the first {max_depth} levels of the tree. "
                note += f"The full tree has a maximum depth of {self.tree_results['tree_structure']['max_depth']}."
                story.append(Paragraph(note, styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
        
        # Feature importance
        if save_feature_importance:
            story.append(Paragraph("Feature Importance", styles['Heading1']))
            
            # Save feature importance to a BytesIO object
            img_io = BytesIO()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(7, 5))
            
            # Get feature importance data
            features = self.tree_results['feature_importance']['features']
            importance = self.tree_results['feature_importance']['importance']
            
            # Sort and get top 15 features
            feature_importance = [(f, i) for f, i in zip(features, importance)]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_features = feature_importance[:15]
            
            if not top_features:
                story.append(Paragraph("No feature importance data available.", styles['Normal']))
            else:
                # Unpack data
                feature_names, importances = zip(*top_features)
                
                # Plot horizontal bar chart
                y_pos = np.arange(len(feature_names))
                ax.barh(y_pos, importances, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_names)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance')
                
                plt.tight_layout()
                
                # Save to BytesIO
                plt.savefig(img_io, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Reset the pointer to the start of BytesIO object
                img_io.seek(0)
                
                # Add image to PDF
                img = Image(ImageReader(img_io))
                img.drawHeight = 4*inch
                img.drawWidth = 6*inch
                
                story.append(img)
            
            story.append(Spacer(1, 0.2*inch))
        
        # Tree structure as text
        if save_tree_structure:
            story.append(Paragraph("Tree Structure", styles['Heading1']))
            
            # Get tree text and format it
            tree_text = self.tree_results['tree_structure']['tree_text']
            
            # If tree text is too long, truncate it
            if len(tree_text) > 5000:
                tree_text = tree_text[:5000] + "...\n\n(Tree text truncated due to length)"
            
            # Add tree text
            story.append(Paragraph(tree_text.replace('\n', '<br/>'), styles['Code']))
            story.append(Spacer(1, 0.2*inch))
        
        # Error analysis for regression
        if save_error_analysis and self.tree_results['parameters']['task_type'] == 'regression':
            story.append(PageBreak())
            story.append(Paragraph("Error Analysis", styles['Heading1']))
            
            # Extract error data
            y_train = self.tree_results['raw_data']['y_train'].reset_index(drop=True)
            y_train_pred = pd.Series(self.tree_results['predictions']['y_train_pred'])
            y_test = self.tree_results['raw_data']['y_test'].reset_index(drop=True)
            y_test_pred = pd.Series(self.tree_results['predictions']['y_test_pred'])
            
            train_error = y_train - y_train_pred
            test_error = y_test - y_test_pred
            
            # Add error summary
            error_summary = f"""
            <b>Error Statistics Summary:</b><br/>
            Mean Error (Train): {train_error.mean():.6f}<br/>
            Mean Error (Test): {test_error.mean():.6f}<br/>
            Mean Absolute Error (Train): {np.abs(train_error).mean():.6f}<br/>
            Mean Absolute Error (Test): {np.abs(test_error).mean():.6f}<br/>
            RMSE (Train): {np.sqrt(np.mean(train_error**2)):.6f}<br/>
            RMSE (Test): {np.sqrt(np.mean(test_error**2)):.6f}<br/>
            Max Absolute Error (Train): {np.abs(train_error).max():.6f}<br/>
            Max Absolute Error (Test): {np.abs(test_error).max():.6f}<br/>
            """
            
            story.append(Paragraph(error_summary, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Create error visualizations
            # 1. Error Distribution
            img_io = BytesIO()
            fig, ax = plt.subplots(figsize=(7, 4))
            
            sns.histplot(test_error, kde=True, ax=ax, color='blue', alpha=0.6, label='Test Error')
            ax.axvline(0, linestyle='--', color='red')
            ax.set_title('Error Distribution (Test Set)')
            ax.set_xlabel('Error')
            ax.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(img_io, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            img_io.seek(0)
            img = Image(ImageReader(img_io))
            img.drawHeight = 3*inch
            img.drawWidth = 6*inch
            
            story.append(Paragraph("Error Distribution", styles['Heading2']))
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
            
            # 2. Actual vs Predicted
            img_io = BytesIO()
            fig, ax = plt.subplots(figsize=(7, 4))
            
            ax.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
            
            # Add perfect prediction line
            min_val = min(y_test.min(), y_test_pred.min())
            max_val = max(y_test.max(), y_test_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_title('Actual vs. Predicted (Test Set)')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            
            plt.tight_layout()
            plt.savefig(img_io, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            img_io.seek(0)
            img = Image(ImageReader(img_io))
            img.drawHeight = 3*inch
            img.drawWidth = 6*inch
            
            story.append(Paragraph("Actual vs. Predicted", styles['Heading2']))
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
            
            # 3. Top Errors Table
            story.append(Paragraph("Top 10 Highest Errors (Test Set)", styles['Heading2']))
            
            # Create table of top errors
            top_errors = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_test_pred,
                'Error': test_error,
                'AbsError': np.abs(test_error)
            }).sort_values('AbsError', ascending=False).head(10)
            
            error_data = [["Index", "Actual", "Predicted", "Error", "Abs Error"]]
            
            for i, (idx, row) in enumerate(top_errors.iterrows(), 1):
                error_data.append([
                    str(idx),
                    f"{row['Actual']:.4f}",
                    f"{row['Predicted']:.4f}",
                    f"{row['Error']:.4f}",
                    f"{row['AbsError']:.4f}"
                ])
            
            error_table = Table(error_data, colWidths=[0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            error_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(error_table)
        
        # Build the PDF
        doc.build(story)


#------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------

def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    window = DecisionTreeAnalysisApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()