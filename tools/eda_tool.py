import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QTabWidget, 
                             QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QComboBox, 
                             QSplitter, QTextEdit, QMessageBox, QProgressBar,
                             QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from datetime import datetime
from PyQt5.QtWidgets import QListWidget, QListWidgetItem


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


class MatplotlibCanvas(FigureCanvas):
    """Canvas for matplotlib plots"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class EDAApp(QMainWindow):
    """Main application window for Exploratory Data Analysis"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.setWindowTitle("PyQt5 Simple EDA Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data storage
        self.df = None
        self.file_path = None
        
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
        self.create_stats_tab()
        self.create_distribution_tab()
        self.create_correlation_tab()
        
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
    
    def create_stats_tab(self):
        """Create tab for statistical analysis"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Controls panel
        controls = QHBoxLayout()
        
        # Statistics type selection
        self.stats_type = QComboBox()
        self.stats_type.addItems(["Basic Statistics", "Correlation Matrix"])
        self.stats_type.currentTextChanged.connect(self.update_stats_display)
        controls.addWidget(QLabel("Statistics Type:"))
        controls.addWidget(self.stats_type)
        
        controls.addStretch()
        # In create_stats_tab (after controls panel)
        self.column_selector = QListWidget()
        self.column_selector.setSelectionMode(QListWidget.MultiSelection)
        self.column_selector.itemSelectionChanged.connect(self.display_correlation_matrix)

        layout.addWidget(QLabel("Select Columns for Correlation Matrix:"))
        layout.addWidget(self.column_selector)
        self.column_selector.setVisible(False)  # Only show when correlation matrix selected
        # Save button
        self.save_stats_btn = QPushButton("Save Statistics")
        self.save_stats_btn.clicked.connect(self.save_statistics)
        controls.addWidget(self.save_stats_btn)
        
        layout.addLayout(controls)
        
        # Stats table
        self.stats_table = QTableWidget()
        layout.addWidget(self.stats_table)
        
        self.tabs.addTab(tab, "Statistics")
    
    def create_distribution_tab(self):
        """Create tab for distribution plots"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Controls panel
        controls = QHBoxLayout()
        
        # Variable selection
        self.dist_var = QComboBox()
        controls.addWidget(QLabel("Select Variable:"))
        controls.addWidget(self.dist_var)
        
        # Plot type selection
        self.dist_plot_type = QComboBox()
        self.dist_plot_type.addItems(["Histogram", "Box Plot", "Violin Plot", "KDE"])
        controls.addWidget(QLabel("Plot Type:"))
        controls.addWidget(self.dist_plot_type)
        
        # Update button
        self.update_dist_btn = QPushButton("Update Plot")
        self.update_dist_btn.clicked.connect(self.update_distribution_plot)
        controls.addWidget(self.update_dist_btn)
        
        controls.addStretch()
        
        # Save button
        self.save_dist_btn = QPushButton("Save Plot")
        self.save_dist_btn.clicked.connect(self.save_distribution_plot)
        controls.addWidget(self.save_dist_btn)
        
        layout.addLayout(controls)
        
        # Plot canvas
        self.dist_canvas = MatplotlibCanvas(self, width=5, height=4)
        self.dist_toolbar = NavigationToolbar(self.dist_canvas, self)
        
        layout.addWidget(self.dist_toolbar)
        layout.addWidget(self.dist_canvas)
        
        self.tabs.addTab(tab, "Distributions")
    
    def create_correlation_tab(self):
        """Create tab for correlation analysis"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Controls panel
        controls = QHBoxLayout()
        
        # Plot type selection
        self.corr_plot_type = QComboBox()
        self.corr_plot_type.addItems(["Heatmap", "Pairplot", "Scatter Plot"])
        controls.addWidget(QLabel("Plot Type:"))
        controls.addWidget(self.corr_plot_type)
        
        # For scatter plot, need x and y variables
        self.x_var = QComboBox()
        self.y_var = QComboBox()
        controls.addWidget(QLabel("X:"))
        controls.addWidget(self.x_var)
        controls.addWidget(QLabel("Y:"))
        controls.addWidget(self.y_var)
        
        # Update button
        self.update_corr_btn = QPushButton("Update Plot")
        self.update_corr_btn.clicked.connect(self.update_correlation_plot)
        controls.addWidget(self.update_corr_btn)
        
        controls.addStretch()
        
        # Save button
        self.save_corr_btn = QPushButton("Save Plot")
        self.save_corr_btn.clicked.connect(self.save_correlation_plot)
        controls.addWidget(self.save_corr_btn)
        
        layout.addLayout(controls)
        
        # Plot canvas
        self.corr_canvas = MatplotlibCanvas(self, width=5, height=4)
        self.corr_toolbar = NavigationToolbar(self.corr_canvas, self)
        
        layout.addWidget(self.corr_toolbar)
        layout.addWidget(self.corr_canvas)
        
        self.tabs.addTab(tab, "Correlations")
    
    
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
        self.save_btn.setEnabled(True)
        
        # Update all UI elements with the new data
        self.update_data_preview()
        self.update_variable_selectors()
        self.update_stats_display()
        
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
    
    def update_variable_selectors(self):
        """Update all combo boxes with column names"""
        if self.df is None:
            return
        
        # Get numeric columns for plots
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Update distribution variable selector
        self.dist_var.clear()
        self.dist_var.addItems(numeric_cols)
        
        # Update correlation variable selectors
        self.x_var.clear()
        self.y_var.clear()
        self.x_var.addItems(numeric_cols)
        self.y_var.addItems(numeric_cols)
        
        # Set different default selections if possible
        if len(numeric_cols) > 1:
            self.y_var.setCurrentIndex(1)
    
    def update_stats_display(self):
        """Update statistics table based on selected type"""
        if self.df is None:
            return
        
        stats_type = self.stats_type.currentText()
        
        if stats_type == "Basic Statistics":
            self.column_selector.setVisible(False)
            self.display_basic_stats()
        elif stats_type == "Correlation Matrix":
            self.column_selector.setVisible(True)
            self.update_column_selector()
            self.display_correlation_matrix()
    
    def display_basic_stats(self):
        """Display basic statistics for all numeric columns"""
        if self.df is None:
            return
        
        # Get numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Calculate statistics
        stats_df = numeric_df.describe().T
        
        # Add additional statistics
        stats_df['skew'] = numeric_df.skew()
        stats_df['kurtosis'] = numeric_df.kurtosis()
        stats_df['missing'] = numeric_df.isnull().sum()
        
        # Setup table
        self.stats_table.setRowCount(stats_df.shape[0])
        self.stats_table.setColumnCount(stats_df.shape[1])
        
        # Set headers
        self.stats_table.setHorizontalHeaderLabels(stats_df.columns)
        self.stats_table.setVerticalHeaderLabels(stats_df.index)
        
        # Populate table
        for i in range(stats_df.shape[0]):
            for j in range(stats_df.shape[1]):
                value = stats_df.iloc[i, j]
                if isinstance(value, (int, float, np.number)):
                    value_str = f"{value:.6g}"
                else:
                    value_str = str(value)
                item = QTableWidgetItem(value_str)
                self.stats_table.setItem(i, j, item)
        
        # Resize to content
        self.stats_table.resizeColumnsToContents()
    def update_column_selector(self):
        """Update column selector list"""
        if self.df is None:
            return
        
        self.column_selector.clear()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            item = QListWidgetItem(col)
            item.setSelected(True)  # Select all by default
            self.column_selector.addItem(item)

    def display_correlation_matrix(self):
        """Display correlation matrix for selected numeric columns"""
        if self.df is None:
            return
        
        selected_cols = [item.text() for item in self.column_selector.selectedItems()]
        
        if not selected_cols:
            QMessageBox.warning(self, "No Columns Selected", "Please select at least one column to display correlation matrix.")
            return
        
        corr = self.df[selected_cols].corr()
        
        # Setup table
        self.stats_table.setRowCount(corr.shape[0])
        self.stats_table.setColumnCount(corr.shape[1])
        
        # Set headers
        self.stats_table.setHorizontalHeaderLabels(corr.columns)
        self.stats_table.setVerticalHeaderLabels(corr.index)
        
        # Populate table with color gradient
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                value = corr.iloc[i, j]
                item = QTableWidgetItem(f"{value:.4f}")
                if value < 0:
                    red = min(255, int(255 * abs(value) * 0.8))
                    item.setBackground(QColor(red, 255 - red, 255 - red))
                else:
                    green = min(255, int(255 * value * 0.8))
                    item.setBackground(QColor(255 - green, 255, 255 - green))
                self.stats_table.setItem(i, j, item)
        
        self.stats_table.resizeColumnsToContents()

    
    def update_distribution_plot(self):
        """Update the distribution plot with selected variable and plot type"""
        if self.df is None:
            return
        
        try:
            # Get selected variable and plot type
            var_name = self.dist_var.currentText()
            plot_type = self.dist_plot_type.currentText()
            
            # Clear previous plot
            self.dist_canvas.fig.clear()
            ax = self.dist_canvas.fig.add_subplot(111)
            
            # Get data
            data = self.df[var_name].dropna()
            
            # Create selected plot type
            if plot_type == "Histogram":
                sns.histplot(data, kde=True, ax=ax)
                
                # Add mean and median lines
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
                ax.legend()
                
            elif plot_type == "Box Plot":
                sns.boxplot(y=data, ax=ax)
                ax.set_ylabel(var_name)
                
            elif plot_type == "Violin Plot":
                sns.violinplot(y=data, ax=ax)
                ax.set_ylabel(var_name)
                
            elif plot_type == "KDE":
                sns.kdeplot(data, ax=ax, fill=True)
            
            # Set title and labels
            ax.set_title(f"{plot_type} of {var_name}")
            if plot_type != "Box Plot" and plot_type != "Violin Plot":
                ax.set_xlabel(var_name)
                ax.set_ylabel("Frequency" if plot_type == "Histogram" else "Density")
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Draw the plot
            self.dist_canvas.fig.tight_layout()
            self.dist_canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Error creating distribution plot: {str(e)}")
    
    def update_correlation_plot(self):
        """Update the correlation plot with selected options"""
        if self.df is None:
            return
        
        try:
            # Get selected plot type
            plot_type = self.corr_plot_type.currentText()
            
            # Clear previous plot
            self.corr_canvas.fig.clear()
            
            # Get numeric columns for correlation
            numeric_df = self.df.select_dtypes(include=[np.number])
            
            # Create selected plot type
            if plot_type == "Heatmap":
                # Calculate correlation
                corr = numeric_df.corr()
                
                # Create heatmap
                ax = self.corr_canvas.fig.add_subplot(111)
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                           square=True, linewidths=0.5, ax=ax)
                ax.set_title("Correlation Matrix")
                
            elif plot_type == "Pairplot":
                # For pairplot, limit to 6 variables max for performance
                if numeric_df.shape[1] > 6:
                    # Select columns with highest variance
                    variances = numeric_df.var()
                    top_columns = variances.nlargest(6).index.tolist()
                    subset_df = numeric_df[top_columns]
                else:
                    subset_df = numeric_df
                
                # Create pairplot
                g = sns.pairplot(subset_df)
                g.fig.suptitle("Pairwise Relationships", y=1.02)
                
                # Set the generated figure to our canvas
                self.corr_canvas.fig = g.fig
                
            elif plot_type == "Scatter Plot":
                # Get selected x and y variables
                x_var = self.x_var.currentText()
                y_var = self.y_var.currentText()
                
                # Create scatter plot with regression line
                ax = self.corr_canvas.fig.add_subplot(111)
                sns.regplot(x=x_var, y=y_var, data=self.df, ax=ax, scatter_kws={'alpha': 0.5})
                
                # Calculate correlation coefficient
                corr_value = self.df[[x_var, y_var]].corr().iloc[0, 1]
                
                # Add correlation text to plot
                ax.annotate(f"Correlation: {corr_value:.4f}", 
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                ax.set_title(f"{y_var} vs {x_var}")
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Draw the plot
            self.corr_canvas.fig.tight_layout()
            self.corr_canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Error creating correlation plot: {str(e)}")
    
    
    
    def save_statistics(self):
        """Save the current statistics to a file"""
        if self.df is None or self.stats_table.rowCount() == 0:
            QMessageBox.warning(self, "Error", "No statistics to save")
            return
        
        try:
            stats_type = self.stats_type.currentText()
            
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Statistics", "", 
                "CSV Files (*.csv);;Excel Files (*.xlsx);;Text Files (*.txt);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Prepare data from table
            headers = []
            for col in range(self.stats_table.columnCount()):
                headers.append(self.stats_table.horizontalHeaderItem(col).text())
            
            rows = []
            row_headers = []
            for row in range(self.stats_table.rowCount()):
                row_data = []
                for col in range(self.stats_table.columnCount()):
                    item = self.stats_table.item(row, col)
                    row_data.append(item.text() if item else "")
                rows.append(row_data)
                
                # Get row header if available
                header_item = self.stats_table.verticalHeaderItem(row)
                row_headers.append(header_item.text() if header_item else f"Row {row}")
            
            # Convert to DataFrame for easy saving
            stats_df = pd.DataFrame(rows, columns=headers, index=row_headers)
            
            # Save based on file extension
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == '.csv':
                stats_df.to_csv(file_path)
            elif ext.lower() == '.xlsx':
                stats_df.to_excel(file_path)
            elif ext.lower() == '.txt':
                with open(file_path, 'w') as f:
                    f.write(f"{stats_type}\n\n")
                    f.write(stats_df.to_string())
            else:
                # Default to CSV
                stats_df.to_csv(file_path)
                
            QMessageBox.information(self, "Success", f"Statistics saved to {file_path}")
            
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Error saving statistics: {str(e)}")
    
    def save_distribution_plot(self):
        """Save the current distribution plot"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
            
        try:
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Plot", "", 
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
            )
            
            if file_path:
                self.dist_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot saved to {file_path}")
                
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Error saving plot: {str(e)}")
    
    def save_correlation_plot(self):
        """Save the current correlation plot"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
            
        try:
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Plot", "", 
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
            )
            
            if file_path:
                self.corr_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot saved to {file_path}")
                
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Error saving plot: {str(e)}")
    
    
    def save_results(self):
        """Save all results to a directory"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
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
            results_dir = os.path.join(dir_path, f"EDA_Results_{timestamp}")
            
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Save data statistics
            if self.stats_table.rowCount() > 0:
                # Basic stats
                self.stats_type.setCurrentText("Basic Statistics")
                self.update_stats_display()
                stats_path = os.path.join(results_dir, "basic_statistics.csv")
                
                # Prepare data from table
                headers = []
                for col in range(self.stats_table.columnCount()):
                    headers.append(self.stats_table.horizontalHeaderItem(col).text())
                
                rows = []
                row_headers = []
                for row in range(self.stats_table.rowCount()):
                    row_data = []
                    for col in range(self.stats_table.columnCount()):
                        item = self.stats_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    rows.append(row_data)
                    
                    # Get row header if available
                    header_item = self.stats_table.verticalHeaderItem(row)
                    row_headers.append(header_item.text() if header_item else f"Row {row}")
                
                # Convert to DataFrame for easy saving
                stats_df = pd.DataFrame(rows, columns=headers, index=row_headers)
                stats_df.to_csv(stats_path)
                
                # Correlation matrix
                self.stats_type.setCurrentText("Correlation Matrix")
                self.update_stats_display()
                corr_path = os.path.join(results_dir, "correlation_matrix.csv")
                
                # Prepare data from table
                headers = []
                for col in range(self.stats_table.columnCount()):
                    headers.append(self.stats_table.horizontalHeaderItem(col).text())
                
                rows = []
                row_headers = []
                for row in range(self.stats_table.rowCount()):
                    row_data = []
                    for col in range(self.stats_table.columnCount()):
                        item = self.stats_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    rows.append(row_data)
                    
                    # Get row header if available
                    header_item = self.stats_table.verticalHeaderItem(row)
                    row_headers.append(header_item.text() if header_item else f"Row {row}")
                
                # Convert to DataFrame for easy saving
                corr_df = pd.DataFrame(rows, columns=headers, index=row_headers)
                corr_df.to_csv(corr_path)
            
            # Save distribution plots for all numeric columns
            dist_dir = os.path.join(results_dir, "distributions")
            if not os.path.exists(dist_dir):
                os.makedirs(dist_dir)
                
            for var in self.df.select_dtypes(include=[np.number]).columns:
                self.dist_var.setCurrentText(var)
                self.dist_plot_type.setCurrentText("Histogram")
                self.update_distribution_plot()
                
                dist_path = os.path.join(dist_dir, f"{var}_histogram.png")
                self.dist_canvas.fig.savefig(dist_path, dpi=300, bbox_inches='tight')
            
            # Save correlation plots
            corr_dir = os.path.join(results_dir, "correlations")
            if not os.path.exists(corr_dir):
                os.makedirs(corr_dir)
                
            # Save heatmap
            self.corr_plot_type.setCurrentText("Heatmap")
            self.update_correlation_plot()
            heatmap_path = os.path.join(corr_dir, "correlation_heatmap.png")
            self.corr_canvas.fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            
            # Save pairplot
            self.corr_plot_type.setCurrentText("Pairplot")
            self.update_correlation_plot()
            pairplot_path = os.path.join(corr_dir, "pairplot.png")
            self.corr_canvas.fig.savefig(pairplot_path, dpi=300, bbox_inches='tight')
                        
            # Save processed data
            data_path = os.path.join(results_dir, "processed_data.csv")
            self.df.to_csv(data_path, index=False)
            
            QMessageBox.information(self, "Success", f"All results saved to {results_dir}")
            
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Error saving results: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = EDAApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()