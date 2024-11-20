import sys
import pandas as pd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QListWidget, QVBoxLayout, 
    QHBoxLayout, QGroupBox, QComboBox, QTextEdit, QFileDialog, QWidget, QMessageBox,QProgressDialog
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

class TrainThread(QThread):
    finished = pyqtSignal(str)  # Signal to indicate completion with results

    def __init__(self, X_train, X_test, y_train, y_test, model):
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model

    def run(self):
        try:
            # Train the model
            self.model.fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_test)

            # Evaluate
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)

            # Emit results
            self.finished.emit(f"Accuracy: {accuracy * 100:.2f}%\n\n{report}")
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")


MAIN_FONT_STYLE = "font-size:18px;"
SECONDARY_FONT_STYLE = "font-size:16px;" 
class MLApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning GUI App")
        self.setGeometry(100, 100, 1000, 800)

        # Initialize variables
        self.dataset = None
        self.features_columns = []
        self.target_column = ""
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.model = None

        # Main layout
        self.main_layout = QVBoxLayout()
        self.create_left_panel()
        self.create_right_panel()

        # Central widget
        central_widget = QWidget()
        central_layout = QHBoxLayout()
        central_layout.addWidget(self.left_panel)
        central_layout.addWidget(self.right_panel)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

    def create_left_panel(self):
        # Left panel for controls
        self.left_panel = QGroupBox("Controls")
        left_layout = QVBoxLayout()
        
        # Load CSV button
        self.load_button = QPushButton("Load CSV File")
        self.load_button.setStyleSheet(MAIN_FONT_STYLE)
        self.load_button.clicked.connect(self.load_data)
        left_layout.addWidget(self.load_button)

        # Features selection
        self.features_label = QLabel("Select Features")
        self.features_label.setStyleSheet(SECONDARY_FONT_STYLE)
        
        
        left_layout.addWidget(self.features_label)

        self.features_list = QListWidget()
        self.features_list.setSelectionMode(QListWidget.MultiSelection)
        self.features_list.setStyleSheet(SECONDARY_FONT_STYLE)
        left_layout.addWidget(self.features_list)

        self.set_features_button = QPushButton("Set Features")
        self.set_features_button.setStyleSheet(MAIN_FONT_STYLE)
        self.set_features_button.clicked.connect(self.set_features)
        left_layout.addWidget(self.set_features_button)

        # Target selection
        self.target_label = QLabel("Select Target Column:")
        self.target_label.setStyleSheet(SECONDARY_FONT_STYLE)
        left_layout.addWidget(self.target_label)

        self.target_list = QListWidget()
        self.target_list.setStyleSheet(SECONDARY_FONT_STYLE)
        left_layout.addWidget(self.target_list)

        self.set_target_button = QPushButton("Set Target")
        self.set_target_button.setStyleSheet(MAIN_FONT_STYLE)
        self.set_target_button.clicked.connect(self.set_target)
        left_layout.addWidget(self.set_target_button)

        # Algorithm selection
        self.algorithm_label = QLabel("Select Algorithm:")
        self.algorithm_label.setStyleSheet(SECONDARY_FONT_STYLE)
        left_layout.addWidget(self.algorithm_label)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "Random Forest", "SVM", "Logistic Regression", "Decision Tree",
            "KNN", "Naive Bayes", "Ridge", "Gradient Boosting"
        ])
        self.algorithm_combo.setStyleSheet(SECONDARY_FONT_STYLE)
        left_layout.addWidget(self.algorithm_combo)

        self.train_button = QPushButton("Train Model")
        self.train_button.setStyleSheet(MAIN_FONT_STYLE)
        self.train_button.clicked.connect(self.train_model)
        left_layout.addWidget(self.train_button)

        self.clear_button = QPushButton("Clear All")
        
        self.clear_button.setStyleSheet(f"background-color: red; color: white;{MAIN_FONT_STYLE}")
        self.clear_button.clicked.connect(self.clear_all)
        left_layout.addWidget(self.clear_button)

        self.left_panel.setLayout(left_layout)

    def create_right_panel(self):
        # Right panel for results and plots
        self.right_panel = QGroupBox("Results and Plots")
        right_layout = QVBoxLayout()

        # Dataset info display
        self.dataset_info = QTextEdit()
        self.dataset_info.setStyleSheet(SECONDARY_FONT_STYLE)
        self.dataset_info.setReadOnly(True)
        right_layout.addWidget(self.dataset_info)

        # Model results display
        self.results_display = QTextEdit()
        self.results_display.setStyleSheet(SECONDARY_FONT_STYLE)
        self.results_display.setReadOnly(True)
        right_layout.addWidget(self.results_display)

        # Plot buttons
        self.plot_confusion_button = QPushButton("Plot Confusion Matrix")
        self.plot_confusion_button.setStyleSheet(MAIN_FONT_STYLE)
        self.plot_confusion_button.clicked.connect(self.plot_confusion_matrix)
        right_layout.addWidget(self.plot_confusion_button)

        self.plot_correlation_button = QPushButton("Plot Correlation Heatmap")
        self.plot_correlation_button.setStyleSheet(MAIN_FONT_STYLE)
        self.plot_correlation_button.clicked.connect(self.plot_correlation_heatmap)
        right_layout.addWidget(self.plot_correlation_button)

        self.plot_pairplot_button = QPushButton("Plot Pairplot")
        self.plot_pairplot_button.setStyleSheet(MAIN_FONT_STYLE)
        self.plot_pairplot_button.clicked.connect(self.plot_pairplot)
        right_layout.addWidget(self.plot_pairplot_button)

        self.right_panel.setLayout(right_layout)

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.dataset = pd.read_csv(file_path)
            QMessageBox.information(self, "Info", f"Data Loaded: {self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns")

            # Populate features and target lists
            self.features_list.clear()
            self.target_list.clear()
            for column in self.dataset.columns:
                self.features_list.addItem(column)
                self.target_list.addItem(column)


            #remove nan value
            self.dataset = self.dataset.dropna()

            # Display dataset info
            self.dataset_info.setText(str(self.dataset.dtypes))

    def set_features(self):
        selected_items = self.features_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one feature")
            return
        self.features_columns = [item.text() for item in selected_items]
        QMessageBox.information(self, "Features Selected", f"Features: {', '.join(self.features_columns)}")

    def set_target(self):
        selected_items = self.target_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a target column")
            return
        self.target_column = selected_items[0].text()
        QMessageBox.information(self, "Target Selected", f"Target: {self.target_column}")

    def train_model(self):
        if self.dataset is None:
            QMessageBox.critical(self, "Error", "No dataset loaded")
            return
        if not self.features_columns or not self.target_column:
            QMessageBox.critical(self, "Error", "Select features and target column")
            return

        # Prepare data
        X = self.dataset[self.features_columns]
        y = self.dataset[self.target_column]

        # Handle non-numeric columns in features
        for column in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])

        # Handle non-numeric target column (if necessary)
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)

        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Select model
        algorithm = self.algorithm_combo.currentText()
        if algorithm == "Random Forest":
            self.model = RandomForestClassifier()
        elif algorithm == "SVM":
            self.model = SVC()
        elif algorithm == "Logistic Regression":
            self.model = LogisticRegression()
        elif algorithm == "Decision Tree":
            self.model = DecisionTreeClassifier()
        elif algorithm == "KNN":
            self.model = KNeighborsClassifier()
        elif algorithm == "Naive Bayes":
            self.model = GaussianNB()
        elif algorithm == "Ridge":
            self.model = Ridge()
        elif algorithm == "Gradient Boosting":
            self.model = GradientBoostingClassifier()


        # show Loading dialog
        progress_dialog = QProgressDialog("Training Model...", "Cancel", 0, 0, self)
        progress_dialog.setWindowTitle("Please Wait!")
        progress_dialog.setModal(True)
        progress_dialog.show()
        # Train and evaluate
        self.train_thread = TrainThread( self.X_train, self.X_test, self.y_train, self.y_test , self.model)
        self.train_thread.finished.connect(lambda result: self.on_training_finished(result, progress_dialog))
        self.train_thread.start()
        
    def on_training_finished(self, result, progress_dialog):
        progress_dialog.close()
        if result.startswith("Error"):
            QMessageBox.critical(self, "Training Error", result)
        else:
            QMessageBox.information(self, "Training Complete", "Training completed successfully!")
            self.results_display.setText(result)

    def plot_confusion_matrix(self):
        try:
            if self.model and self.X_test is not None:
                y_pred = self.model.predict(self.X_test)
                cm = confusion_matrix(self.y_test, y_pred)
                sns.heatmap(cm, annot=True, cmap="Blues")
                plt.title("Confusion Matrix")
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_correlation_heatmap(self):
        try:
            if self.dataset is not None:
                sns.heatmap(self.dataset.corr(), annot=True, cmap="coolwarm")
                plt.title("Correlation Heatmap")
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_pairplot(self):
        try:
            if self.dataset is not None and self.features_columns:
                sns.pairplot(self.dataset[self.features_columns])
                plt.title("Feature Pairplot")
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def clear_all(self):
        self.dataset = None
        self.features_columns = []
        self.target_column = ""
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.features_list.clear()
        self.target_list.clear()
        self.dataset_info.clear()
        self.results_display.clear()
        QMessageBox.information(self, "Cleared", "All selections cleared.")

# Run application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec_())
