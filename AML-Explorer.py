import sys
import pandas as pd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QListWidget, QVBoxLayout, QDialog,
    QDialogButtonBox,QHBoxLayout, QGroupBox, QComboBox, QTextEdit, QFileDialog, QWidget, QMessageBox,QProgressDialog,QSpinBox
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



MAIN_FONT_STYLE = "font-size:18px;"
SECONDARY_FONT_STYLE = "font-size:16px;" 


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
        # central_layout.addWidget(self.far_right_panel)
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

     

        h_layout = QHBoxLayout()
        self.test_split_percentage_label = QLabel("Test Split Percentage:")
        self.test_split_percentage_label.setStyleSheet(SECONDARY_FONT_STYLE)
        h_layout.addWidget(self.test_split_percentage_label)
        
        self.test_split_percentage_input = QSpinBox()
        self.test_split_percentage_input.setStyleSheet(SECONDARY_FONT_STYLE)
        self.test_split_percentage_input.setMinimum(1)
        self.test_split_percentage_input.setMaximum(100)
        self.test_split_percentage_input.setValue(20)
        h_layout.addWidget(self.test_split_percentage_input)
        left_layout.addLayout(h_layout)

        # Random state
        h_layout = QHBoxLayout()
        self.random_state_label = QLabel("Random state:")
        self.random_state_label.setStyleSheet(SECONDARY_FONT_STYLE)
        h_layout.addWidget(self.random_state_label)
        
        self.random_state_input = QSpinBox()
        self.random_state_input.setStyleSheet(SECONDARY_FONT_STYLE)
        self.random_state_input.setMinimum(1)
        self.random_state_input.setMaximum(100)
        self.random_state_input.setValue(42)
        h_layout.addWidget(self.random_state_input)
        left_layout.addLayout(h_layout)

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
        self.results_display.setStyleSheet(MAIN_FONT_STYLE)
        self.results_display.setReadOnly(True)
        right_layout.addWidget(self.results_display)


        self.dialog = QDialog(self)
        self.dialog.setWindowTitle("Plot")
        self.features_to_plot_list= QListWidget()
        self.features_to_plot_list.setSelectionMode(QListWidget.MultiSelection)
        self.features_to_plot_list.setStyleSheet(SECONDARY_FONT_STYLE)
        self.features_to_plot_list.setFixedSize(300, 380)

        h_layout_1 = QHBoxLayout()
        h_layout_2 = QHBoxLayout()
        h_layout_3 = QHBoxLayout()
        # Plot buttons
        self.plot_confusion_button = QPushButton("Confusion Matrix")
        
        self.plot_confusion_button.setStyleSheet(MAIN_FONT_STYLE)
        self.plot_confusion_button.clicked.connect(self.plot_confusion_matrix)

        h_layout_1.addWidget(self.plot_confusion_button)

        self.plot_correlation_button = QPushButton("Corr Heatmap")
        self.plot_correlation_button.setStyleSheet(MAIN_FONT_STYLE)
        self.plot_correlation_button.clicked.connect(self.plot_correlation_heatmap)
        # self.plot_correlation_button.clicked.connect(self.plot_correlation_heatmap)
        h_layout_1.addWidget(self.plot_correlation_button)

        self.plot_pairplot_button = QPushButton("Plot Pairplot")
        self.plot_pairplot_button.setStyleSheet(MAIN_FONT_STYLE)
        self.plot_pairplot_button.clicked.connect(self.plot_pairplot)
        h_layout_1.addWidget(self.plot_pairplot_button)


        self.plot_scatter_button = QPushButton("Plot Scatter")
        self.plot_scatter_button.setStyleSheet(MAIN_FONT_STYLE)
        self.plot_scatter_button.clicked.connect(self.plot_scatter)
        h_layout_2.addWidget(self.plot_scatter_button)

        self.plot_histogram_button = QPushButton("Plot Histogram")
        self.plot_histogram_button.setStyleSheet(MAIN_FONT_STYLE)
        self.plot_histogram_button.clicked.connect(self.plot_histogram)
        h_layout_2.addWidget(self.plot_histogram_button)

        self.plot_boxplot_button = QPushButton("Plot Boxplot") 
        self.plot_boxplot_button.setStyleSheet(MAIN_FONT_STYLE)
        self.plot_boxplot_button.clicked.connect(self.plot_boxplot)
        h_layout_2.addWidget(self.plot_boxplot_button)

        self.plot_barplot_button = QPushButton("Plot Barplot")
        self.plot_barplot_button.setStyleSheet(MAIN_FONT_STYLE)
        self.plot_barplot_button.clicked.connect(self.plot_barplot)
        h_layout_3.addWidget(self.plot_barplot_button)

        self.violin_plot_button = QPushButton("Violin Plot")
        self.violin_plot_button.setStyleSheet(MAIN_FONT_STYLE)
        self.violin_plot_button.clicked.connect(self.plot_violinPlot)
        h_layout_3.addWidget(self.violin_plot_button)

        self.line_plot_button = QPushButton("Line Plot")
        self.line_plot_button.setStyleSheet(MAIN_FONT_STYLE)
        self.line_plot_button.clicked.connect(self.plot_linePlot)
        h_layout_3.addWidget(self.line_plot_button)

        right_layout.addLayout(h_layout_1)
        right_layout.addLayout(h_layout_2)
        right_layout.addLayout(h_layout_3)


        self.right_panel.setLayout(right_layout)

       


    def open_dialog(self):        
        self.dialog_layout = QVBoxLayout(self)
        self.dialog.setFixedSize(500, 400)
        self.dialog_layout.addWidget(self.features_to_plot_list)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)    
        self.button_box.accepted.connect(self.accept_dialog)
        self.button_box.rejected.connect(self.dialog.reject)
        self.dialog_layout.addWidget(self.button_box)
        self.dialog.setLayout(self.dialog_layout)
        self.dialog.exec_()
            

    def accept_dialog(self):
        self.set_features_to_plot()
        self.dialog.accept()


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
                self.features_to_plot_list.addItem(column)


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

        test_split = self.test_split_percentage_input.value() /100
        random_state = self.random_state_input.value()
        print("train_test_split",test_split)  
        print("random_state",random_state)
        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)

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
            # QMessageBox.information(self, "Training Complete", "Training completed successfully!")
            self.results_display.setText(result)




    def set_features_to_plot(self):
        selected_items = self.features_to_plot_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one feature")
            return
        self.features_columns_to_plot = [item.text() for item in selected_items]
        QMessageBox.information(self, "Features Selected", f"Features: {', '.join(self.features_columns_to_plot)}")

    def plot_confusion_matrix(self):
        self.open_dialog()
        if self.model is None:
            QMessageBox.critical(self, "Error", "Please train a model first")
            return
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
        
        self.open_dialog()

        try:
            dataset = self.dataset[self.features_columns_to_plot]
            if self.dataset is not None:
                sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm")
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


    def plot_scatter(self):
        self.open_dialog()
        if self.target_column == "":
            QMessageBox.critical(self, "Error", "Please select a target column first")
            return
        
        try:
            if self.dataset is not None and self.features_columns_to_plot:
                sns.scatterplot(data=self.dataset, x=self.features_columns_to_plot[0], y=self.features_columns_to_plot[1], hue=self.target_column)
                plt.title("Scatter Plot")
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    
    def plot_histogram(self):
        self.open_dialog()
        if self.target_column == "":
            QMessageBox.critical(self, "Error", "Please select a target column first")
            return
        try:
            if self.dataset is not None and self.features_columns_to_plot:
                sns.histplot(data=self.dataset, x=self.features_columns_to_plot[0], hue=self.target_column)
                plt.title("Histogram")
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_boxplot(self):
        self.open_dialog()
        if self.target_column == "":
            QMessageBox.critical(self, "Error", "Please select a target column first")
            return
        try:
            if self.dataset is not None and self.features_columns_to_plot:
                sns.boxplot(data=self.dataset, x=self.features_columns_to_plot[0], y=self.features_columns_to_plot[1], hue=self.target_column)
                plt.title("Boxplot")
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_barplot(self):
        self.open_dialog()
        if self.target_column == "":
            QMessageBox.critical(self, "Error", "Please select a target column first")
            return
        try:
            if self.dataset is not None and self.features_columns_to_plot:
                sns.barplot(data=self.dataset, x=self.features_columns_to_plot[0], y=self.features_columns_to_plot[1], hue=self.target_column)
                plt.title("Barplot")
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    
    def plot_violinPlot(self):
        self.open_dialog()
        if self.target_column == "":
            QMessageBox.critical(self, "Error", "Please select a target column first")
            return
        try:
            if self.dataset is not None and self.features_columns_to_plot:
                sns.violinplot(data=self.dataset, x=self.features_columns_to_plot[0], y=self.features_columns_to_plot[1], hue=self.target_column)
                plt.title("Violin Plot")
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_linePlot(self):
        self.open_dialog()
        if self.target_column == "":
            QMessageBox.critical(self, "Error", "Please select a target column first")
            return
        try:
            if self.dataset is not None and self.features_columns_to_plot:
                sns.lineplot(data=self.dataset, x=self.features_columns_to_plot[0], y=self.features_columns_to_plot[1], hue=self.target_column)
                plt.title("Line Plot")
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
        self.features_columns_to_plot = []
        self.features_to_plot_list.clear()
        self.test_split_percentage_input.setValue(20)
        self.random_state_input.setValue(42)
        QMessageBox.information(self, "Cleared", "All selections cleared.")

# Run application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec_())
