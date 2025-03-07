import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import json
import os
import Prediction_System as Prediction_System
import General_System as General_System

class ModelExplainerGUI:
    def __init__(self, root, models):
        self.root = root
        self.models = models
        self.root.title("Model Prediction and Explanation GUI")
        
        # Set a larger window size
        self.root.geometry("600x600")
        
        # Center the window
        window_width = 600
        window_height = 750
        
        # Get screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate the position to center the window
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        
        # Set the position of the window
        self.root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')
        
        # Apply a modern style to the widget
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", font=("Helvetica", 12))
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("TCombobox", font=("Helvetica", 12))
        style.configure("TEntry", font=("Helvetica", 12))

        # Feature Input Fields
        self.entries = {}
        self.relevant_features = ["sex","age","race","juv_fel_count","juv_misd_count","juv_other_count",
                                  "c_charge_degree","r_charge_degree","r_days_from_arrest","vr_charge_degree"]
        ttk.Label(root, text="Enter Feature Values:").grid(row=0, column=0, columnspan=2, pady=10)

        for i, feature in enumerate(self.relevant_features):
            ttk.Label(root, text=feature).grid(row=i + 1, column=0, sticky="w", padx=10, pady=5)
            entry = ttk.Entry(root)
            entry.grid(row=i + 1, column=1, padx=10, pady=5)
            self.entries[feature] = entry

        # Model Selection
        ttk.Label(root, text="Select Model:").grid(row=len(self.relevant_features) + 1, column=0, sticky="w", padx=10, pady=10)
        self.model_choice = ttk.Combobox(root, values=list(self.models), width=20)
        self.model_choice.grid(row=len(self.relevant_features) + 1, column=1, padx=10, pady=5)
        
        # Local Explanation Method Selection
        ttk.Label(root, text="Select Local Explanation Method:").grid(row=len(self.relevant_features) + 2, column=0, sticky="w", padx=10, pady=10)
        self.local_explainer_choice = ttk.Combobox(root, values=["SHAP", "LIME", "Anchors"], width=20)
        self.local_explainer_choice.grid(row=len(self.relevant_features) + 2, column=1, padx=10, pady=5)

        # Global Explanation Method Selection
        ttk.Label(root, text="Select Global Explanation Method:").grid(row=len(self.relevant_features) + 3, column=0, sticky="w", padx=10, pady=10)
        self.global_explainer_choice = ttk.Combobox(root, values=["SHAP"], width=20)  # Assuming only SHAP is available for global
        self.global_explainer_choice.grid(row=len(self.relevant_features) + 3, column=1, padx=10, pady=5)

        # Predict Button
        self.predict_btn = ttk.Button(root, text="Predict", command=self.make_prediction)
        self.predict_btn.grid(row=len(self.relevant_features) + 4, column=0, columnspan=2, pady=15)

        # Explanation Section (Separate local and global)
        ttk.Label(root, text="Local Explanation:").grid(row=len(self.relevant_features) + 5, column=0, sticky="w", padx=10, pady=10)
        self.explain_local_btn = ttk.Button(root, text="Explain Local", command=self.explain_local)
        self.explain_local_btn.grid(row=len(self.relevant_features) + 5, column=1, padx=10, pady=5)

        ttk.Label(root, text="Global Explanation:").grid(row=len(self.relevant_features) + 6, column=0, sticky="w", padx=10, pady=10)
        self.explain_global_btn = ttk.Button(root, text="Explain Global", command=self.explain_global)
        self.explain_global_btn.grid(row=len(self.relevant_features) + 6, column=1, padx=10, pady=5)

        # See Model Performance Button
        ttk.Label(root, text="Model Performance:").grid(row=len(self.relevant_features) + 7, column=0, sticky="w", padx=10, pady=10)
        self.model_performance_btn = ttk.Button(root, text="See Performance", command=self.see_model_performance)
        self.model_performance_btn.grid(row=len(self.relevant_features) + 7, column=1, padx=10, pady=5)


    def make_prediction(self):
        # Get the model and the explainer
        model_name = self.model_choice.get()
        local_explainer = self.local_explainer_choice.get()
        if not model_name:
            messagebox.showerror("Error", "Please select a model.")
            return

        # Collect user inputs
        self.save_user_input()
        data_point = self.read_user_input()

        #Make prediction on the input
        predictor = Prediction_System.Predictor(classification_model=model_name,local_explainer=local_explainer,data_point=data_point)
        prediction = predictor.classification()

        messagebox.showinfo("Prediction Result", f"Predicted score_text: {prediction[0]}")

        # Save the model and data for the explanations
        self.last_input = data_point
        self.last_model = model_name

    def explain_local(self):
        explainer_type = self.local_explainer_choice.get()
        if not explainer_type or self.last_model is None:
            messagebox.showerror("Error", "Make a prediction first and select a local explainer.")
            return
        
        predictor = Prediction_System.Predictor(classification_model=self.last_model,local_explainer=explainer_type,data_point=self.last_input)
        predictor.local_explanation()


    def explain_global(self):
        explainer_type = self.global_explainer_choice.get()
        model_name = self.model_choice.get()
        if not explainer_type or model_name is None:
            messagebox.showerror("Error", "Select a global explainer and a classification model.")
            return

        general_explainer = General_System.General(classification_model=model_name,global_explainer=explainer_type)
        general_explainer.global_explanation()

    def see_model_performance(self):
        explainer_type = self.global_explainer_choice.get()
        model_name = self.model_choice.get()
        if not explainer_type or model_name is None:
            messagebox.showerror("Error", "Select a global explainer and a classification model.")
            return

        general_explainer = General_System.General(classification_model=model_name,global_explainer=explainer_type)
        general_explainer.performance_evaluation()


    def save_user_input(self):
        user_data = {}
        for feat in self.relevant_features:
            value = self.entries[feat].get()
            try:
                # Convert numerical values to int/float
                if value.replace('.', '', 1).isdigit():
                    user_data[feat] = float(value) if '.' in value else int(value)
                else:
                    user_data[feat] = value  # Keep strings as they are
            except ValueError:
                messagebox.showerror("Error", f"Invalid input for {feat}.")
                return

        # Save to a .txt file
        file_path = "AI_Decoded/dataPoint.txt"
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(user_data, file, indent=4)
            messagebox.showinfo("Success", "User input saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {e}")

    def read_user_input():
        file_path = "AI_Decoded/dataPoint.txt"

        if not os.path.exists(file_path):
            messagebox.showerror("Error", "No saved data found.")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                user_data = json.load(file)
            messagebox.showinfo("Success", "User input loaded successfully.")
            return user_data
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {e}")
            return None




models = ["Random Forest", "Decision Tree", "MLP NN", "Explainable Boosting Machine"]

root = tk.Tk()
app = ModelExplainerGUI(root, models)
root.mainloop()
