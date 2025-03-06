import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

class ModelExplainerGUI:
    def __init__(self, root, models):
        self.root = root
        self.models = models
        self.root.title("Model Prediction and Explanation GUI")
        
        # Set a larger window size
        self.root.geometry("600x600")
        
        # Center the window
        window_width = 600
        window_height = 600
        
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
        self.model_choice = ttk.Combobox(root, values=list(self.models.keys()), width=20)
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

    def make_prediction(self):
        model_name = self.model_choice.get()
        if not model_name:
            messagebox.showerror("Error", "Please select a model.")
            return

        # Collect user inputs
        try:
            input_data = np.array([float(self.entries[feat].get()) for feat in self.relevant_features]).reshape(1, -1)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values.")
            return

        model = self.models[model_name]
        prediction = model.predict(input_data)
        messagebox.showinfo("Prediction Result", f"Predicted score_text: {prediction[0]}")
        self.last_input = input_data
        self.last_model = model

    def explain_local(self):
        explainer_type = self.local_explainer_choice.get()
        if not explainer_type or self.last_model is None:
            messagebox.showerror("Error", "Make a prediction first and select a local explainer.")
            return

        if explainer_type == "SHAP":
            explainer = shap.TreeExplainer(self.last_model)
            shap_values = explainer.shap_values(self.last_input)
            shap.force_plot(explainer.expected_value[1], shap_values[1], self.last_input, matplotlib=True)
            plt.show()
        elif explainer_type == "LIME":
            explainer = lime.lime_tabular.LimeTabularExplainer(self.last_input, mode='classification')
            explanation = explainer.explain_instance(self.last_input[0], self.last_model.predict_proba)
            explanation.show_in_notebook()
        elif explainer_type == "Anchors":
            messagebox.showinfo("Info", "Anchors explanation method is not implemented yet.")

    def explain_global(self):
        explainer_type = self.global_explainer_choice.get()
        if not explainer_type or self.last_model is None:
            messagebox.showerror("Error", "Make a prediction first and select a global explainer.")
            return

        if explainer_type == "SHAP":
            explainer = shap.TreeExplainer(self.last_model)
            shap_values = explainer.shap_values(self.last_input)
            shap.summary_plot(shap_values[1], self.last_input, feature_names=self.relevant_features)
            plt.show()
        else:
            messagebox.showinfo("Info", "Global explanation for selected method is not available.")

# Example models dictionary (replace with actual trained models)
models = {
    "Random Forest": None,  # Replace with trained model instance
    "Decision Tree": None,
    "MLP NN": None,
    "Explainable Boosting Machine": None
}

root = tk.Tk()
app = ModelExplainerGUI(root, models)
root.mainloop()
