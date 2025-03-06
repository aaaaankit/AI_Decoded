import lime
import lime.lime_tabular
import numpy as np

class LimeAnalysis:
    def __init__(self, model, X_train, X_test, y_test, feature_names, class_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True
        )

    def perform_lime_analysis(self, test_idx):
        # Get the instance from the test set
        instance = self.X_test[test_idx]
        # Generate explanation for the instance
        explanation = self.explainer.explain_instance(instance, self.model.predict_proba, num_features=len(self.feature_names))
        
        print('LIME Explanation for instance %d:' % test_idx)
        try:
            # Attempt to use predict_proba
            prediction = self.model.predict_proba(instance.reshape(1, -1))
            print('Prediction: %s' % self.class_names[np.argmax(prediction)])
        except AttributeError:
            # If predict_proba is not available, fall back to predict
            prediction = self.model.predict(instance.reshape(1, -1))
            print('Prediction: %s' % self.class_names[prediction[0]])

        # Print the explanation
        print('LIME Explanation:')
        for feature, weight in explanation.as_list():
            print(f'{feature}: {weight}')
        #explanation.show_in_notebook(show_table=True, show_all=False)
        
        print('LIME explanation generated successfully.')

    def perform_lime_analysis_instance(self, instance):
        # Generate explanation for the given instance
        explanation = self.explainer.explain_instance(instance, self.model.predict_proba, num_features=len(self.feature_names))
        
        print('LIME Explanation for the given instance:')
        try:
            # Attempt to use predict_proba
            prediction = self.model.predict_proba(instance.reshape(1, -1))
            print('Prediction: %s' % self.class_names[np.argmax(prediction)])
        except AttributeError:
            # If predict_proba is not available, fall back to predict
            prediction = self.model.predict(instance.reshape(1, -1))
            print('Prediction: %s' % self.class_names[prediction[0]])

        # Print the explanation
        print('LIME Explanation:')
        for feature, weight in explanation.as_list():
            print(f'{feature}: {weight}')
        #explanation.show_in_notebook(show_table=True, show_all=False)

        print('LIME explanation generated successfully.')
