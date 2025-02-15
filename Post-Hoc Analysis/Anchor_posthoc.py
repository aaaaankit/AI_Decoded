import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from alibi.explainers import AnchorTabular


    
class AnchorAnalysis:
    def __init__(self, model, X_train, X_test, y_test, feature_names, class_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        predict_fn = lambda x: model.predict_proba(x)
        self.explainer = AnchorTabular(predict_fn, feature_names, categorical_names={})
        self.explainer.fit(X_train)
        self.class_names = class_names

    def perform_anchor_analysis(self, test_idx):
        explanation = self.explainer.explain(self.X_test[test_idx].reshape(1, -1)) 
        print('Anchor: %s' % (' AND '.join(explanation.anchor)))
        print('Precision: %.2f' % explanation.precision)
        print('Coverage: %.2f' % explanation.coverage)
        # print instance like we did before
        instance = self.X_test[test_idx]