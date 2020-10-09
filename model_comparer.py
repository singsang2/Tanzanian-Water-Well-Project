from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, roc_auc_score, roc_curve, auc

class ModelCompare:

    def __init__(self):
        self.models = []
        self.round_num = 1



    def add_model(self, models):
        self.models.append(models)
        self.round_num += 1