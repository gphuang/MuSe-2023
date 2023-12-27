import pandas as pd
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score



print("Comparing")
print("System 1: "+sys.argv[1])
print("System 1: "+sys.argv[2])

#dev comparison:
dev1 = pd.read_csv(sys.argv[1]+"predictions_devel.csv")
dev2 = pd.read_csv(sys.argv[2]+"predictions_devel.csv")
dev1_pred = dev1.prediction
dev2_pred = dev2.prediction
dev_label = dev1.label
print("----System 1 devel performance----")
print("Acc: ", accuracy_score(dev_label, dev1_pred>0.5))
print("ROC-AUC: ", roc_auc_score(dev_label, dev1_pred))
print("Confusion matrix:\n", confusion_matrix(dev_label, dev1_pred>0.5))

print("----System 2 devel performance----")
print("Acc: ", accuracy_score(dev_label, dev2_pred>0.5))
print("ROC-AUC: ", roc_auc_score(dev_label, dev2_pred))
print("Confusion matrix:\n", confusion_matrix(dev_label, dev2_pred>0.5))

#test comparison
test1 = pd.read_csv(sys.argv[1]+"predictions_test.csv")
test2 = pd.read_csv(sys.argv[2]+"predictions_test.csv")
test1_pred = test1.prediction
test2_pred = test2.prediction
print("----System 1 vs System 2 test comparison----")
print("Acc: ", accuracy_score(test1_pred>0.5, test2_pred>0.5))
#print("ROC-AUC: ", roc_auc_score(test1_pred>0.5, test2_pred))
print("Confusion matrix:\n", confusion_matrix(test1_pred>0.5, test2_pred>0.5))
