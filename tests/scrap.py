from sklearn import metrics
import pandas as pd
import wandb

wandb.init(entity='surajpai', project='FacialEmotionRecognition')

y_test = [1, 0, 2]
y_pred_class = [1, 0, 1]


report = metrics.classification_report(y_test, y_pred_class, output_dict=True)

df = pd.DataFrame(report)
data = df.values.tolist()

print(data)
print(df.columns)
table = wandb.Table(data=data, columns=df.columns.values)

wandb.log({"examples": table})