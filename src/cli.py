from modules.data_pipeline import pipeline, label_func
from fastai.vision.all import *

path = './trials/'
for file in os.listdir(path):
    if file.endswith('.png'):
        os.remove(os.path.join(path, file))
prediction = pipeline(path, './models/xgb', './models/cnn', check_threshold=-1, verbose=True)
if prediction == 'NA':
    print('No cough detected, please try again.')
else:
    print(f"""
COVID19 detected:           {bool(prediction[0])}
Prediction probability:     {(1-prediction[1])*100:.2f}%
Cough probability:          {(1-prediction[1])*100:.2f}%
    """)

