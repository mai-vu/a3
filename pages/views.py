from django.shortcuts import render
from keras.models import load_model
import pickle
import pandas as pd
import numpy as np


def homePageView(request):
    return render(request, 'home.html')


def evalCar(request):
    if request.method == 'POST':
        # Get form data
        buying = int(request.POST.get('buying'))
        maint = int(request.POST.get('maint'))
        safety = int(request.POST.get('safety'))
        lug_boot = int(request.POST.get('lug_boot'))
        persons = int(request.POST.get('persons'))

        # Load models
        ann_model = load_model('./BinaryFolder/best_ANN_model.h5')
        gb_model = pickle.load(open('./BinaryFolder/best_model.pkl', 'rb'))
        stacked_model = pickle.load(open('./BinaryFolder/stacked_model.pkl', 'rb'))
        models = [ann_model, gb_model]

        # Create input data for prediction
        X = pd.DataFrame(
            {'buying': [buying], 'maint': [maint], 'safety': [safety], 'persons': [persons],'lug_boot': [lug_boot]})

        # Predict
        dfPredictions = pd.DataFrame(index=X.index)
        for i, model in enumerate(models):
            predictions = model.predict(X)
            if i == 0:  # ANN model
                predictions = np.argmax(predictions, axis=1)
            colName = str(i)
            dfPredictions[colName] = predictions

        y_pred = stacked_model.predict(dfPredictions)


        if y_pred[0] == 0:
            y_pred = 'Unacceptable'
        elif y_pred[0] == 1:
            y_pred = 'Acceptable'
        elif y_pred[0] == 2:
            y_pred = 'Good'
        elif y_pred[0] == 3:
            y_pred = 'Very Good'
        else:
            y_pred = 'Unknown'

        # Pass data to result page
        return render(request, 'result.html', {'predicted_class': y_pred})

    return render(request, 'home.html')


#