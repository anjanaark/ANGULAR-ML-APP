from datetime import datetime
import os
import itertools
import warnings
from flask import Flask, jsonify, request, session, send_file, Response
from flask_cors import CORS
import pandas as pd
from statsmodels.graphics.tsaplots import plot_predict
import numpy as np
import pickle
from flask import send_file
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 8, 6
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your secret key'
CORS(app)


@app.route('/', methods=['POST', 'GET'])
def hello_world():
    req = request.get_json()
    oem = req['oem']
    model = req['model']
    body = req['body']
    price = float(req['price'])
    years = int(req['years'])
    month = int(req['month'])
    scaler_path1 = r'C:\Users\User\Desktop\dsappkaar\backend\linearmodel.sav'
    model1 = pickle.load(open(scaler_path1, "rb"))
    scaler_path2 = r'C:\Users\User\Desktop\dsappkaar\backend\dtrmodel.sav'
    model2 = pickle.load(open(scaler_path2, "rb"))
    scaler_path3 = r'C:\Users\User\Desktop\dsappkaar\backend\rfmodel.sav'
    model3 = pickle.load(open(scaler_path3, "rb"))
    Y_pred = model1.predict(pd.DataFrame([[oem, model, body, price, years, month]], columns=[
                            'OEM', 'MODEL', 'BODY_TYPE', 'PRICE_IN_LAKHS', 'YEAR', 'MONTH']))
    Y_pred1 = model2.predict(pd.DataFrame([[oem, model, body, price, years, month]], columns=[
                             'OEM', 'MODEL', 'BODY_TYPE', 'PRICE_IN_LAKHS', 'YEAR', 'MONTH']))
    Y_pred2 = model3.predict(pd.DataFrame([[oem, model, body, price, years, month]], columns=[
                             'OEM', 'MODEL', 'BODY_TYPE', 'PRICE_IN_LAKHS', 'YEAR', 'MONTH']))
    return jsonify({'Linear-reg': float(Y_pred), 'DT-reg': float(Y_pred1), 'RF-reg': float(Y_pred2)})


@app.route('/dashboard', methods=['POST', 'GET'])
def forecast():
    reqp = request.get_json()
    oems = reqp['oems']
    models = reqp['models']
    bodys = reqp['bodys']
    months = int(reqp['months'])
    dataset = pd.read_csv(r'C:/Users/User/Downloads/india-car-dataset.csv')
    dataset = dataset.dropna()
    dataset.sort_values(by=["YEAR",'MONTH'])
    new_df=dataset[  (dataset['OEM'] ==  oems)  & (dataset['MODEL'] ==  models) & (dataset['BODY_TYPE'] == bodys)  ]
    new_df= new_df.drop(['OEM','MODEL',	'BODY_TYPE' ,	'PRICE_IN_LAKHS',	'SALE_OUT_OF'],axis=1)
    new_df['YEAR'] = pd.to_datetime(new_df.YEAR.astype(str) + '/' + new_df.MONTH.astype(str) + '/28')
    new_df=new_df.drop(['MONTH'],axis=1)
    indexedDataset = new_df.set_index(['YEAR'])
    rolmean = indexedDataset.rolling(window=12).mean()
    rolstd = indexedDataset.rolling(window=12).std()
    indexedDataset_logScale = np.log(indexedDataset)  
    movingAverage = indexedDataset_logScale.rolling(window=12).mean()
    movingSTD = indexedDataset_logScale.rolling(window=12).std()
    datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage
    datasetLogScaleMinusMovingAverage.dropna(inplace=True)
    exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
    datasetLogScaleMinusExponentialMovingAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
    datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
    datasetLogDiffShifting.dropna(inplace=True)
    model = ARIMA(indexedDataset_logScale, order=(2,1,0))
    results_AR = model.fit()
    model = ARIMA(indexedDataset_logScale, order=(0,1,2))
    results_MA = model.fit()
    model = ARIMA(indexedDataset_logScale, order=(2,1,2))
    results_ARIMA = model.fit()
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    print(predictions_ARIMA_diff.head())
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    print(predictions_ARIMA_diff_cumsum)
    predictions_ARIMA_log = pd.Series(indexedDataset_logScale['SALE'].iloc[0], index=indexedDataset_logScale.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    predictions_ARIMA_log.head()
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    x=plot_predict(results_ARIMA,1,len(indexedDataset_logScale)-1)
    path=os.path.join(r"fronteng\src\assets\py-imgbefore.png")
    x.savefig(path)
    y=plot_predict(results_ARIMA,1,len(indexedDataset_logScale)-1+months)
    path=os.path.join(r"fronteng\src\assets\py-img.png")
    y.savefig(path)
    return jsonify({"result" : "success"})

@app.route('/dashboardtry', methods=['POST', 'GET'])
def forecastdate():    
    for file in request.files.getlist('file'):
        df = pd.read_csv(file)    
    df.drop('Row ID',axis = 1, inplace = True)
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y') 
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y') 
    df.sort_values(by=['Order Date'], inplace=True, ascending=True)
    df.set_index("Order Date", inplace = True)
    new_data = pd.DataFrame(df['Sales'])
    new_data =  pd.DataFrame(new_data['Sales'].resample('D').mean())
    new_data = new_data.interpolate(method='linear')
    train, test, validate = np.split(new_data['Sales'].sample(frac=1), [int(.6*len(new_data['Sales'])),int(.8*len(new_data['Sales']))])
    decomposition = sm.tsa.seasonal_decompose(new_data, model='additive')
    p = d = q = range(0, 2) 
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq_comb = [(i[0], i[1], i[2], 12) for i in list(itertools.product(p, d, q))]
    for parameters in pdq: #for loop for determining the best combination of seasonal parameters for SARIMA
        for seasonal_param in seasonal_pdq_comb:
            try:
                mod = sm.tsa.statespace.SARIMAX(new_data,
                                                order=parameters,
                                                seasonal_param_order=seasonal_param,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False) 
                results = mod.fit()
            except:
                continue
    mod = sm.tsa.statespace.SARIMAX(new_data,
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
    results = mod.fit() 
    pred = results.get_prediction(start=pd.to_datetime('2015-01-03'), dynamic=False)
    pred_val = pred.conf_int()
    date = request.form['name']
    start='2018-12-30'
    count = (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
    forecast = results.forecast(steps=count)
    forecast_df = forecast.to_frame()
    forecast_df.reset_index(level=0, inplace=True)
    forecast_df.columns = ['Prediction Date', 'Predicted Sales'] 
    prediction = pd.DataFrame(forecast_df).to_csv('prediction.csv',index=False)
    pathcsv = r'C:\Users\User\Desktop\dsappkaar\prediction.csv'

    return send_file(pathcsv,as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
