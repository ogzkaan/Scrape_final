from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from .models import Productjson
import joblib

def pricePred(pName):
    np.seterr(divide='ignore', invalid='ignore')
    qSet=Productjson.objects.filter(name=pName).values()
    data=pd.DataFrame(list(qSet))
    
    data.rename(columns={"aggregaterating_reviewcount":"aggregateRating.reviewCount","aggregaterating_ratingcount":"aggregateRating.ratingCount","aggregaterating_ratingvalue":"aggregateRating.ratingValue","aggregaterating_type": "aggregaterating_@type","offers_type":"offers_@type","field_context":"@context","field_type":"@type","id":"@id","brand_type":"brand_@type"}, inplace = True)
    ###
    dropped_data = data.drop(columns=['@id',"description", '@context','@type','name','image','gtin13',
    'brand_@type','url','offers_@type','offers_url','offers_pricecurrency','offers_itemcondition',
    'offers_availability','aggregaterating_@type','field_id','aggregateRating.ratingCount','aggregateRating.reviewCount'])
    print(dropped_data.info())
    ###
    train = dropped_data.iloc[:600,:]
    test = dropped_data.iloc[:256:,]
    ####
    dropped_data['kategori']=dropped_data['kategori'].str.split('-', n=1, expand=True)[1]
    dropped_data['kategori']=dropped_data['kategori'].str[3:]
    dropped_data['kategori'] = dropped_data['kategori'].astype(int)
    dropped_data['kategori'] = np.log(dropped_data['kategori'])
    ###
    dropped_data['offers_price'] = np.log(dropped_data['offers_price'].astype(float))
    ###
    dropped_data['sku'] = dropped_data['sku'].astype(int)
    dropped_data['sku'] = np.log(dropped_data['sku'])
    ###
    enc_nom = (dropped_data.groupby('brandname').size()) / len(dropped_data)
    enc_nom
    dropped_data['brandname_encode'] = dropped_data['brandname'].apply(lambda x : enc_nom[x])
    ###
    from datetime import datetime
    dropped_data['dt_created'] = pd.to_datetime(dropped_data['dt_created'])
    dropped_data['Month'] = [date.month for date in dropped_data.dt_created]
    ###
    dropped_data.drop(['brandname','dt_created'],axis=1,inplace = True)
    ###
    combine=dropped_data
    print(combine.head())
    ###
    dropped_data['aggregateRating.ratingValue']=dropped_data['aggregateRating.ratingValue'].astype(float)
    ###
    test=combine.drop(['offers_price'], axis=1)
    ###
    y_pred_totcat = []
    errcat = []
    cat = CatBoostRegressor(
        n_estimators = 500,
        learning_rate = 0.1,
        loss_function = 'MAE',
        eval_metric = 'RMSE')
    cat.load_model('C:\Project_Scrape\scrape\models\catboost', format='cbm')
    p = cat.predict(test)
    y_pred_totcat.append(p)
    np.mean(errcat,0)
    final = np.exp(np.mean(y_pred_totcat,0))
    ###
    rf = RandomForestRegressor(n_estimators=100,criterion='squared_error',
                           
                           min_samples_leaf=1, 
                           min_samples_split = 5, 
                           random_state=100,)
    loaded_model = joblib.load("C:\Project_Scrape\scrape\models\-rf")
    predictions = loaded_model.predict(test)
    predictions=np.exp(predictions)
    ###
    lgbm = lgb.Booster(model_file='C:\Project_Scrape\scrape\models\lgbr_base.txt')
    lgbm_test=lgbm.predict(test)
    lgbm_test=np.exp(lgbm_test)
    print(lgbm_test)
    ###

    result=pd.DataFrame(final,columns=["CatBoost"])
    result["RFG"]=predictions
    result["lgbm"]=lgbm_test
    pred=result.to_dict(orient='list')
    return pred