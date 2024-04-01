import os
import pandas as pd
import numpy as np
import joblib
import dill
from mastml.feature_generators import ElementalFeatureGenerator, OneHotGroupGenerator

def get_barrier(df_test):
    d = 'model_perovskite_ASR/Barrier_model'
    scaler = joblib.load(os.path.join(d, 'StandardScaler.pkl'))
    model = joblib.load(os.path.join(d, 'RandomForestRegressor.pkl'))
    df_features = pd.read_csv(os.path.join(d, 'X_train.csv'))

    features = df_features.columns.tolist()
    X_barrier = df_test[features]
    X_barrier = scaler.transform(X_barrier)
    barriers = model.predict(X_barrier)

    return barriers

def get_preds_ebars_domains(df_test):
    d = 'model_perovskite_ASR'
    scaler = joblib.load(os.path.join(d, 'StandardScaler.pkl'))
    model = joblib.load(os.path.join(d, 'RandomForestRegressor.pkl'))
    df_features = pd.read_csv(os.path.join(d, 'X_train.csv'))
    recal_params = pd.read_csv(os.path.join(d, 'recal_dict.csv'))

    features = df_features.columns.tolist()
    df_test = df_test[features]

    X = scaler.transform(df_test)

    # Make predictions
    preds = model.predict(X)

    # Get ebars and recalibrate them
    errs_list = list()
    a = recal_params['a'][0]
    b = recal_params['b'][0]
    for i, x in X.iterrows():
        preds_list = list()
        for pred in model.model.estimators_:
            preds_list.append(pred.predict(np.array(x).reshape(1, -1))[0])
        errs_list.append(np.std(preds_list))
    ebars = a * np.array(errs_list) + b 

    # Get domains
    with open(os.path.join(d, 'model.dill'), 'rb') as f:
        model_domain = dill.load(f)

    domains = model_domain.predict(X)

    return preds, ebars, domains

def process_data(comp_list):
    X = pd.DataFrame(np.empty((len(comp_list),)))
    y = pd.DataFrame(np.empty((len(comp_list),)))

    df_test = pd.DataFrame({'Material composition': comp_list})

    # Try this both ways depending on mastml version used.
    try:
        X, y = ElementalFeatureGenerator(composition_df=df_test['Material composition'],
                                    feature_types=['composition_avg', 'arithmetic_avg', 'max', 'min','difference'],
                                    remove_constant_columns=False).evaluate(X=X, y=y, savepath=os.getcwd(), make_new_dir=False)
    except:
        X, y = ElementalFeatureGenerator(featurize_df=df_test['Material composition'],
                                         feature_types=['composition_avg', 'arithmetic_avg', 'max', 'min',
                                                        'difference'], remove_constant_columns=False).evaluate(X=X, y=y, savepath=os.getcwd(), make_new_dir=False)

    df_test = pd.concat([df_test, X], axis=1)

    return df_test

def make_predictions(comp_list, elec_list):

    # Process data
    df_test = process_data(comp_list)

    elec_cls_0 = list()
    elec_cls_1 = list()
    elec_cls_2 = list()
    elec_cls_3 = list()
    for elec in elec_list:
        if elec == 'ceria':
            elec_cls_0.append(1)
            elec_cls_1.append(0)
            elec_cls_2.append(0)
            elec_cls_3.append(0)
        elif elec == 'mixed':
            elec_cls_0.append(0)
            elec_cls_1.append(1)
            elec_cls_2.append(0)
            elec_cls_3.append(0)
        elif elec == 'perovskite':
            elec_cls_0.append(0)
            elec_cls_1.append(0)
            elec_cls_2.append(1)
            elec_cls_3.append(0)
        elif elec == 'zirconia':
            elec_cls_0.append(0)
            elec_cls_1.append(0)
            elec_cls_2.append(0)
            elec_cls_3.append(1)
        else:
            raise ValueError('Invalid electrolyte choice detected. Valid choices are "ceria", "mixed", "perovskite", "zirconia"')

    df_test['Electrolyte class_0'] = elec_cls_0  # ceria
    df_test['Electrolyte class_1'] = elec_cls_1  # mixed
    df_test['Electrolyte class_2'] = elec_cls_2  # perovskite
    df_test['Electrolyte class_3'] = elec_cls_3  # zirconia

    barriers = get_barrier(df_test)
    df_test['ML pred ASR barrier (eV)'] = barriers

    # Get the ML predicted values
    preds, ebars, domains = get_preds_ebars_domains(df_test)

    pred_dict = {'Predicted log ASR at 500C (Ohm-cm2)': preds,
                 'Ebar log ASR at 500C (Ohm-cm2)': ebars}

    for d in domains.columns.tolist():
        pred_dict[d] = domains[d]

    del pred_dict['y_pred']
    #del pred_dict['d_pred']
    del pred_dict['y_stdu_pred']
    del pred_dict['y_stdc_pred']

    return pd.DataFrame(pred_dict)
