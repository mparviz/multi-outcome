# import packages
import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset

# sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer  # , PowerTransformer
from sklearn.pipeline import make_pipeline


def check_dtype(df):
    my_type = 'float64'

    dtypes = df.dtypes.to_dict()

    for col_name, typ in dtypes.items():
        if typ != my_type:  # <---
            raise ValueError(f"Yikes - 'dataframe['{col_name}'].dtype == {typ}' not {my_type}")


def split_data(x, y, set_id):
    x_train = x.loc[set_id['Train_Val_Test'] == 1, :]
    x_valid = x.loc[set_id['Train_Val_Test'] == 2, :]
    x_test = x.loc[set_id['Train_Val_Test'] == 3, :]
    y_train = y.loc[set_id['Train_Val_Test'] == 1, :]
    y_valid = y.loc[set_id['Train_Val_Test'] == 2, :]
    y_test = y.loc[set_id['Train_Val_Test'] == 3, :]

    x_train.reset_index(drop=True, inplace=True)
    x_valid.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_valid.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def impute_scale(x_train, x_valid, x_test):
    def drop_entirely_missing(x_tr_oh, x_va_oh, x_te_oh):
        empty_tr_cols = list(x_tr_oh.columns[x_tr_oh.count() == 0])
        x_tr_oh_m = x_tr_oh.drop(empty_tr_cols, axis=1)
        x_va_oh_m = x_va_oh.drop(empty_tr_cols, axis=1)
        x_te_oh_m = x_te_oh.drop(empty_tr_cols, axis=1)

        return x_tr_oh_m, x_va_oh_m, x_te_oh_m

    def drop_constant(x_tr_oh, x_va_oh, x_te_oh):
        constant_tr_cols = list(x_tr_oh.columns[x_tr_oh.nunique() == 1])
        x_tr_oh_m = x_tr_oh.drop(constant_tr_cols, axis=1)
        x_va_oh_m = x_va_oh.drop(constant_tr_cols, axis=1)
        x_te_oh_m = x_te_oh.drop(constant_tr_cols, axis=1)

        return x_tr_oh_m, x_va_oh_m, x_te_oh_m

    # missing values: impute
    # impute_params = {'missing_values': np.nan, 'strategy': 'mean', 'add_indicator': False}
    x_train, x_valid, x_test = drop_entirely_missing(x_train, x_valid, x_test)
    x_train, x_valid, x_test = drop_constant(x_train, x_valid, x_test)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=False)  # (impute_params)
    # processor = make_pipeline(QuantileTransformer(), imp_mean)
    processor = make_pipeline(QuantileTransformer(), imp_mean)

    # processor = make_pipeline(StandardScaler(), imp_mean)
    processor.fit(x_train)  # scale and impute
    # small_scale_idx = np.where(processor.steps[0][1].scale_ < 1e-2)
    # processor.steps[0][1].scale_[small_scale_idx] = 1
    is_x_train = processor.transform(x_train)
    is_x_valid = processor.transform(x_valid)
    is_x_test = processor.transform(x_test)

    is_x_train = pd.DataFrame(data=is_x_train, columns=x_train.columns)
    is_x_valid = pd.DataFrame(data=is_x_valid, columns=x_valid.columns)
    is_x_test = pd.DataFrame(data=is_x_test, columns=x_test.columns)

    return is_x_train, is_x_valid, is_x_test


def file2data(root_dir, yaml_fn, do_pre, n_lvl_th, outcome_str, sep_cols=False):
    """ Censor, discard, and load outcomes"""

    # read file names (yaml format)
    stream = open(yaml_fn, 'r')
    file_names = yaml.safe_load(stream)
    # file_names = yaml.load_all(stream, Loader=yaml.FullLoader)
    data_frames = dict()
    for description, file_name in file_names['raw'].items():
        print(description + " : " + str(file_name))
        data_frames[description] = pd.read_csv(root_dir + file_name)

    # NoTreatment_or_Death_Before_Prediction Point
    # 1) find samples with less than 2year follow-up and no outcome.
    data_frames['outcomes']['Last Follow Up'] = pd.to_datetime(data_frames['outcomes']['Last Follow Up'], format='%Y-%m-%d')
    data_frames['outcomes']['Diag Date'] = pd.to_datetime(data_frames['outcomes']['Diag Date'], format='%Y-%m-%d')
    delta_t = pd.Timedelta(2 * 365, unit='d')
    right_censored = (data_frames['outcomes']['Last Follow Up'] < data_frames['outcomes']['Diag Date'] + delta_t) & \
                     (data_frames['outcomes']['Death Outcome (2yr)'] == 0) & \
                     (data_frames['outcomes']['Infection Outcome (2yr)'] == 0) & \
                     (data_frames['outcomes']['Treatment Outcome (2yr)'] == 0)  # and or of Treatment and Infection

    # 2) discard sample with events before-prediction point
    left_censored = ~data_frames['outcomes']['NoTreatment_or_Death_Before_Prediction Point']
    print(left_censored.sum(), right_censored.sum())
    cond = ~ (left_censored | right_censored)
    # 3) set column names?
    # data_frames['feature_matrix'].columns = data_frames[description]

    # x and y
    outcomes_list = [o + ' Outcome (2yr)' for o in outcome_str]
    print(outcomes_list)
    # y = data_frames['outcomes'].loc[cond, ['Death Outcome (2yr)', 'Infection Outcome (2yr)', 'Treatment Outcome (2yr)']]
    y = data_frames['outcomes'].loc[cond, outcomes_list]
    set_id = data_frames['outcomes'].loc[cond, ['Train_Val_Test']]
    x = data_frames['feature_matrix'].loc[cond, :]
    x = x.drop('Unnamed: 0', axis=1)  # drop the first column containing redundant index values

    # x.reset_index(drop=True, inplace=True)
    # y.reset_index(drop=True, inplace=True)
    # check_dtype(x)

    x.replace([np.inf, -np.inf], np.nan, inplace=True)

    # feature names
    x.columns = data_frames['feature_names']['0']
    # print(len(x), (set_id['Train_Val_Test'] == 1).sum(), (set_id['Train_Val_Test'] == 2).sum(), (set_id['Train_Val_Test'] == 3).sum())
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(x, y, set_id)
    if do_pre:
        x_train, x_valid, x_test = impute_scale(x_train, x_valid, x_test)

    if sep_cols:
        no_levels = x_train.nunique()
        cat_cols = no_levels[no_levels <= n_lvl_th].index.to_list()
        cont_cols = no_levels[no_levels > n_lvl_th].index.to_list()
    else:
        cat_cols = []
        cont_cols = list(x_train.columns)

    features = {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test}
    outcomes = {'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test}
    # identify categorical features

    # save files
    '''
    for description, file_name in file_names['processed'].items():
        print(description + " : " + str(file_name))
        eval(description).to_csv(root_dir + file_name, index=False)
    '''
    return features, outcomes, cat_cols, cont_cols


class TimDataset(Dataset):
    """CLL-TIM Dataset."""

    def __init__(self, features, outcomes, transform=None):

        self.transform = transform
        self.features = features
        self.outcomes = outcomes

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        sample = {'features': self.features.iloc[item, :], 'outcomes': self.outcomes.iloc[item, :]}
        if self.transform:
            sample = self.transform(sample)

        return sample

