from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import pandas as pd
import os
import numpy as np
import csv
from scipy.stats import pearsonr
import sys
from src.utils.prediction_utils import load_HCP_csv_file

def get_data_corr(csv_file, split_file: str, file_root: str, atlas: str, connectome_file: str, trait: str, normalization: str, deliminator: str = ','):
    subjects = []
    with open(split_file, newline='') as f:
        reader = csv.reader(f)
        curr_subjects = list(reader)
    for curr_subject in curr_subjects:
        subjects.append(curr_subject)
    
    num_rois = int(atlas[0:3])
    iu1 = np.triu_indices(num_rois, k=1)
    X = []
    y = []
    for subject_id in subjects:
        subject_id = subject_id[0]
        sc = pd.read_csv(os.path.join(file_root, subject_id, atlas, connectome_file), sep=deliminator, header=None)
        sc_proc = sc.values
        np.fill_diagonal(sc_proc, 0)
        sc_proc = np.nan_to_num(sc_proc)
        sc_proc = sc_proc.astype(float)
        
        if normalization == 'log10':
            if connectome_file[-6:-4] == 't2':
                # round to closest int first because otherwise the log10 will transform values in (0,1) to negative values
                sc_proc = np.round(sc_proc)
                sc_proc = sc_proc.astype(float)
            sc_proc[sc_proc > 0] = np.log10(sc_proc[sc_proc > 0])
        
        if subject_id in csv_file['Subject'].values.astype(str):
            trait_score = csv_file[csv_file['Subject'] == subject_id][trait].values[0]
            y.append(trait_score)
            X.append(sc_proc[iu1])

    y = np.array(y, dtype=int)
    X = np.array(X)
    return {'X': X, 'y': y}

def inner_loop(train_data: dict, num_rois:int, normalization: str = 'none'):
    assert normalization in ['none', 'global_min_max', 'log10']
    X = train_data['X']
    y = train_data['y']

    alphas = [0.001, 0.1, 1, 10, 50, 100, 500, 1000, 5000, 10000]
    n_features = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, num_rois]

    kf = KFold(n_splits=5)
    corr_coeffs_folds = []
    X_train_folds = []
    X_test_folds = []
    y_train_folds = []
    y_test_folds = []

    for train_index, test_index in kf.split(X):
        X_train = X[train_index, ...]
        X_train_folds.append(X_train)
        y_train = y[train_index]
        y_train_folds.append(y_train)
        X_test_folds.append(X[test_index, ...])
        y_test_folds.append(y[test_index])

        corr_coeffs = np.zeros(X_train.shape[1])
        for k in range(X_train.shape[1]):
            corr_coeffs[k] = np.abs(np.corrcoef(X_train[:, k], y_train)[0,1])
        corr_coeffs_folds.append(corr_coeffs)


    results = []
    for n_feats in n_features:
        for alpha in alphas:
            rs = []
            for fold in range(5):
                X_train = X_train_folds[fold][:, np.argpartition(np.nan_to_num(corr_coeffs_folds[fold]), -n_feats)[-n_feats:]]
                y_train = y_train_folds[fold]
                X_test = X_test_folds[fold][:, np.argpartition(np.nan_to_num(corr_coeffs_folds[fold]), -n_feats)[-n_feats:]]
                y_test = y_test_folds[fold]

                if normalization == 'global_min_max' or normalization == 'log10':
                    train_max = X_train.max()
                    train_min = X_train.min()
                    X_train = (X_train - train_min) / (train_max - train_min)
                    X_test = (X_test - train_min) / (train_max - train_min)

                reg = Ridge(random_state=1, alpha=alpha)
                reg.fit(X_train, y_train)
                y_test_hat = reg.predict(X_test)
                rs.append(np.nan_to_num(np.corrcoef(y_test_hat, y_test)[0, 1]))
            results.append([n_feats, alpha, np.asarray(rs).mean()])  
    results = np.asarray(results)
    i_max = np.argmax(results[:, -1])
    best_alpha = results[i_max, 1]
    best_n_feats = int(results[i_max, 0])
    return best_n_feats, best_alpha

def Ridge_Corr_CV(test_data: dict,
                    train_data: dict,
                    correlation_coefficients: np.array,
                    num_rois: int,
                    normalization: str = 'none'):
    assert normalization in ['none', 'global_min_max', 'log10']

    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    best_n_feats, best_alpha = inner_loop(train_data, num_rois, normalization)

    X_train = X_train[:, np.argpartition(np.nan_to_num(correlation_coefficients), -best_n_feats)[-best_n_feats:]]

    if normalization == 'global_min_max' or normalization == 'log10':
        train_max = X_train.max()
        train_min = X_train.min()
        X_train = (X_train - train_min) / (train_max - train_min)

    X_test = X_test[:, np.argpartition(np.nan_to_num(correlation_coefficients), -best_n_feats)[-best_n_feats:]]

    if normalization == 'global_min_max' or normalization == 'log10':
        X_test = (X_test - train_min) / (train_max - train_min)
    
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    reg = Ridge(random_state=1, alpha=best_alpha)
    reg.fit(X_train, y_train)
    y_train_hat = reg.predict(X_train)
    y_test_hat = reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_test_hat)
    r = np.corrcoef(y_test, y_test_hat)[0,1]
    mae_train = mean_absolute_error(y_train, y_train_hat)
    r_train = np.corrcoef(y_train, y_train_hat)[0,1]

    return mae_train, r_train, mae, r, y_train_hat, y_train, y_test_hat, y_test


if __name__ == '__main__':
    csv_file_path = '/Users/amelie/Datasets/HCP-YA/unrestricted_ameliecr_2_3_2023_4_4_30.csv'
    csv_file = load_HCP_csv_file(csv_file_path)
    data_root = '/Users/amelie/Datasets/HCP-YA/SCs/19_parcellations'
    save_root = '/Users/amelie/bigfive_dmri/results/cv'
    
    assert len(sys.argv) == 1 or len(sys.argv) == 6
    if len(sys.argv) == 1:
        atlas = '031'
        split_root = '/Users/amelie/Datasets/Splits/unrelated'
        file_name = '031_MIST_10M_mni152_count.csv'
        norm = 'log10'
        target = 'personality'
    else:
        atlas = sys.argv[1]
        split_root = sys.argv[2]
        file_name = sys.argv[3]
        norm = sys.argv[4]
        target = sys.argv[5]
    assert target in ['personality', 'cognition']

    for repeat in range(100):
        print(repeat)
    
        num_rois = int(atlas[0:3])
        weighting = file_name[-6:-4]
        if weighting == 'nt':
            weighting = 'count'
        elif weighting == 't2':
            weighting = 'sift2'
        deliminator = ','
        
        subject_selection = split_root[-1]
        if subject_selection != 'f' and subject_selection != 'm':
            subject_selection = 'all'
        
        if target == 'personality':
            traits = ['NEOFAC_A', 'NEOFAC_C', 'NEOFAC_E', 'NEOFAC_N', 'NEOFAC_O']
        elif target == 'cognition':
            traits = ['CogTotalComp_AgeAdj']

        for trait in traits:
            save_folder = os.path.join(save_root, atlas + '_' + weighting + '_' + norm + '_corr_' + trait + '_' + str(subject_selection))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            train_data = []
            test_data = []
            correlation_coefficients_folds = []
            for i in range(5):
                train_data.append(get_data_corr(csv_file, os.path.join(split_root, str(repeat), 'train_' + str(i) + '.csv'), data_root, atlas, file_name, trait, norm))
                test_data.append(get_data_corr(csv_file, os.path.join(split_root, str(repeat), 'test_' + str(i) + '.csv'), data_root, atlas, file_name, trait, norm))
                correlation_coefficients = np.zeros(train_data[0]['X'].shape[1])
                for k in range(train_data[i]['X'].shape[1]):
                    correlation_coefficients[k] = np.abs(np.corrcoef(train_data[i]['X'][:, k], np.array(train_data[i]['y']))[0,1])
                correlation_coefficients_folds.append(correlation_coefficients)

            save_name = os.path.join(save_folder, str(repeat) + '.csv')
            result_array = np.zeros([23])
            mae_list_test = []
            r_list_test = []
            mae_list_train = []
            r_list_train = []
            ys_train = []
            y_hats_train = []
            ys_test = []
            y_hats_test = []
            for k in range(5):
                mae_train, r_train, mae_test, r_test, y_train_hat, y_train, y_test_hat, y_test = Ridge_Corr_CV(test_data[k], train_data[k], correlation_coefficients_folds[k], num_rois=num_rois, normalization=norm)
                mae_list_train.append(mae_train)
                r_list_train.append(r_train)
                mae_list_test.append(mae_test)
                r_list_test.append(r_test)
                for i in range(y_train.shape[0]):
                    ys_train.append(y_train[i])
                    y_hats_train.append(y_train_hat[i])
                for i in range(y_test.shape[0]):
                    ys_test.append(y_test[i])
                    y_hats_test.append(y_test_hat[i])
            result_list = []
            for mae in mae_list_train:
                result_list.append(mae)
            for r in r_list_train:
                result_list.append(r)
            for mae in mae_list_test:
                result_list.append(mae)
            for r in r_list_test:
                result_list.append(r)
            result_list.append(np.corrcoef(np.array(ys_train), np.array(y_hats_train))[0,1])
            r_test, p_test = pearsonr(np.array(ys_test), np.array(y_hats_test))
            result_list.append(r_test)
            result_list.append(p_test)
            result_array = np.array(result_list)
            np.savetxt(save_name, result_array, delimiter=',')