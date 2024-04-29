from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, make_scorer
import pandas as pd
import os
import numpy as np
import csv
from scipy.stats import pearsonr
from src.utils.prediction_utils import load_HCP_csv_file, pearson_r_for_scorer
import sys

# Permutation tests for the RCP feature class.

def get_data_permutation(csv_file, split_files: list, file_root: str, atlas: str, connectome_file: str, trait: str, seed: int, deliminator: str = ','):
    subjects = []
    for split_file in split_files:
        with open(split_file, newline='') as f:
            reader = csv.reader(f)
            curr_subjects = list(reader)
        for curr_subject in curr_subjects:
            subjects.append(curr_subject)
    
    X = []
    y = []
    for subject_id in subjects:
        subject_id = subject_id[0]
        sc = pd.read_csv(os.path.join(file_root, subject_id, atlas, connectome_file), sep=deliminator, header=None)
        sc_proc = sc.values
        np.fill_diagonal(sc_proc, 0)
        sc_proc = np.nan_to_num(sc_proc)
        sc_proc = sc_proc.astype(float)
        
        if subject_id in csv_file['Subject'].values.astype(str):
            trait_score = csv_file[csv_file['Subject'] == subject_id][trait].values[0]
            y.append(trait_score)
            X.append(sc_proc)

    y = np.array(y, dtype=int)
    np.random.seed(seed)
    y = np.random.permutation(y)
    return {'X': X, 'y': y}

def Ridge_RCP(region: int,
              test_data: dict,
              train_data: dict,
              normalization: str = 'none'):
    assert normalization in ['none', 'global_min_max', 'log10']

    X_test = []
    X_train = []
    y_test = test_data['y']
    y_train = train_data['y']

    for sc in train_data['X']:
        sc_proc = sc[region, :]
        
        if normalization == 'log10':
            if weighting == 'sift2':
            # round to closest int first because otherwise the log10 will transform values in (0,1) to negative values
                sc_proc = (np.round(sc_proc)).astype(float)
            sc_proc[sc_proc > 0] = np.log10(sc_proc[sc_proc > 0])
        
        X_train.append(sc_proc)

    X_train = np.array(X_train)

    if normalization == 'global_min_max' or normalization == 'log10':
        train_max = X_train.max()
        train_min = X_train.min()
        X_train = (X_train - train_min) / (train_max - train_min)


    for sc in test_data['X']:
        sc_proc = sc[region, :]
        if normalization == 'log10':
            sc_proc[sc_proc > 0] = np.log10(sc_proc[sc_proc > 0])
        
        X_test.append(sc_proc)

    X_test = np.array(X_test)

    if normalization == 'global_min_max' or normalization == 'log10':
        X_test = (X_test - train_min) / (train_max - train_min)

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    parameters = {'alpha': [0.001, 0.1, 1, 10, 50, 100, 500, 1000, 5000, 10000]}
    reg = Ridge(random_state=1)
    clf = GridSearchCV(reg, parameters, scoring=make_scorer(pearson_r_for_scorer))
    clf.fit(X_train, y_train)
    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_test_hat)
    r = np.corrcoef(y_test, y_test_hat)[0,1]
    mae_train = mean_absolute_error(y_train, y_train_hat)
    r_train = np.corrcoef(y_train, y_train_hat)[0,1]

    return mae_train, r_train, mae, r, y_train_hat, y_train, y_test_hat, y_test


if __name__ == '__main__':
    csv_file_path = '/Users/amelie/Datasets/HCP-YA/unrestricted_ameliecr_2_3_2023_4_4_30.csv'
    csv_file = load_HCP_csv_file(csv_file_path)
    data_root = '/Users/amelie/Datasets/HCP-YA/SCs/19_parcellations'
    save_root = '/Users/amelie/bigfive_dmri/results/permutation'

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
    
    np.random.seed(22)
    seeds = np.random.randint(0, 5000, 100)
    for repeat in range(100):
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

        regions = list(range(0, num_rois, 1))

        for trait in traits:
            save_folder = os.path.join(save_root, atlas + '_' + weighting + '_' + norm + '_rcp_' + trait + '_' + str(subject_selection) + '_permutation')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            train_data = []
            test_data = []
            all_data = get_data_permutation(csv_file, [os.path.join(split_root, '0', 'train_0.csv'), os.path.join(split_root, '0', 'test_0.csv')], data_root, atlas, file_name, trait, seeds[repeat])
            kf = KFold(n_splits=5, random_state=seeds[repeat], shuffle=True)

            for train_index, test_index in kf.split(all_data['X']):
                train_data.append({'X': np.asarray(all_data['X'])[train_index], 'y': np.asarray(all_data['y'])[train_index]})
                test_data.append({'X': np.asarray(all_data['X'])[test_index], 'y': np.asarray(all_data['y'])[test_index]})
        
            save_name = os.path.join(save_folder, str(repeat) + '.csv')
            result_array = np.zeros([23, len(regions)])
            for j, region in enumerate(regions):
                print(region)
                mae_list_test = []
                r_list_test = []
                mae_list_train = []
                r_list_train = []
                ys_train = []
                y_hats_train = []
                ys_test = []
                y_hats_test = []
                for k in range(5):
                    mae_train, r_train, mae_test, r_test, y_train_hat, y_train, y_test_hat, y_test = Ridge_RCP(region, test_data[k], train_data[k], normalization=norm)
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
                result_array[:, j] = np.array(result_list)
            np.savetxt(save_name, result_array, delimiter=',')