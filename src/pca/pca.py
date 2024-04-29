from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
import pandas as pd
import os
import numpy as np
import csv
from scipy.stats import pearsonr
import sys
from sklearn.decomposition import PCA
from src.utils.prediction_utils import load_HCP_csv_file, pearson_r_for_scorer

def get_data_pca(csv_file: pd.DataFrame,
                 split_file: str,
                 file_root: str,
                 atlas: str,
                 connectome_file: str,
                 trait: str,
                 normalization: str,
                 deliminator: str = ',',
                 X_max: float = None,
                 X_min: float = None):

    assert normalization in ['none', 'global_min_max', 'log10']
    num_rois = int(atlas[0:3])
    iu1 = np.triu_indices(num_rois, k=1)

    subjects = []

    with open(split_file, newline='') as f:
        reader = csv.reader(f)
        curr_subjects = list(reader)
    for curr_subject in curr_subjects:
        subjects.append(curr_subject)

    X = []
    y = []
    for subject_id in subjects:
        subject_id = subject_id[0]
        sc = pd.read_csv(os.path.join(file_root, subject_id,
                         atlas, connectome_file), sep=deliminator, header=None)
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

        sc_proc = np.nan_to_num(sc_proc[iu1])

        if subject_id in csv_file['Subject'].values.astype(str):
            trait_score = csv_file[csv_file['Subject']
                                   == subject_id][trait].values[0]
            y.append(trait_score)
            X.append(sc_proc)

    X = np.array(X)

    if normalization == 'global_min_max' or normalization == 'log10':
        train_min = X.min() if X_min == None else X_min
        train_max = X.max() if X_max == None else X_max
        X = (X - train_min) / (train_max - train_min)

    y = np.array(y, dtype=int)
    return {'X': X, 'y': y}


def pca_calculation_train(data: np.array, num_comps: int):
    pca = PCA(n_components=num_comps, random_state=1)
    pca.fit(data)
    data = pca.transform(data)
    return data, pca

def pca_transformation_test(data: np.array, pca_object):
    data = pca_object.transform(data)
    return data

def Ridge_PCA(n_comps: int,
              test_data: dict,
              train_data: dict):

    X_test = test_data['X'][:, 0:n_comps]
    X_train = train_data['X'][:, 0:n_comps]
    y_test = test_data['y']
    y_train = train_data['y']

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    parameters = {'alpha': [0.001, 0.1, 1,
                            10, 50, 100, 500, 1000, 5000, 10000]}
    reg = Ridge(random_state=1)
    clf = GridSearchCV(reg, parameters, scoring=make_scorer(
        pearson_r_for_scorer))
    clf.fit(X_train, y_train)
    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)

    mae = mean_absolute_error(y_test, y_test_hat)
    r = np.corrcoef(y_test, y_test_hat)[0, 1]
    mae_train = mean_absolute_error(y_train, y_train_hat)
    r_train = np.corrcoef(y_train, y_train_hat)[0, 1]

    return mae_train, r_train, mae, r, y_train_hat, y_train, y_test_hat, y_test


if __name__ == '__main__':
    csv_file_name = '/Users/amelie/Datasets/HCP-YA/unrestricted_ameliecr_2_3_2023_4_4_30.csv'
    csv_file = load_HCP_csv_file(csv_file_name)
    data_root = '/Users/amelie/Datasets/HCP-YA/SCs/19_parcellations'
    save_root = '/Users/amelie/bigfive_dmri/results'
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

    # repeat 100 times for random splits of the data
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
            traits = ['NEOFAC_A', 'NEOFAC_C',
                      'NEOFAC_E', 'NEOFAC_N', 'NEOFAC_O']
        elif target == 'cognition':
            traits = ['Cognition']

        if num_rois < 200:
            ks = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70,
                  80, 90, 100, 150, 200, num_rois]
        else:
            ks = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70,
                  80, 90, 100, 150, 200]

        for trait in traits:
            save_folder = os.path.join(save_root, atlas + '_' + weighting + '_' + norm + '_pca_' + trait + '_' + str(subject_selection))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            train_data = []
            test_data = []

            # Load training and test data
            for i in range(5):
                train_data.append(get_data_pca(csv_file, os.path.join(split_root, str(repeat), 'train_' + str(
                    i) + '.csv'), data_root, atlas, file_name, trait, norm))
                test_data.append(get_data_pca(csv_file, os.path.join(
                    split_root, str(repeat), 'test_' + str(i) + '.csv'), data_root, atlas, file_name, trait, norm, ',', train_data[0]['X'].max(), train_data[0]['X'].min()))
            # PCA: Calculate and transform
            for i in range(5):
                train_data[i]['X'], pca_object = pca_calculation_train(
                    train_data[i]['X'], np.asarray(ks).max())
                test_data[i]['X'] = pca_transformation_test(
                    test_data[i]['X'], pca_object)

            save_name = os.path.join(save_folder, str(repeat) + '.csv')
            result_array = np.zeros([23, len(ks)])
            for j, region in enumerate(ks):
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
                    mae_train, r_train, mae_test, r_test, y_train_hat, y_train, y_test_hat, y_test = Ridge_PCA(
                        region, test_data[k], train_data[k])
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
                result_list.append(np.corrcoef(
                    np.array(ys_train), np.array(y_hats_train))[0, 1])
                r_test, p_test = pearsonr(
                    np.array(ys_test), np.array(y_hats_test))
                result_list.append(r_test)
                result_list.append(p_test)
                result_array[:, j] = np.array(result_list)
            np.savetxt(save_name, result_array, delimiter=',')