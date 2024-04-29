import pandas as pd
from scipy.stats import pearsonr
import numpy as np


def load_HCP_csv_file(csv_file_name: str,
                      fields: list = ['Subject', 'NEOFAC_A', 'NEOFAC_C',
                                      'NEOFAC_E', 'NEOFAC_N', 'NEOFAC_O', 'CogTotalComp_AgeAdj']
                      ) -> pd.DataFrame:
    """
    Read unrestricted HCP-YA behavior csv file to obtain subject IDs and target values

    Parameters
    ----------
    csv_file_name : str
        Path of the csv file
    fields : list
        Optional list of fields (str) to load from the csv file
    """
    csv_file = pd.read_csv(csv_file_name, usecols=fields,
                           converters={'Subject': str})
    csv_file = csv_file.dropna()
    return csv_file


def pearson_r_for_scorer(y, y_hat):
    r, p = pearsonr(y, y_hat)
    return np.nan_to_num(r)