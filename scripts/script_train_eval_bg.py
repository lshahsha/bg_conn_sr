"""
script for training models 
@ Ladan Shahshahani, Joern Diedrichsen Jan 30 2023 12:57
"""
import os
import numpy as np
import deepdish as dd
import pathlib as Path
import pandas as pd
import re
import sys
from collections import defaultdict
import nibabel as nb
import Functional_Fusion.dataset as fdata # from functional fusion module
import cortico_cereb_connectivity.globals as gl
import cortico_cereb_connectivity.run_model as rm
import cortico_cereb_connectivity.model as cm
import cortico_cereb_connectivity.scripts.script_train_eval_models as cte
import json


def train_all(dataset = "MDTB",
              subcortex='MNIAsymBg2', 
              crossed = "half",
              type = "CondHalf",
              train_ses = 'ses-s1',
              add_rest = True,
              parcellation = "Icosahedron1002",
              subj_list = "all",
              validate_model = False):
    """_running all the models for bg_

    Args:
        dataset (str, optional): _description_. Defaults to "MDTB".
        subcortex (str, optional): _description_. Defaults to 'MNIAsymBg2'.
        crossed (str, optional): _description_. Defaults to "half".
        type (str, optional): _description_. Defaults to "CondHalf".
        train_ses (str, optional): _description_. Defaults to 'ses-s1'.
        add_rest (bool, optional): _description_. Defaults to True.
        parcellation (str, optional): _description_. Defaults to "Icosahedron1002".
        subj_list (str, optional): _description_. Defaults to "all".
        validate_model (bool, optional): _description_. Defaults to False.
    """
   
    methods = ['L2regression', 'L1regression', 'WTA']
    alpha_list = [[0, 2, 4, 6, 8, 10, 12], [-1, -2, -3, -4, -5], [None]]
    for method, alpha in zip(methods, alpha_list):
        cte.train_models(logalpha_list = alpha, 
                 crossed = crossed, 
                 type = type,
                 train_ses = train_ses,
                 dataset = dataset,
                 add_rest = add_rest,
                 parcellation = parcellation,
                 subj_list = subj_list, 
                 subcortex=subcortex, 
                 method = method,
                 validate_model = validate_model)
        
    return

def avrg_all(train_data = "MDTB",
               train_ses= "ses-s1",
               parcellation = 'Icosahedron1002',
               subcortex='MNIAsymBg2',
               parameters=['scale_','coef_'],
               avrg_mode = 'avrg_sep',
               avg_id = 'avg'):
    """_summary_

    Args:
        train_data (str, optional): _description_. Defaults to "MDTB".
        train_ses (str, optional): _description_. Defaults to "ses-s1".
        parcellation (str, optional): _description_. Defaults to 'Icosahedron1002'.
        subcortex (str, optional): _description_. Defaults to 'MNIAsymBg2'.
        parameters (list, optional): _description_. Defaults to ['scale_','coef_'].
        avrg_mode (str, optional): _description_. Defaults to 'avrg_sep'.
        avg_id (str, optional): _description_. Defaults to 'avg'.
    """
    methods = ['L2regression', 'L1regression', 'WTA']
    alpha_list = [[0, 2, 4, 6, 8, 10, 12], [-1, -2, -3, -4, -5], [None]]
    for method, alpha in zip(methods, alpha_list):
        cte.avrg_model(logalpha_list = alpha,
                        train_data = train_data,
                        train_ses= train_ses,
                        parcellation = parcellation,
                        method=method,
                        subcortex=subcortex,
                        parameters=parameters,
                        avrg_mode = avrg_mode,
                        avg_id = avg_id)

def eval_all(train_data = "MDTB",
             train_ses= "ses-s1",
             parcellation = 'Icosahedron1002',
             subcortex='MNIAsymBg2',
             eval_data = ["MDTB"],
             eval_ses = "ses-s2",
             eval_type = ["CondHalf"],
             eval_id = 'Md_s1',
             model = 'avg', # set to loo for leave one out
             crossed = "half",
             add_rest = True,
             append = False,):
    """_summary_
    """
    methods = ['L2regression', 'L1regression', 'WTA']
    alpha_list = [[0, 2, 4, 6, 8, 10, 12], [-1, -2, -3, -4, -5], [None]]
    for method, alpha in zip(methods, alpha_list):
        cte.eval_models(ext_list = alpha,
                train_dataset = train_data,
                train_ses = train_ses,
                method = method,
                parcellation = parcellation, 
                subcortex=subcortex, 
                eval_dataset = eval_data,
                eval_type = eval_type,
                eval_ses  = eval_ses,
                eval_id = eval_id,
                crossed = crossed,
                add_rest= add_rest,
                model = model,
                append = append)
    return


if __name__ == "__main__":
    pass

  