#!/usr/bin/env python
# coding: utf-8

# # Membership  in CHANCES Low-$z$ mocks with Random Forest #
# 
# We implement a supervised Machine Learning approach to address the cluster membership task. The main goal is to identify true members separating them from line-of-sight interlopers. We leverage a Random Forest approach, training on mock cluster catalogs from the CHileAN Cluster galaxy Evolution Survey (CHANCES) Low-$z$.

# In[23]:


import os
import glob
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy.interpolate import make_interp_spline
import joblib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import gc
import itertools

#matplotlib
from matplotlib import rc, rcParams
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as pe

#astropy
from astropy.io import ascii, fits
from astropy.table import Table , vstack, hstack
from astropy import units as u
from astropy import cosmology
from astropy.cosmology import LambdaCDM, Planck18, Planck15
from astropy.coordinates import SkyCoord, Distance
from astropy.coordinates import ICRS, Galactic, FK4, FK5 
from astropy.coordinates import Angle, Latitude, Longitude  
from astropy.stats import sigma_clip
from astropy.constants import c

#sklearn & ML
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KernelDensity, KDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut, learning_curve, cross_val_predict, GridSearchCV, cross_val_score, LeaveOneGroupOut
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, recall_score, f1_score, precision_score, average_precision_score
from sklearn.base import clone

# from cuml.ensemble import RandomForestClassifier       #cuML


# ## Plotting functions

# In[24]:


# =============================================================================
# CONFIGURATION
# =============================================================================
# Colors
c_tn = '#b0b0b0'         # colors for confusion matrix flags
c_tp = 'green' 
c_fn = 'red' 
c_fp = 'gold'  
c_comp = '#004c6d'   # colors for purity and completeness
c_pur = '#d62728'

# Cosmology
OMegaM = 0.3089
OmegaL = 0.6911 
h = 0.6774
cosmo = LambdaCDM(H0=h*100, Om0=OMegaM, Ode0=OmegaL)

c_km_s = c.to(u.km / u.s).value    # c constant
# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_learning_curve(model, X, y, groups, ylim = None, cv = None, n_jobs = -1, train_sizes = np.linspace(.1, 1.0, 5), scoring = 'f1', model_name = None):
    """
    Plots a learning curve (metric vs. training examples).
        
    Args:
        model (sklearn.base.BaseEstimator): Model to use for the fit.
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        groups (array-like): groups for the Cross-Validation (e.g. Leave-One-Group-Out ). 
        ylim (tuple, optional): Defines minimum and maximum y values (ymin, ymax) in the plot.
        cv (int or cross-validation generator): Determines the Cross-Validation strategy.
        n_jobs (int, optional): Number of jobs to run in parallel.
        train_sizes (array-like): Relative or absolute numbers of training examples.
        scoring (str): Target metric to evaluate (F1- Score by default). 
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """
    c_train = 'red'           # colors
    c_test  = '#1E90FF'

    # Learning curve data
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, groups = groups, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes, scoring = scoring)
    train_mean, train_std = np.mean(train_scores, axis = 1), np.std(train_scores, axis = 1)
    test_mean, test_std = np.mean(test_scores, axis = 1), np.std(test_scores, axis = 1)
    
    fig, ax = plt.subplots(figsize = (10, 6))
    if ylim is not None:
        plt.ylim(*ylim)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha = 0.1, color = c_train)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha = 0.1, color = c_test)
    ax.plot(train_sizes, train_mean, 'o-', color = c_train, label = 'Training score', linewidth = 2, markersize = 6)
    ax.plot(train_sizes, test_mean, 'o-', color = c_test, label = 'Cross-Validation score',linewidth = 2, markersize = 6)

    def k_formatter(x, pos):           # Formatter for x-axis (e,g,, 1000 -> 1k)
        return f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'
        
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.set_xlabel('Training examples', fontsize = 14, labelpad = 20)
    ax.set_ylabel('F1-Score', fontsize = 14)    # can change if you switch validation metric
    ax.legend(loc='best', fontsize = 12, frameon = True, fancybox = True, framealpha = 0.9)
    
    if model_name:
        plt.savefig(f'{model_name}_LC.pdf', dpi = 250, bbox_inches = 'tight')
    plt.show()

############################################################################
############################################################################
############################################################################

def plot_confusion_matrix(cm, classes = ['Interloper', 'Member'], normalize = True, title=' Normalized Confusion Matrix', model_name = None, cmap= 'Blues'):
    """
    Prints and plots a confusion matrix with or without normalization.
        
    Args:
        cm (np.ndarray): Confusion matrix array.
        classes (list): List of class names for the axis labels.
        normalize(bool, optional): Wether to normalize the confusion matrix or not. 
        title (str, optional): Title of the plot.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
        cmap (str or Colormap, optional): Colormap for the heatmap.
    """
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        cm_norm = cm   #  Use raw count 
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    plt.figure(figsize = (7,6))
    
    im = plt.imshow(cm_norm if normalize else cm, interpolation = 'nearest', cmap = cmap)
    
    if normalize:
        im.set_clim(0, 1)      # Ensure range is 0-1 for normalized
    
    cbar = plt.colorbar(im)
    cbar.set_label('Rate' if normalize else 'Count', rotation = 270, labelpad = 20, fontsize = 14)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = 14)
    plt.yticks(tick_marks, classes, fontsize = 14, rotation = 90, va = 'center')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):        
        
        count_val = cm[i, j]                # Absolute value  (Counts)
        if normalize: 
            rate_val = cm_norm[i, j]   # Rate
            text_top = f'{rate_val:.2f}'
            text_color_check = rate_val
        else:
            text_top = f'{count_val}'
            text_color_check = count_val / cm.max()
        text_color = 'white' if text_color_check > 0.5 else 'black'   # Text color: white if background is dark/strong
            
        plt.text(j, i , text_top, ha = 'center', va = 'center', color = text_color, fontsize = 20, fontweight = 'bold')   # rate
        if normalize:
            plt.text(j, i+0.1, f'({count_val})', ha = 'center', va = 'center', color = text_color, fontsize = 13)                # count
        
    plt.tight_layout()
    plt.ylabel('True label', fontsize = 14, labelpad = 20)
    plt.xlabel('Predicted label', fontsize = 14, labelpad = 20)
    
    if model_name:
        plt.savefig(f'{model_name}_CM.pdf', bbox_inches = 'tight', dpi = 250)    
    plt.show()

############################################################################
############################################################################
############################################################################

def plot_feature_importances(model, X, y, model_name=None):
    """
    Plots the Random Forest Feature Importance (Mean Decrease Impurity).
        
    Args:
        model (sklearn.base.BaseEstimator): Model to use for the fit.
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """
    
    # Ensure it is fitted
    if not hasattr(model, "feature_importances_"):
        print('Model not fitted. Fitting now...')
        model.fit(X, y)
    
    feature_importances = model.feature_importances_
    importance = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by = 'Importance', ascending = True)

    names = {'V_norm': r'$V_{pec}/\sigma_{200}$',       # labels for ticks
                     'R_norm': r'$r_{proj}/r_{200}$',
                     'log_local_density': r'log $\Sigma_{10}$', 
                     'Mvir': r'$M_{200}$',
                     'r_mag': r'$m_{r}$'}

    importance['Feature'] = importance['Feature'].replace(names)
    
    fig, ax = plt.subplots(figsize = (10, 6))

    # Normalize colormap
    norm = mcolors.Normalize(vmin = importance['Importance'].min(), vmax = importance['Importance'].max())
    cmap = plt.get_cmap('Blues')
    bar_colors = cmap(norm(importance['Importance'].values))

    bars = ax.barh(importance['Feature'], importance['Importance'], color = bar_colors, edgecolor = 'none', height = 0.7)

    max_val = importance['Importance'].max()
    ax.set_xlim(0, max_val * 1.10)

     # Add label to each bar
    for rect, value in zip(bars, importance['Importance']):
        width = rect.get_width()
        font_weight = 'bold' if value == max_val else 'normal'
            
        ax.text(width + (max_val * 0.01), rect.get_y() + rect.get_height()/2, f'{value:.3f}', va = 'center', ha = 'left', 
                    fontsize = 11, fontweight = font_weight, color = 'black')
    
    ax.set_xlabel('Feature Importance', fontsize = 14)
    ax.tick_params(axis = 'y', labelsize = 13, length = 0)

    plt.tight_layout()
    if model_name:
        plt.savefig(f'{model_name}_FeatureImportance.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()

############################################################################
############################################################################
############################################################################

def plot_phase_space(results,model_name=None):
    """
    Plots a Phase Space Diagram color-coded witht the confusion matrix flags (tp, tn, fp, fn).
    
    Args:
        results (pd.DataFrame): Dataframe with summarized results and predictions of the model.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """
    fig, ax = plt.subplots(figsize = (10, 8))

    mask_TP = results['class'] == 'TP'
    mask_FP = results['class'] == 'FP'
    mask_TN = results['class'] == 'TN'
    mask_FN = results['class'] == 'FN'
    
    ax.scatter(results.loc[mask_TN, 'R_norm'], results.loc[mask_TN, 'V_norm'], s = 2, alpha = 0.3, color = c_tn, label='TN', zorder=1)
    ax.scatter(results.loc[mask_TP,  'R_norm'], results.loc[mask_TP, 'V_norm'],  s = 5, alpha = 0.8, color = c_tp, label='TP', zorder=2)
    ax.scatter(results.loc[mask_FN, 'R_norm'], results.loc[mask_FN, 'V_norm'], s = 4, alpha = 0.5, color = c_fn, label='FN', zorder=3)
    ax.scatter(results.loc[mask_FP,  'R_norm'], results.loc[mask_FP, 'V_norm'],  s = 4, alpha = 0.5, color = c_fp, label='FP', zorder=4)    

    kde_args = {'ax': ax, 'levels': 5, 'thresh': 0.3, 'linewidths': 2.5}       # density contours
    sns.kdeplot(x = results.loc[mask_TP, 'R_norm'], y = results.loc[mask_TP, 'V_norm'],  color = 'darkgreen', zorder = 7, **kde_args)
    sns.kdeplot(x = results.loc[mask_FN,'R_norm'], y = results.loc[mask_FN, 'V_norm'], color = 'darkred', zorder = 5, **kde_args)
    sns.kdeplot(x = results.loc[mask_FP, 'R_norm'], y = results.loc[mask_FP, 'V_norm'], color = 'darkorange', zorder = 6, **kde_args)
    
    ax.set_xlabel(r'$r_{proj} / r_{200}$', fontsize=22)
    ax.set_ylabel(r'$\Delta v / \sigma_{200}$', fontsize=22)
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-4, 4)

    legend_elements = [   # custom legend
        Line2D([0], [0], color = c_tn, marker = 'o', linestyle = 'None', markersize = 5, label = 'TN', alpha = 0.5),
        Line2D([0], [0], color = c_tp, marker = 'o', markersize = 5, label = 'TP', linestyle = '-'),
        Line2D([0], [0], color = c_fn, marker = 'o', markersize = 5, label = 'FN', linestyle = '-'),
        Line2D([0], [0], color = c_fp, marker = 'o', markersize = 5, label = 'FP', linestyle = '-')]

    ax.legend(handles = legend_elements, fontsize = 13, loc = 1, framealpha = 0.9)
    plt.tight_layout()
    if model_name:
        plt.savefig(f'{model_name}_PPS.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()

############################################################################
############################################################################
############################################################################

def plot_purity_and_completeness(y, y_pred, results, model_name=None, n_bins=25):
    """
    Plots purity and completeness as a function of the cluster-centric distance in bins. 
    Includes boxplots for each bin to visualize the variance between clusters.

    Args:
        y (array-like): True labels.
        y_pred (array-like): Predicted labels.
        results (pd.DataFrame): Dataframe with the summarized results and predictions for the model.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
        n_bins (int): Number of radial bins to use. Defaults to 25.
    """
    # bin configuration
    r_min, r_max = 0, 5
    bins = np.linspace(r_min, r_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    width_box = bin_width * 0.25

    global_purity = precision_score(y, y_pred)      # global metrics (over full confusion matrix)
    global_comp = recall_score(y, y_pred)
    
    tp_global, _ = np.histogram(results.loc[results['class'] == 'TP', 'R_norm'], bins = bins)   # global metrics per bin
    fp_global, _ = np.histogram(results.loc[results['class'] == 'FP', 'R_norm'], bins = bins)
    fn_global, _ = np.histogram(results.loc[results['class'] == 'FN', 'R_norm'], bins = bins)

    purity_bin = np.divide(tp_global, tp_global + fp_global, out = np.full_like(tp_global, np.nan, dtype = float), 
                           where = (tp_global+fp_global) != 0)   # safe division to handle empty bins
    completeness_bin = np.divide(tp_global, tp_global + fn_global, out = np.full_like(tp_global, np.nan, dtype = float), 
                                 where = (tp_global+fn_global) != 0)

##################################################################################
    print(f"{'Bin Center '} | {'TP':<6} | {'FN':<6} | {'Total':<10} | {'Completeness':<12}") #stats table
    print('-' * 55)
    for i in range(len(bin_centers)):
        total_true = tp_global[i] + fn_global[i]
        comp_val = completeness_bin[i]
        print(f'{bin_centers[i]:<11} | {tp_global[i]:<6} | {fn_global[i]:<6} | {total_true:<10} | {comp_val:.4f}')
##################################################################################
    
    # Metrics per cluster (boxplots)
    clusters = results['mock_id'].unique()
    matrix_pur = np.zeros((len(clusters), n_bins))   # initialize list of lists
    matrix_com = np.zeros((len(clusters), n_bins))
    matrix_pur[:] = np.nan
    matrix_com[:] = np.nan

    for i, cl_id in enumerate(clusters):
        sub = results[results['mock_id'] == cl_id]
        tp, _ = np.histogram(sub.loc[sub['class'] == 'TP', 'R_norm'], bins = bins)
        fp, _ = np.histogram(sub.loc[sub['class'] == 'FP', 'R_norm'], bins = bins)
        fn, _ = np.histogram(sub.loc[sub['class'] == 'FN', 'R_norm'], bins = bins)
        denom_p = tp + fp
        denom_c = tp + fn
        matrix_pur[i, :] = np.divide(tp, denom_p, out = np.full_like(tp, np.nan, dtype = float), where = denom_p != 0)
        matrix_com[i, :] = np.divide(tp, denom_c, out = np.full_like(tp, np.nan, dtype = float), where = denom_c != 0)

    # clean NaNs for boxplots
    data_pur = []
    data_com = []
    for i in range(n_bins):
        col_p = matrix_pur[:, i]
        col_c = matrix_com[:, i]
        data_pur.append(col_p[~np.isnan(col_p)])
        data_com.append(col_c[~np.isnan(col_c)])

    # PLOT
    fig, ax = plt.subplots(figsize=(12, 6))

    # Benchmarks from Farid et al. (2023) Table 2 (RF +mratio). Best deterministic model values
    f23_orb_rec, f23_orb_pur = 0.89, 0.70
    f23_inf_rec, f23_inf_pur = 0.45, 0.64

    ax.axhline(f23_orb_rec, color = c_comp, linestyle = ':', alpha = 1.0, linewidth = 1.6, zorder = 0)      # Orbiting
    ax.axhline(f23_orb_pur, color = c_pur, linestyle = ':', alpha = 1.0, linewidth = 1.6,  zorder = 0)
    ax.axhline(f23_inf_rec, color = c_comp, linestyle = '--', alpha = 0.8, linewidth = 1.2, zorder = 0)    # infalling
    ax.axhline(f23_inf_pur, color = c_pur, linestyle = '--', alpha = 0.8, linewidth = 1.2, zorder = 0)
    
    # Global metric lines 
    ax.plot(bin_centers, completeness_bin, color = c_comp, marker = 'o', markersize = 5, linestyle = '-', linewidth = 2,   # global completeness
                 label = f'Completeness (Global: {global_comp:.2f})', zorder = 10)

    ax.plot(bin_centers, purity_bin, color = c_pur, marker = 's', markersize = 5, linestyle = '--', linewidth = 2,                  # global purity
                 label = f'Purity (Global: {global_purity:.2f})', zorder = 10)

    # boxplots configuration
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color = color, linewidth = 1)
        plt.setp(bp['whiskers'], color = color, linewidth = 1)
        plt.setp(bp['caps'], color = color, linewidth = 1)
        plt.setp(bp['medians'], color = color, linewidth = 1.5)

    # Boxplots, no outliers
    bp_com = ax.boxplot(data_com, positions = bin_centers, widths = width_box, patch_artist = True, showfliers = False, zorder = 2)     # completeness
    bp_pur = ax.boxplot(data_pur, positions = bin_centers, widths = width_box, patch_artist = True, showfliers = False, zorder = 2)        # purity
    
    set_box_color(bp_com, c_comp)    # boxplots colors
    set_box_color(bp_pur, c_pur)
    for patch in bp_com['boxes']:
        patch.set_facecolor(c_comp)
        patch.set_alpha(0.4)
    for patch in bp_pur['boxes']:
        patch.set_facecolor(c_pur)
        patch.set_alpha(0.4)
        
    ax.text(0.02, f23_orb_rec-0.02, 'F+23 Orb', color = c_comp, va = 'center', fontsize = 8)    # labels for benchmarks
    ax.text(0.02, f23_orb_pur-0.02, 'F+23 Orb', color = c_pur, va = 'center', fontsize = 8)
    ax.text(0.02, f23_inf_rec-0.02, 'F+23 Inf', color = c_comp, va = 'center', fontsize = 8)
    ax.text(0.02, f23_inf_pur-0.02, 'F+23 Inf', color = c_pur, va = 'center', fontsize = 8)
    
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r'$r_{proj} / r_{200}$', fontsize = 17)
    ax.set_ylabel('Rate', fontsize = 15)
    ax.set_xticks( [0, 1, 2, 3, 4, 5])
    ax.set_xticklabels( [0, 1, 2, 3, 4, 5], fontsize = 12)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    handles = [plt.Line2D([0], [0], color = c_comp, marker = 'o', label = f'This Work - Comp. (Global: {global_comp:.2f})'),           # custom legend
               plt.Line2D([0], [0], color = c_pur, marker = 's', linestyle = '--', label = f'This Work - Pur. (Global: {global_purity:.2f})'),
               Patch(facecolor = 'gray', edgecolor = 'gray', alpha = 0.4, label = 'Cluster Variability (IQR)'),
               plt.Line2D([0], [0], color ='gray', linestyle = ':', label = f'Farid+23 (Orbiting, P: {f23_orb_pur:.2f}   C: {f23_orb_rec:.2f})'),
               plt.Line2D([0], [0], color ='gray', linestyle = '--', alpha = 0.5, label = f'Farid+23 (Infalling, P: {f23_inf_pur:.2f}   C: {f23_inf_rec:.2f})')]

    ax.legend(handles = handles, fontsize = 12, loc = 'lower left', framealpha = 0.75, fancybox = True)
    plt.tight_layout()
    
    if model_name:
        plt.savefig(f'{model_name}_purandcomp.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()

############################################################################
############################################################################
############################################################################

def plot_roc_curve(y_true, y_prob, thresh, model_name=None):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and highlights the selected threshold.
        
    Args:
        y_true (array-like): True labels.
        y_prob (array-like): Probability estimates of the positive class.
        thresh (float): Decision threshold chosen for predictions.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """
    
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    # Find best F1-Score based on Precision-Recall 
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
    fscore = (2 * precision * recall) / (precision + recall)
    ix_f1 = np.argmax(fscore)
    best_thresh_f1 = thresholds_pr[ix_f1]

    # Find indices for plotting points
    idx_f1 = np.argmin(np.abs(thresholds_roc - best_thresh_f1))  # max f1 index
    idx_chosen = np.argmin(np.abs(thresholds_roc - thresh))         # chosen threshold index

    fig, ax = plt.subplots(figsize = (8, 7))

    ax.plot(fpr, tpr, color = '#004c6d', lw = 2, label = f'AUC = {auc_score:.2f}', zorder = 1)      # ROC
    ax.fill_between(fpr, tpr, alpha = 0.1, color = '#1E90FF', zorder = 2)

    ax.scatter(fpr[idx_chosen], tpr[idx_chosen], s = 100, facecolors = 'white', edgecolors = '#d62728', lw = 2, zorder = 4)
    ax.scatter(fpr[idx_chosen], tpr[idx_chosen], s = 20, color = '#d62728', label = f'Threshold used = {thresh}', zorder = 5)
    ax.scatter(fpr[idx_f1], tpr[idx_f1], marker = 'x', color = 'black', s = 100, label = f'Max F1 (Th={best_thresh_f1:.2f})', zorder = 6)
    ax.plot([0, 1], [0, 1], linestyle = '--', lw = 1.5, color = 'gray', label = 'Random Guessing')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate', fontsize = 14)
    ax.set_ylabel('True Positive Rate', fontsize = 14)
    ax.legend(loc = 'lower right', frameon = True, fontsize = 11)
    plt.tight_layout()

    if model_name:
        plt.savefig(f'{model_name}_ROC.pdf', dpi = 300)
    plt.show()

############################################################################
############################################################################
############################################################################

def plot_3d_cm_per_cluster(dff, cluster_catalogue, results, model_name):
    """
    Generates 3D plots color-coded with the confusion matrix flags (tp, tn, fp, fn) for each cluster.
    
    Note: This function creates a directory and saves multiple PDF files.
    
    Args:
        dff (pd.DataFrame): Dataframe with galaxy data (x, y, z, mock_id, etc.).
        cluster_catalogue (pd.DataFrame): Cluster catalog with cluster data (RA, DEC, R200, etc.).
        results (pd.DataFrame): Dataframe with the summarized results and predictions for the model.
        model_name (str): Name of the model, used for directory and file names.
    """
        
    df_working = dff.copy()
    df_working['class_pred'] = results['class'].values

    output_dir = f'3Dplot_ConfusionMatrix_{model_name}'  # directory based on model's name
    os.makedirs(output_dir, exist_ok = True)

    total_clusters = len(cluster_catalogue)

    # Iterate over each mock
    for i, row in cluster_catalogue.iterrows():
        cluster_id = int(row['Cluster'])
        
        # Filter galaxies for this mock
        df_mock = df_working[df_working['mock_id'] == cluster_id].copy()    
        x_cent, y_cent, z_cent = row['X_cent'], row['Y_cent'], row['Z_cent']
        R200 = row['R_200_mpc']

        df_mock['x0'] = df_mock['x'] - x_cent   # center coords
        df_mock['y0'] = df_mock['y'] - y_cent
        df_mock['z0'] = df_mock['z'] - z_cent

        lim = 5 * R200 * 1.3   # limit
        mask_in_box = ((df_mock['x0'].abs() <= lim) & (df_mock['y0'].abs() <= lim) & (df_mock['z0'].abs() <= lim))
        df_plot = df_mock[mask_in_box]

        tp = df_plot[df_plot['class_pred'] == 'TP']  # split by classification
        fp = df_plot[df_plot['class_pred'] == 'FP']
        fn = df_plot[df_plot['class_pred'] == 'FN'] 
        tn = df_plot[df_plot['class_pred'] == 'TN'] 

        # Downsampling TN for visual clarity
        if len(tn) > 5000:
            tn = tn.sample(n = 5000, random_state = 42)

        # PLOT 
        fig = plt.figure(figsize = (10, 8))
        ax = fig.add_subplot(111, projection = '3d')
        
        ax.scatter(tn['x0'], tn['y0'], tn['z0'], c = c_tn, s = 5, alpha = 0.3, label = f'TN: {len(tn)}', zorder = 1)
        ax.scatter(tp['x0'], tp['y0'], tp['z0'], c = c_tp, s = 10, alpha = 0.6, label = f'TP: {len(tp)}', zorder = 2)
        ax.scatter(fn['x0'], fn['y0'], fn['z0'], c = c_fn, s = 15, marker = '^', alpha = 0.8, label = f'FN: {len(fn)}', zorder = 11)
        ax.scatter(fp['x0'], fp['y0'], fp['z0'], c = c_fp, s = 15, marker = 'x', alpha = 0.9, label = f'FP: {len(fp)}', zorder = 12)

        # Wireframe  sphere (5R_200)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r_sphere = 5 * R200
        x_sph = r_sphere * np.cos(u) * np.sin(v)
        y_sph = r_sphere * np.sin(u) * np.sin(v)
        z_sph = r_sphere * np.cos(v)
        ax.plot_wireframe(x_sph, y_sph, z_sph, color = 'k', alpha = 0.1, linewidth = 0.5)

        ax.xaxis.pane.fill = False     # style
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        grid_style = {'color': 'gray', 'linestyle': ':', 'linewidth': 0.5, 'alpha': 0.3}
        ax.xaxis._axinfo["grid"].update(grid_style)   
        ax.yaxis._axinfo["grid"].update(grid_style)
        ax.zaxis._axinfo["grid"].update(grid_style)

        ax.set_xlabel('x [Mpc]')
        ax.set_ylabel('y [Mpc]')
        ax.set_zlabel('z [Mpc]')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_title(f'Cluster {cluster_id}')
        ax.legend(loc = 'upper right', fontsize = 'small')

        ax.set_box_aspect([1,1,1])
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'Cluster_{cluster_id}_CM.pdf')
        plt.savefig(filename, dpi = 300)
        plt.close(fig) 
    print('Done')

############################################################################
############################################################################
############################################################################

def plot_3d_cm_stacked(dff, cluster_catalogue, results, model_name = None):
    """
    Plots the stacked 3D clusters color-coded with the confusion matrix flags (tp, tn, fp, fn).

    Args:
        dff (pd.DataFrame): Dataframe with galaxy data (x, y, z, mock_id, etc.).
        cluster_catalogue (pd.DataFrame): Cluster catalog with cluster data (RA, DEC, R200, etc.).
        results (pd.DataFrame): Dataframe with summarized results and predictions for the model.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """

    df_working = dff.copy()
    df_working['class_pred'] = results['class'].values
    stack_tp, stack_fp, stack_fn, stack_tn = [], [], [], []
    total_clusters = len(cluster_catalogue)
    
    for i, row in cluster_catalogue.iterrows():    
        cluster_id = int(row['Cluster'])
    
        df_mock = df_working[df_working['mock_id'] == cluster_id].copy()
        
        x_c, y_c, z_c = row['X_cent'], row['Y_cent'], row['Z_cent']
        r200 = row['R_200_mpc']

        df_mock['x_norm'] = (df_mock['x'] - x_c) / r200
        df_mock['y_norm'] = (df_mock['y'] - y_c) / r200
        df_mock['z_norm'] = (df_mock['z'] - z_c) / r200
        
        lim = 6.0
        mask_box = (df_mock['x_norm'].abs() < lim) & (df_mock['y_norm'].abs() < lim) & (df_mock['z_norm'].abs() < lim)
        df_plot = df_mock[mask_box]
        
        stack_tp.append(df_plot[df_plot['class_pred'] == 'TP'][['x_norm','y_norm','z_norm']])
        stack_fp.append(df_plot[df_plot['class_pred'] == 'FP'][['x_norm','y_norm','z_norm']])
        stack_fn.append(df_plot[df_plot['class_pred'] == 'FN'][['x_norm','y_norm','z_norm']])
        
        tn_subset = df_plot[df_plot['class_pred'] == 'TN']
        tn_subset = tn_subset.sample(frac=0.5, random_state=42)  # downsampling TN
        stack_tn.append(tn_subset[['x_norm','y_norm','z_norm']])

    df_tp = pd.concat(stack_tp) if stack_tp else pd.DataFrame()   # concatenate all stacks
    df_fp = pd.concat(stack_fp) if stack_fp else pd.DataFrame()
    df_fn = pd.concat(stack_fn) if stack_fn else pd.DataFrame()
    df_tn = pd.concat(stack_tn) if stack_tn else pd.DataFrame()

    # PLOT
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df_tn['x_norm'], df_tn['y_norm'], df_tn['z_norm'], c = c_tn, s = 2, alpha = 0.4, label = 'TN', zorder = 1, rasterized = False)
    ax.scatter(df_tp['x_norm'], df_tp['y_norm'], df_tp['z_norm'], c = c_tp, s = 2, alpha = 0.3, label = 'TP', zorder = 2, rasterized = False)
    ax.scatter(df_fn['x_norm'], df_fn['y_norm'], df_fn['z_norm'], c = c_fn, s = 5, marker = 'x', alpha = 0.8, label = 'FN', zorder = 10, rasterized = False)
    ax.scatter(df_fp['x_norm'], df_fp['y_norm'], df_fp['z_norm'], c = c_fp, s = 5, marker = '^', alpha = 0.8, label = 'FP', zorder = 11, rasterized = False)

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]

    # Sphere R_200
    x = 1.0 * np.cos(u) * np.sin(v)
    y = 1.0 * np.sin(u) * np.sin(v)
    z = 1.0 * np.cos(v)
    ax.plot_wireframe(x, y, z, color = 'black', alpha = 0.4, linewidth = 1.0, label = '$R_{200}$') 

    # Sphere 5R_200
    x5 = 5.0 * np.cos(u) * np.sin(v)
    y5 = 5.0 * np.sin(u) * np.sin(v)
    z5 = 5.0 * np.cos(v)
    ax.plot_wireframe(x5, y5, z5, color = 'black', alpha = 0.3, linewidth = 1.0, linestyle = '--', label = '$5R_{200}$')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    grid_style = {'color': 'gray', 'linestyle': ':', 'linewidth': 0.2, 'alpha': 0.3}
    ax.xaxis._axinfo['grid'].update(grid_style)
    ax.yaxis._axinfo['grid'].update(grid_style)
    ax.zaxis._axinfo['grid'].update(grid_style)

    ax.set_xlabel(r'$x / r_{200}$', fontsize=17, labelpad=10)
    ax.set_ylabel(r'$y / r_{200}$', fontsize=17, labelpad=10)
    ax.set_zlabel(r'$z / r_{200}$', fontsize=17, labelpad=10)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1,1,1])

    leg = plt.legend(loc = 'upper right', fancybox = True, fontsize = 13)
    for lh in leg.legend_handles: 
        if hasattr(lh, '_sizes'):    # make legend markers more visible
            lh._sizes = [30]
            lh.set_alpha(1)

    plt.tight_layout()

    if model_name:
        plt.savefig(f'AllClusters_3D_CM_{model_name}.pdf', dpi = 400, bbox_inches = 'tight')
    plt.show()

############################################################################
############################################################################
############################################################################

def plot_3d_cm_obs_coords(dff, cluster_catalogue, results, model_name):
    """
    Generates observational 3D space (RA, DEC, z) plots color-coded with the confusion matrix flags 
    (tp, tn, fp, fn) for each cluster.

    Note: This function creates a directory and saves multiple PDF files.

    Args:
        dff (pd.DataFrame): Dataframe with galaxy data (x, y, z, mock_id, etc.).
        cluster_catalogue (pd.DataFrame): Cluster catalog with cluster data (RA, DEC, R200, etc.).
        results (pd.DataFrame): Dataframe with summarized results and predictions for the model.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """
    
    output_dir = f'3Dplot_ObsCoords_{model_name}'
    os.makedirs(output_dir, exist_ok = True)

    df = dff.copy()
    df['class_pred'] = results['class'].values

    for i, row in cluster_catalogue.iterrows():
        cluster_id = int(row['Cluster'])
        RA_cl, DEC_cl, z_cl = row['RA'], row['Dec'], row['redshift']
        R200_mpc = row['R_200_mpc']
        df_mock = df[df['mock_id'] == cluster_id].copy()
        
        df_mock['dra'] = (df_mock['RA'] - RA_cl)
        df_mock['ddec'] = df_mock['DEC'] - DEC_cl
        df_mock['dz'] = df_mock['redshift_S_1'] - z_cl
        
        da = cosmo.angular_diameter_distance(z_cl).value   # visual limit based on R200
        r5_deg = np.rad2deg((5 * R200_mpc) / da)
        
        hz = cosmo.H(z_cl).value
        r5_dz = (hz * (5 * R200_mpc)) / c_km_s     # 5*R200: dz = (H(z) * dist) / c

        lim_ang = r5_deg * 3.0   # limit for angular coords, zoom out box in order to see FOG
        lim_z = r5_dz * 4.0         # limit for redshift

        mask_in_box = ((df_mock['dra'].abs() <= lim_ang) & (df_mock['ddec'].abs() <= lim_ang) & (df_mock['dz'].abs() <= lim_z))
        df_plot = df_mock[mask_in_box]
    
        tp = df_plot[df_plot['class_pred'] == 'TP']
        fp = df_plot[df_plot['class_pred'] == 'FP']
        fn = df_plot[df_plot['class_pred'] == 'FN'] 
        tn = df_plot[df_plot['class_pred'] == 'TN'] 

        if len(tn) > 4000:
            tn = tn.sample(n = 4000, random_state = 42)    #downsampling of TN if there are too many

        # Plot
        fig = plt.figure(figsize = (9, 9))
        ax = fig.add_subplot(111, projection = '3d')
        
        ax.scatter(tn['dra'], tn['ddec'], tn['dz'], c = c_tn, s = 5, alpha = 0.3, label = f'TN: {len(tn)}', zorder=1)
        ax.scatter(tp['dra'], tp['ddec'], tp['dz'], c = c_tp, s = 8, alpha = 0.6, label = f'TP: {len(tp)}', zorder=2)
        ax.scatter(fn['dra'], fn['ddec'], fn['dz'], c =  c_fn, s = 12, marker = '^', alpha = 0.8, label = f'FN: {len(fn)}', zorder = 11)
        ax.scatter(fp['dra'], fp['ddec'], fp['dz'], c = c_fp,  s = 12, marker = 'x', alpha = 0.9, label = f'FP: {len(fp)}', zorder = 12)
        
        # 5R200 Sphere
        phi, theta = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x_sph = r5_deg * np.cos(phi) * np.sin(theta)
        y_sph = r5_deg * np.sin(phi) * np.sin(theta)
        z_sph = r5_dz * np.cos(theta)
        ax.plot_wireframe(x_sph, y_sph, z_sph, color="black", alpha=0.2, linewidth=0.5)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_facecolor('none')         
        grid_style = {'color': 'gray', 'linestyle': ':', 'linewidth': 0.5, 'alpha': 0.3}
        ax.xaxis._axinfo["grid"].update(grid_style)   
        ax.yaxis._axinfo["grid"].update(grid_style)
        ax.zaxis._axinfo["grid"].update(grid_style)

        ax.set_xlabel(r'$\Delta \alpha$ [deg]', fontsize = 14, labelpad = 10)
        ax.set_ylabel(r'$\Delta \delta$ [deg]', fontsize = 14, labelpad = 10)
        ax.set_zlabel(r'$\Delta z$', fontsize = 14, labelpad = 15)
        ax.tick_params(axis = 'z', pad = 8)
        
        ax.set_xlim(-lim_ang, lim_ang)
        ax.set_ylim(-lim_ang, lim_ang)
        ax.set_zlim(-lim_z, lim_z)
        ax.set_title(f'Mock {cluster_id} Observational Space', fontsize = 16)

        leg = ax.legend(loc = 'upper right', fontsize = 12, frameon = True, facecolor = 'white', framealpha = 0.8)
        for lh in leg.legend_handles:
            lh.set_sizes([50])
            lh.set_alpha(1)

        # perspective
        ax.set_box_aspect([1, 1, 1.333])       # elongated on z since the limits are different, preserves the sphere
        ax.view_init(elev = 15, azim = -60)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'Cluster_{cluster_id}_ObsCM.pdf')
        plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
        plt.close(fig)
    print('Done')


# ## Dataset

# CHANCES catalog (<span style="color: blue"> BLUE: used here </span> , <span style="color: red"> RED: not CHANCES</span>)
# 
# - <span style="color: blue">00: Abell 85</span>
# - <span style="color: blue">01: Abell 119</span>
# - <span style="color: blue">02: Abell 133</span>
# - <span style="color: blue">03: Abell 147</span>
# - <span style="color: blue">04: Abell 151</span>
# - <span style="color: blue">05: Abell 168</span>
# - 06: Abell 194
# - <span style="color: blue">07: Abell 496</span>
# - 08: Abell 500
# - <span style="color: blue">09: Abell 548</span>
# - <span style="color: blue">10: Abell 754</span>
# - <span style="color: blue">11: Abell 780</span>
# - <span style="color: blue">12: Abell 957</span>
# - <span style="color: blue">13: Abell 970</span>
# - <span style="color: blue">14: Abell 1069</span>
# - 15: Abell 1520
# - <span style="color: blue">16: Abell 1631</span>
# - <span style="color: blue">17: Abell 1644</span>
# - <span style="color: blue">18: Abell 2399</span>
# - <span style="color: blue">19: Abell 2415</span>
# - <span style="color: blue">20: Abell 2457</span>
# - <span style="color: blue">21: Abell 2717</span>
# - <span style="color: blue">22: Abell 2734</span>
# - <span style="color: blue">23: Abell 2870</span>
# - <span style="color: blue">24: Abell 2877</span>
# - <span style="color: blue">25: Abell 3223</span>
# - 26: Abell 3266
# - <span style="color: blue">27: Abell 3301</span>
# - <span style="color: blue">28: Abell 3341</span>
# - <span style="color: blue">29: Abell 3376</span>
# - <span style="color: red">30: Abell 3389</span>
# - <span style="color: blue">31: Abell 3391</span>
# - <span style="color: blue">32: Abell 3395</span>
# - <span style="color: blue">33: Abell 3490</span>
# - <span style="color: blue">34: Abell 3497</span>
# - <span style="color: blue">35:</span> <span style="color: red">Abell 3526</span>
# - <span style="color: blue">36: Abell 3565</span>
# - 37: Abell 3571
# - <span style="color: blue">38: Abell 3574</span>
# - 39: Abell 3581
# - <span style="color: blue">40: Abell 3651</span>
# - <span style="color: blue">41: Abell 3667</span>
# - <span style="color: blue">42: Abell 3716</span>
# - 43: Abell 3809
# - <span style="color: blue">44: </span><span style="color: red">Abell 4038</span>
# - <span style="color: blue">45: Abell 4059</span>
# - 46: Abell S560
# - <span style="color: blue">47: Antlia</span>
# - 48: Fornax
# - 49: Hydra (A1060)
# - <span style="color: blue">50: IIZw108</span>
# - <span style="color: blue">51: MKW4</span>
# - <span style="color: blue">52: MKW8</span>

# In[25]:


###        ACA DEBERIA QUEDAR  TODO EN UN ARCHIVO CON LAS SUBESTRUCTURAS QUE VAMOS A OCUPAR PARA EL ANALISIS, ESPERANDO NUEVAS COLUMNAS

# READ FILES

# d = '/media/alonso_rodriguez/HDD/Santa María/10mo semestre/Tesis/'
d = 'D:\\Santa María\\10mo semestre\\Tesis\\'

parquet_file = os.path.join(d, 'dff_master.parquet')                                        # parquet file with the mocks (with redshift cuts)
# parquet_file = os.path.join(d, 'dff_master_53mocks.parquet')    

cluster_catalogue = pd.read_csv(d+'Mock-clusters_True_information.csv')                # Contains clusters specific information
# cluster_catalogue = pd.read_csv(d+'Mock-clusters_True_information_full.csv')     # Full 53 cluster information

df_subs = pd.read_parquet(d+'members_subs_master.parquet')                                # substructures
############################################################################################

if os.path.exists(parquet_file):                        # If parquet does not exist it creates one, might take some time but highly recommended 
    dff = pd.read_parquet(parquet_file)          # to accelerate the process next time

else:
    fits_files = sorted([f for f in os.listdir(d+'Mocks') if f.endswith('.fits') and 'with_true_subs' in f])
    cluster_dfs = []

    for fits_file in fits_files:
        mock_id = fits_file.split("_")[2]            # Extracts cluster id
        csv_file = f'CHANCES_lowz_{mock_id}_with_true_subs_true_members.csv'
        csv_path = os.path.join(d+'Mocks', csv_file)
        fits_path = os.path.join(d+'Mocks', fits_file)
        
        data = fits.getdata(fits_path, 1) 
        df = pd.DataFrame(np.array(data).byteswap().view(data.dtype.newbyteorder('=')))
        
        # Make cut
        # cluster_info = cluster_catalogue[cluster_catalogue['Cluster'] == int(mock_id)]
        # redshift_cl = cluster_info.iloc[0]['redshift']
        # sigma_cl = cluster_info.iloc[0]['sigma_true_r200']

        # Delta z = (Delta v) * (1 + z_cluster)
        # delta_v = 4 * sigma_cl                        
        # delta_z = (delta_v / c.to('km/s').value) * (1 + redshift_cl)   
        # z_min = redshift_cl - delta_z
        # z_max = redshift_cl + delta_z
        # df = df[(df['redshift_S_1'] >= z_min) & (df['redshift_S_1'] <= z_max)].copy()  # redshift cut at 4 sigma, if  you need everything, comment, 
                                                                                                                                                # mind RAM usage!
        df = df[df['redshift_S_1'] < 0.08].copy()   # redshift cut in 0.08 in order to work  only with low-z
                                                                              # if you do not want this, mind RAM usage (+60MM gal.)!
        
        df_members = pd.read_csv(csv_path)
        df['true_member'] = df['id'].isin(df_members['id']).astype(int)          # get true members
        df['mock_id'] = int(mock_id)
        cluster_dfs.append(df)
        print(f"Cluster {mock_id}: {len(df)} galaxias ({df['true_member'].sum()} miembros)")
        del data, df, df_members
        gc.collect()                                                            # releases RAM

    dff = pd.concat(cluster_dfs, ignore_index=True)  # concatenates clusters in a single dataframe 
    del cluster_dfs
    gc.collect()
    dff.to_parquet(parquet_file, index=False)           # make parquet

print(f"\nTotal N° galaxies: {len(dff)}")
print(f"Total true members: {dff['true_member'].sum()}")


# In[26]:


dff['true_member'].value_counts()


# Full dataset (42 mocks):
# 
# ![image.png](attachment:7eb1edcc-2efc-4fa7-8d17-4708a8555f2f.png)

# In[27]:


dff.columns


# ## Feature Engineering

# In[28]:


dff['mock_id'] = dff['mock_id'].astype(int)
cluster_catalogue['Cluster'] = cluster_catalogue['Cluster'].astype(int)

# initiate columns
dff['cluster_id_true'] = -1            # -1: no associated cluster
dff['cluster_id_closest']= -1
dff['D_closest_cl'] = np.inf
dff['D_3D_closest_cl'] = np.inf
dff['Vpec_closest_cl'] = np.nan
dff['D_real_mock'] = np.nan
dff['V_pec_mock'] = np.nan

cg = SkyCoord(dff['RA'].values * u.deg, dff['DEC'].values * u.deg, frame='icrs')             # skycoords for galaxies

for i, row in cluster_catalogue.iterrows():
    ##################   CLUSTER PROPERTIES   ###################
    
    x_cl, y_cl, z_cl = row['X_cent'] , row['Y_cent'] , row['Z_cent'] 
    cluster_id = row['Cluster']
    RA_cl, DEC_cl = row['RA'], row['Dec']
    redshift_cl = row['redshift']
    R200_mpc = row['R_200_mpc']
    # R200_kpc = R200_mpc*1e3*u.kpc 
    # R200_deg = ((cosmo.arcsec_per_kpc_proper(redshift_cl)*R200_kpc/(3600.*u.arcsec))*u.deg).value
    sigma_true_r200 = row['sigma_true_r200']

    D_3D_Mpc = np.sqrt((dff['x'] - x_cl)**2 + (dff['y']  - y_cl)**2 + (dff['z']  - z_cl)**2)
    mask_true = (D_3D_Mpc <= 5*R200_mpc)     # within 5R200 

    #######################    FEATURES    #######################
    
    cc = SkyCoord(RA_cl * u.deg, DEC_cl * u.deg, frame='icrs')                                      # skycoords for Cluster
    sep_deg = cg.separation(cc).deg                                                                                   # angular separation between cluster and each galaxy
    sep_mpc = ((cosmo.kpc_proper_per_arcmin(redshift_cl) * sep_deg * 60 * u.arcmin).value) / 1e3
    
    v_pec = c.to('km/s').value * (dff['redshift_S_1'] - redshift_cl) / (1 + redshift_cl)  # peculiar velocity

    #####################  NEW COLUMNS  #######################

    mask_3D = D_3D_Mpc < dff['D_3D_closest_cl']

    # Update if the cluster is the closest in 3D 
    dff.loc[mask_true, 'cluster_id_true'] = cluster_id 
    dff.loc[mask_3D, 'D_3D_closest_cl'] = D_3D_Mpc[mask_3D]
    dff.loc[mask_3D, 'V_true'] = v_pec[mask_3D]

    mask_closer = sep_mpc < dff['D_closest_cl']                                    
    
    # Update if the cluster is the closest in projection
    dff.loc[mask_closer, 'D_closest_cl'] = sep_mpc[mask_closer]                   # Projected distance to the closest cluster 
    dff.loc[mask_closer, 'Vpec_closest_cl'] = v_pec[mask_closer]                   # Vpec to the closest cluster
    dff.loc[mask_closer, 'cluster_id_closest'] = cluster_id

    mask_origin = (dff['mock_id'] == cluster_id)                                              # Only for Phase Space with 3D distance
    dff.loc[mask_origin, 'D_real_mock'] = D_3D_Mpc[mask_origin]
    dff.loc[mask_origin, 'V_pec_mock'] = v_pec[mask_origin]
    dff.loc[mask_origin, 'D_proj_mock'] = sep_mpc[mask_origin]

#############################################################
# R_proj and V_pec normalization

cols_delete = ['Cluster', 'R_200_mpc', 'sigma_true_r200',  'R_norm', 'V_norm', 'D_real_norm', 'R_200_mpc_x', 'R_200_mpc_y', 
                        'sigma_true_r200_x', 'sigma_true_r200_y', 'Cluster_x', 'Cluster_y']
dff = dff.drop(columns = cols_delete, errors = 'ignore')       # cleans columns if you already ran this cell, otherwise ignores

# Normalization with mock of origin properties (R_200 and sigma_200) 
dff = dff.merge(cluster_catalogue[['Cluster', 'R_200_mpc', 'sigma_true_r200']], left_on = 'mock_id', right_on = 'Cluster', how = 'left')
dff['R_norm'] = dff['D_proj_mock'] / dff['R_200_mpc']
dff['V_norm'] = dff['V_pec_mock'] / dff['sigma_true_r200']

# 3D distance vs. V.pec 
dff['D_real_norm_mock'] = dff['D_real_mock'] / dff['R_200_mpc']
dff['V_norm_mock'] = dff['V_pec_mock'] / dff['sigma_true_r200']

#############################################################
# Local density (surface density)

# RA, DEC to radians
coords_rad = np.deg2rad(dff[['DEC', 'RA']].values)

nbrs = NearestNeighbors(n_neighbors = 11, metric = 'haversine', n_jobs = -1).fit(coords_rad) # NN with k=10
dist_rad, _ = nbrs.kneighbors(coords_rad)

theta_10_rad = dist_rad[:, -1]                         # distance to 10th neighbor in rad
da_mpc = cosmo.angular_diameter_distance(dff['redshift_S_1'].values).value
r10 = theta_10_rad * da_mpc                        # distance to 10th neighbor in Mpc
dff['local_density'] = 10 / (np.pi * r10**2)  # [galaxies / Mpc^2]

dff['log_local_density'] = np.log10(dff['local_density'])

# 3D density (should not be used for the model)
coords_3d = dff[['x', 'y', 'z']].values

nbrs = NearestNeighbors(n_neighbors=11, n_jobs=-1).fit(coords_3d)
distances, _ = nbrs.kneighbors(coords_3d)

r10_3d = distances[:, -1]
dff['local_density_3D'] = 10 / ((4/3) * np.pi * r10_3d**3)  # [n galaxies / Mpc^3]


# Local Density (N-th neighbor): $$\Sigma_N = \frac{N}{\pi r^2_N}$$

# In[29]:


dff['mock_id'].value_counts()


# In[30]:


dff['cluster_id_true'].value_counts()


# ## Visualizations

# In[31]:


# inspect redshift distribution for the selected cluster

target_mock_id = 33
df_cluster = dff[(dff['mock_id'] == target_mock_id) & (dff['true_member'] == 1)].copy()

z_mean = df_cluster['redshift_S_1'].mean()
z_min = df_cluster['redshift_S_1'].min()
z_max = df_cluster['redshift_S_1'].max()
z_std = df_cluster['redshift_S_1'].std()

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(data = df_cluster, x = 'redshift_S_1', bins = 30, kde = True, color = 'crimson', alpha = 0.6, label = 'True Members')
ax.axvline(z_mean, color = 'black', linestyle = '--', linewidth=2, label = fr'Mean $z$: {z_mean:.4f}')
ax.axvline(z_min, color = 'blue', linestyle = ':', linewidth=1.5, label = fr'Min $z$: {z_min:.4f}')
ax.axvline(z_max, color = 'blue', linestyle = ':', linewidth=1.5, label = fr'Max $z$: {z_max:.4f}')

ax.set_xlabel(r'$z$', fontsize = 14)
ax.set_ylabel('Count', fontsize = 14)
ax.set_title(fr'Member $z$ Distribution - Cluster {target_mock_id}', fontsize=16)
ax.legend(fontsize=12)


deltaz = z_max - z_min
ax.text(0.79, 0.77, fr'$\Delta z$: {deltaz:.4f}', transform = ax.transAxes, fontsize = 12, va = 'top')

plt.tight_layout()
plt.show()


# In[32]:


# %matplotlib widget
# %matplotlib inline

fig, axs = plt.subplots(1, 2, figsize = (16, 6))
axs[0].scatter(dff['RA'], dff['DEC'], s = 0.01, alpha = 0.5)
axs[0].set_xlabel('RA [deg]')
axs[0].set_ylabel('DEC [deg]')
axs[0].set_title('Sky Distribution')
axs[0].invert_xaxis()
axs[1].hexbin(dff['x'], dff['y'], gridsize = 50, cmap = 'inferno')
hb = axs[1].hexbin(dff['x'], dff['y'], gridsize = 50, cmap = 'inferno', mincnt = 1)
cb = fig.colorbar(hb, ax = axs[1], label = 'N objects', shrink = 0.7)
axs[1].set_xlabel('x [Mpc]')
axs[1].set_ylabel('y [Mpc')
axs[1].set_title('Density map x-y')
axs[1].set_aspect('equal')
axs[1].invert_xaxis()
plt.show()


fig, axs = plt.subplots(1, 2, figsize = (12, 4))
axs[0].hist(dff['RA'], bins = 60, alpha = 0.7)
axs[0].set_title('RA Distribution')
axs[0].set_xlabel('RA')
axs[0].set_ylabel('counts')
axs[1].hist(dff['DEC'], bins = 60, alpha = 0.7, color = 'orange')
axs[1].set_title('DEC Distribution')
axs[1].set_xlabel('DEC')
axs[1].set_ylabel('counts')
plt.tight_layout()
plt.show()


# In[33]:


fig, axs = plt.subplots(1, 3, figsize = (18, 5))

# local density - 3D distance to closest cluster - Projected distance to closest cluster distributions
sns.histplot(dff['log_local_density'], bins = 150, color = 'crimson', ax = axs[0])
axs[0].set_xlabel(r'$\log_{10}(\Sigma_{10})$ [gal / Mpc$^2$]', fontsize = 14)
axs[0].set_ylabel('Count', fontsize = 14)
axs[0].set_title('Local Density Distribution', fontsize = 14)

sns.histplot(dff['D_3D_closest_cl'], bins = 300, color = 'darkblue', ax = axs[1])
axs[1].set_xlabel(r'$r_{3D}$ [Mpc]', fontsize = 14)
axs[1].set_ylabel('Count', fontsize = 14)
axs[1].set_title('3D Distance to closest cluster', fontsize = 14)

sns.histplot(dff['D_closest_cl'], bins = 300, color = 'orange', ax = axs[2])
axs[2].set_xlabel(r'$r_{proj}$ [Mpc]', fontsize = 14)
axs[2].set_ylabel('Count', fontsize = 14)
axs[2].set_title('Projected Distance to closest cluster', fontsize = 14)

plt.tight_layout()
plt.show()

# peculiar velocity  - r mag distributions
fig, axs = plt.subplots(1, 2, figsize = (14, 5))

sns.histplot(dff['Vpec_closest_cl'], bins = 200, ax = axs[0], color = 'darkblue')
axs[0].set_xlabel(r'$V_{pec}$ [km/s]', fontsize = 14)
axs[0].set_ylabel('Count', fontsize = 14)
axs[0].set_title('Peculiar Velocity Distribution', fontsize = 14)

sns.histplot(dff['r_mag'], bins = 200, ax = axs[1], color = 'green')
axs[1].set_xlabel(r'$m_r$', fontsize = 14)
axs[1].set_ylabel('Count', fontsize = 14)
axs[1].set_title(r'$r$-band Distribution', fontsize = 14)

plt.tight_layout()
plt.show()

# comparisons
fig, axs = plt.subplots(1, 3, figsize = (18, 6))

axs[0].scatter(np.abs(dff['Vpec_closest_cl']), dff['D_closest_cl'], s = 1, alpha = 0.5, color = 'teal')
axs[0].set_xlabel(r'$|V_{pec}|$ to closest cluster', fontsize = 14)
axs[0].set_ylabel(r'$r_{proj}$ to closest cluster', fontsize = 14)

axs[1].scatter(np.abs(dff['Vpec_closest_cl']), dff['D_3D_closest_cl'], s = 1, alpha = 0.5, color = 'purple')
axs[1].set_xlabel(r'$|V_{pec}|$ to closest cluster', fontsize = 14)
axs[1].set_ylabel(r'$r_{3D}$ to closest cluster', fontsize = 14)

axs[2].scatter(np.abs(dff['V_true']), dff['D_3D_closest_cl'], s = 1, alpha = 0.5, color = 'brown')
axs[2].set_xlabel(r'$|V_{pec}|$', fontsize = 14)
axs[2].set_ylabel(r'$r_{3D}$ to closest cluster', fontsize = 14)

plt.tight_layout()
plt.show()


# In[34]:


# output_dir = 'PPS_clusters'                       # Directory for PPS
# os.makedirs(output_dir, exist_ok=True)

#################      PHASE SPACE DIAGRAMS      ###################

members = dff[dff['true_member'] == 1]
interlopers = dff[dff['true_member'] == 0]

###############         INTERLOPERS          #############

plt.figure(figsize = (8, 6))
plt.scatter(interlopers['R_norm'], interlopers['V_norm'],  s = 3, color = 'gray', alpha = 0.3, label = 'Interlopers', rasterized = True)

plt.xlabel(r'$r_{proj} / r_{200}$', fontsize = 17)
plt.ylabel(r'$\Delta v / \sigma_{200}$', fontsize = 17)
plt.xlim(0, 5)
plt.ylim(-4, 4)
plt.legend(fontsize = 12, loc = 'upper right', fancybox = True)
plt.tight_layout()
# plt.savefig('PPS_clusters\\PPS_All_nomembers.pdf', dpi=450)
plt.show()

###############         MEMBERS & INTERLOPERS          #############

plt.figure(figsize = (8, 6))
plt.scatter(interlopers['R_norm'], interlopers['V_norm'],  s = 3, color = 'gray', alpha = 0.3, label = 'Interlopers', rasterized = True)
plt.scatter(members['R_norm'],  members['V_norm'],  s = 3, color = 'crimson', alpha = 0.5, label = 'Members', rasterized = True)
plt.xlabel(r'$r_{proj} / r_{200}$', fontsize = 17)
plt.ylabel(r'$\Delta v / \sigma_{200}$', fontsize = 17)
plt.xlim(0, 5)
plt.ylim(-4, 4)
plt.legend(fontsize = 12, loc = 'upper right', fancybox = True)
plt.tight_layout()
# plt.savefig('PPS_clusters\\PPS_All.pdf', dpi=300)
plt.show()


###############         NO LIMITS          #############

plt.figure()
plt.scatter(interlopers['R_norm'], interlopers['V_norm'],  s = 3, color = 'gray', alpha = 0.3, label = 'Interlopers')
plt.scatter(members['R_norm'], members['V_norm'],  s = 3, color = 'crimson', alpha = 0.5, label = 'Members')
plt.xlabel(r'$r_{proj} / r_{200}$', fontsize = 13)
plt.ylabel(r'$\Delta v / \sigma_{200}$', fontsize = 13)
plt.legend(fontsize = 10, loc = 'upper right')
plt.tight_layout()
plt.show()


# In[13]:


mask_interlopers = (interlopers['R_norm'] < 5) & (interlopers['V_norm'] < 4) & (interlopers['V_norm'] > -4)
interlopers_plot = interlopers[mask_interlopers]

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6), sharey = False)

kde_args = {'levels': 5, 'thresh': 0.2, 'linewidths': 2.5}

# Left panel
axes[0].scatter(interlopers_plot['R_norm'], interlopers_plot['V_norm'], s = 3, color = 'gray', alpha = 0.3, label = 'Interlopers', rasterized = True)
axes[0].scatter(members['R_norm'], members['V_norm'], s = 3, color = 'crimson', alpha = 0.5, label = 'Members', rasterized = True)
sns.kdeplot(x = members['R_norm'], y = members['V_norm'], ax = axes[0], color = 'darkred', zorder = 5, **kde_args)

legend_0 = [Line2D([0], [0], color = 'gray', marker = 'o', linestyle = 'None', markersize = 5, label = 'Interlopers', alpha = 0.5),
                     Line2D([0], [0], color = 'darkred', marker = 'o', markerfacecolor = 'crimson', markeredgecolor = 'crimson', 
                     markersize = 5, linestyle ='-', label = 'Members')]

axes[0].legend(handles = legend_0, fontsize = 14, loc = 'upper right', fancybox = True, framealpha = 0.9)

#  Right Panel
axes[1].scatter(interlopers_plot['R_norm'], interlopers_plot['V_norm'], s = 3, color = 'gray', alpha = 0.3, label = 'Interlopers', rasterized = True)
sns.kdeplot(x = interlopers_plot['R_norm'], y = interlopers_plot['V_norm'], ax = axes[1], color = '#333333', zorder = 5, **kde_args)

legend_1 = [Line2D([0], [0], color = '#333333', marker = 'o', markerfacecolor = 'gray', markeredgecolor = 'gray', 
                     markersize = 5, linestyle = '-', label = 'Interlopers', alpha = 0.8)]

axes[1].legend(handles = legend_1, fontsize = 14, loc = 'upper right', fancybox = True, framealpha = 0.9)

for ax in axes:
    ax.set_xlabel(r'$r_{proj} / r_{200}$', fontsize = 22)
    ax.set_ylabel(r'$\Delta v / \sigma_{200}$', fontsize = 22)
    ax.set_xlim(0, 5)
    ax.set_ylim(-4, 4)
    ax.tick_params(labelsize = 13)

#---------------------------------------------------------------------------------------------------------
# axes[1].axvspan(1.0, 5.0, color = 'orange', alpha = 0.1, zorder = 0, label = 'Infall Region')
# axes[1].text(3.2, -3.5, 'CLUSTER OUTSKIRTS\n(High Contamination)', fontsize = 15, color = 'black', ha = 'center', fontweight = 'bold', zorder = 10)

# r_vals = np.linspace(0.1, 5, 100)
# v_trumpet = 2.5 / np.sqrt(r_vals + 0.2) 

# for ax in axes:
    # ax.plot(r_vals, v_trumpet, color = 'k', linestyle = '--', linewidth = 2, alpha = 0.7)
    # ax.plot(r_vals, -v_trumpet, color = 'k', linestyle = '--', linewidth = 2, alpha = 0.7)

# axes[1].annotate('Interlopers projected\ninto the cluster', xy = (3, 0), xytext = (3.2, 2), arrowprops = dict(facecolor = 'k', shrink = 0.05),
             # fontsize = 11, fontweight = 'bold', color = 'k',zorder=10, bbox = dict(boxstyle = 'round,pad=0.3', fc = 'white', ec = 'gray', alpha = 0.9))
#---------------------------------------------------------------------------------------------------------
plt.tight_layout()
# plt.savefig('PPS_clusters\\PPS_All_Comparison.pdf', dpi=300) 
plt.show()


# In[15]:


# PPS for each cluster separately

output_dir = 'PPS_clusters'
os.makedirs(output_dir, exist_ok=True)

for cl_id in dff['mock_id'].unique():
    subset = dff[dff['mock_id'] == cl_id].copy()
    members_mask = subset['true_member'] == 1
    nonmembers_mask = subset['true_member'] == 0
    
    plt.figure(figsize = (7,5))
    plt.scatter(subset.loc[members_mask, 'R_norm'], subset.loc[members_mask, 'V_norm'], s = 6, alpha = 0.6, color = 'crimson', label = 'Members')
    plt.scatter(subset.loc[nonmembers_mask,'R_norm'], subset.loc[nonmembers_mask, 'V_norm'], s = 5, alpha = 0.3, color = 'gray', label ='Interlopers')
    plt.xlabel(r'$r / r_{200}$', fontsize = 14)
    plt.ylabel(r'$\Delta v / \sigma_{200}$', fontsize = 14)
    plt.title(f'Mock {cl_id} Projected Phase Space', fontsize = 14)
    plt.xlim(0, 5)
    plt.ylim(-4, 4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'PPS_cluster_{cl_id}.png'), dpi = 250)
    plt.close()


# In[14]:


# Phase Space in 3D

plt.figure(figsize = (8, 6))
plt.scatter(interlopers['D_real_norm_mock'],  interlopers['V_norm_mock'],  s = 5, color = 'gray', alpha = 0.3, label = 'Interlopers')
plt.scatter(members['D_real_norm_mock'],  members['V_norm_mock'],  s = 5, color = 'crimson', alpha = 0.5, label = 'Members')

plt.xlabel(r'$r_{3D} / r_{200}$', fontsize = 16)
plt.ylabel(r'$\Delta v / \sigma_{200}$', fontsize = 16)
plt.xlim(0, 40)
plt.ylim(-7, 7)
plt.legend(fontsize = 12, loc = 'best')
plt.tight_layout()
plt.show()


# In[41]:


dff.columns


# In[70]:


labels = {'Mvir': r'$M_{200}$ [$M_{\odot}$]',
               'log_local_density': r'$\log(\Sigma_{10})$ [Mpc$^{-2}$]', 
               'r_mag': r'$m_r$',
               'V_norm': r'$\Delta v / \sigma_{200}$',
               'R_norm': r'$r_{proj} / r_{200}$'}

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 6)

ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])
ax3 = fig.add_subplot(gs[0, 4:6])
ax4 = fig.add_subplot(gs[1, 1:3])
ax5 = fig.add_subplot(gs[1, 3:5])
axes = [ax1, ax2, ax3, ax4, ax5]
var_keys = ['Mvir', 'log_local_density', 'r_mag', 'V_norm', 'R_norm']

for i, col_name in enumerate(var_keys):
    ax = axes[i]
    members = dff[dff['true_member'] == 1][col_name]
    interlopers = dff[dff['true_member'] == 0][col_name]

    # Bins
    if col_name == 'Mvir':
        ax.set_xscale('log')
        min_val = min(members.min(), interlopers.min())
        max_val = max(members.max(), interlopers.max())
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 80)
    else:
        bins = 60

    # Normalized histograms
    ax.hist(interlopers, bins = bins, color = 'gray', alpha = 0.4, label = 'Interlopers', density = True, histtype = 'stepfilled', edgecolor = 'k')
    ax.hist(members, bins = bins, color = 'crimson', alpha = 0.5, label = 'Members', density = True, histtype = 'stepfilled', edgecolor = 'red')
    
    ax.set_xlabel(labels[col_name], fontsize = 16)
    ax.set_ylabel('Probability Density', fontsize = 14)

plt.suptitle(f'Distribution of Features', fontsize = 20)
plt.tight_layout()

plt.savefig(f'Features_distribution.pdf', dpi = 250, bbox_inches = 'tight')
plt.show()


# In[15]:


#  N° members per cluster, clusters with N_memb < 150 are excluded to train the models 

cluster_catalogue_full = pd.read_csv(d+'Mock-clusters_True_information_full.csv')   # 53 cluster csv
member_count = cluster_catalogue_full['all_members']

members_sorted = member_count.sort_values(ascending = False).reset_index(drop = True)
colors = ['#12436D' if n > 150 else 'lightblue' for n in members_sorted]

fig, ax = plt.subplots(figsize=(12, 6))

# Bars
bars = ax.bar(members_sorted.index, members_sorted, color = colors, alpha = 0.9, width = 0.8)
ax.axhline(y = 150, color = 'red', linestyle = '--', linewidth = 1.8, zorder = 5)
ax.text(x = len(members_sorted)-1, y = 165, s = r'Threshold (N > 150)', color = 'crimson', fontsize = 12, fontweight = 'bold', ha = 'right', va = 'bottom')

ax.set_xlabel('Cluster member rank', fontsize = 16)
ax.set_ylabel(r'Member count', fontsize = 16)
ax.set_xlim(-1, len(members_sorted))
ax.tick_params(labelsize = 13)

kept_patch = Patch(color = '#12436D', label = 'Selected Clusters')
excluded_patch = Patch(color = 'lightblue', label = 'Excluded ($N \leq 150$)')
ax.legend(handles=[kept_patch, excluded_patch], fontsize = 13, loc = 'upper right', fancybox = True, framealpha = 0.9)

plt.tight_layout()
# plt.savefig('Nmember_cut.pdf', dpi=250)
plt.show()


# In[16]:


# %matplotlib widget


# In[61]:


fig = plt.figure(figsize = (18, 9), layout = 'constrained')

target_cl = cluster_catalogue.iloc[0]   # mock 0 (Abell 85)
cluster_id = int(target_cl['Cluster'])
x_cent, y_cent, z_cent = target_cl['X_cent'], target_cl['Y_cent'], target_cl['Z_cent']
R200 = target_cl['R_200_mpc']

df_members = dff[(dff['true_member'] == 1) & (dff['mock_id'] != cluster_id)]    # all members but Abell 85
df_a85 = dff[(dff['mock_id'] == cluster_id) & (dff['true_member'] == 1)]            # Abell 85 members

#=========================================
# LEFT PANEL
#=========================================
ax1 = fig.add_subplot(1, 2, 1, projection = '3d')

df_nonmembers = dff[dff['true_member'] == 0]
nonmemb_sample = df_nonmembers.sample(frac = 0.5, random_state = 42)    # downsampling to optimize and not saturate plot

ax1.scatter(nonmemb_sample['x'], nonmemb_sample['y'], nonmemb_sample['z'], color = 'gray', s = 0.1, alpha = 0.1,             # Interlopers
                    label = 'Interlopers (Sample)', zorder = 1, rasterized = True) 

ax1.scatter(df_members['x'], df_members['y'], df_members['z'], color = 'crimson', edgecolors = 'white', s = 7, alpha = 0.8,   # Members
                    label = 'Members', zorder = 10, linewidth = 0.1, rasterized = True) 

ax1.scatter(df_a85['x'], df_a85['y'], df_a85['z'], color = 'blue', edgecolors = 'black', s = 20, alpha = 0.8, zorder = 100,           # Members A85
                    label = 'Members A85 Mock', linewidth = 0.5)

ax1.set_xlabel('x [Mpc]', fontsize = 14, labelpad = 10)
ax1.set_ylabel('y [Mpc]', fontsize = 14, labelpad = 10)
ax1.set_zlabel('z [Mpc]', fontsize = 14, labelpad = 10)

ax1.set_box_aspect([1, 1, 1])
ax1.view_init(elev = 30, azim = -60)   # view

# custom legend
legend = ax1.legend(loc = 'upper right', frameon = True, fancybox = True, framealpha = 0.8, fontsize = 12)
legend.legend_handles[0]._sizes = [20] 
legend.legend_handles[0].set_alpha(0.4)
legend.legend_handles[1]._sizes = [30]
legend.legend_handles[1]._sizes = [30]
legend.legend_handles[2]._sizes = [30]

#=========================================
# RIGHT PANEL
#=========================================
ax2 = fig.add_subplot(1, 2, 2, projection = '3d')

# Filter mock
df_true = dff[dff['mock_id'] == cluster_id]
df_mock = df_true.copy()

# center coords
df_mock['x0'] = df_mock['x'] - x_cent
df_mock['y0'] = df_mock['y'] - y_cent
df_mock['z0'] = df_mock['z'] - z_cent

lim = 5 * R200 * 1.2       
mask_in_lim = ((df_mock['x0'] >= -lim) & (df_mock['x0'] <= lim) &    # mask for points located within [-lim, lim] in 3D
                           (df_mock['y0'] >= -lim) & (df_mock['y0'] <= lim) &
                           (df_mock['z0'] >= -lim) & (df_mock['z0'] <= lim))

df_plot = df_mock[mask_in_lim]
memb = df_plot['true_member'] == 1
nonmemb = df_plot['true_member'] == 0

ax2.scatter(df_plot.loc[nonmemb, 'x0'], df_plot.loc[nonmemb, 'y0'], df_plot.loc[nonmemb, 'z0'], color = 'gray', s = 5, alpha = 0.7, label = 'Interlopers')
ax2.scatter(df_plot.loc[memb, 'x0'], df_plot.loc[memb, 'y0'], df_plot.loc[memb, 'z0'], color = 'red', s = 8, alpha = 0.7, label = 'Members')

# 5R200 Sphere
theta = np.linspace(0, 2 * np.pi, 80)
phi = np.linspace(0, np.pi, 40)
theta, phi = np.meshgrid(theta, phi)
r = 5 * R200

xx = r * np.sin(phi) * np.cos(theta)
yy = r * np.sin(phi) * np.sin(theta)
zz = r * np.cos(phi)
ax2.plot_surface(xx, yy, zz, color = 'blue', alpha = 0.15, linewidth = 0)

ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.set_zlim(-lim, lim)
ax2.set_xlabel('x [Mpc]', fontsize = 14, labelpad = 8)
ax2.set_ylabel('y [Mpc]', fontsize = 14, labelpad = 8)
ax2.set_zlabel('z [Mpc]', fontsize = 14, labelpad = 8)
ax2.set_title(f'A85 Mock 3D Distribution', fontsize = 16)
ax2.legend(fontsize = 13)
ax2.set_box_aspect([1,1,1])
grid_style = {'color': 'gray', 'linestyle': ':', 'linewidth': 0.2, 'alpha': 0.3}

for ax in [ax1, ax2]:
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.xaxis._axinfo['grid'].update(grid_style)
    ax.yaxis._axinfo['grid'].update(grid_style)
    ax.zaxis._axinfo['grid'].update(grid_style)

# =========== Zoom-in lines  =============

start_top = (0.208, 0.604)        # starting points
start_bot = (0.208, 0.59)

top_line = ConnectionPatch(xyA = start_top, coordsA = 'figure fraction', xyB = (0.45, 0.77), coordsB = 'axes fraction',    # Top line
                          axesA = ax1, axesB = ax2, arrowstyle = '-', color = 'k', linestyle = '--', linewidth = 0.8, alpha = 0.6)

bot_line = ConnectionPatch(xyA = start_bot, coordsA = 'figure fraction', xyB = (0.45, 0.26), coordsB = 'axes fraction',   # Bottom line
                              axesA = ax1, axesB = ax2, arrowstyle = '-', color = 'k', linestyle = '--', linewidth = 0.8, alpha = 0.6)
                             
fig.add_artist(top_line)
fig.add_artist(bot_line)

plt.savefig('Mocks_3D+A85.png', dpi = 250, bbox_inches = 'tight', pad_inches = 0.25)
plt.show()


# In[44]:


# 3D plot for each cluster

output_dir = '3Dplot_clusters'
os.makedirs(output_dir, exist_ok = True)

# Iterate over clusters
for i, row in cluster_catalogue.iterrows():
    cluster_id = int(row['Cluster'])
    x_cent, y_cent, z_cent = row['X_cent'], row['Y_cent'], row['Z_cent']
    R200 = row['R_200_mpc']

    # Search mock_id of true members of this cluster
    df_true = dff[dff['mock_id'] == cluster_id]

    mock_id = df_true['mock_id'].iloc[0]
    df_mock = dff[dff['mock_id'] == mock_id].copy()  # galaxies in the mock

    # 3D distance to cluster center
    df_mock['x0'] = df_mock['x'] - x_cent
    df_mock['y0'] = df_mock['y'] - y_cent
    df_mock['z0'] = df_mock['z'] - z_cent
    df_mock['Dist_Mpc'] = np.sqrt(df_mock['x0']**2 + df_mock['y0']**2 + df_mock['z0']**2)

    lim = 5 * R200 * 1.2
    mask_in_lim = ((df_mock['x0'] >= -lim) & (df_mock['x0'] <= lim) &
                               (df_mock['y0'] >= -lim) & (df_mock['y0'] <= lim) &
                               (df_mock['z0'] >= -lim) & (df_mock['z0'] <= lim))

    df_plot = df_mock[mask_in_lim]

    memb = df_plot['true_member'] == 1
    nonmemb = df_plot['true_member'] == 0

    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df_plot.loc[nonmemb, 'x0'], df_plot.loc[nonmemb, 'y0'], df_plot.loc[nonmemb, 'z0'], color = 'gray', s = 5, alpha = 0.7, label = 'Interlopers')
    ax.scatter(df_plot.loc[memb, 'x0'], df_plot.loc[memb, 'y0'], df_plot.loc[memb, 'z0'], color = 'red', s = 8, alpha = 0.7, label = 'Members')

    # 5R200 Sphere
    theta = np.linspace(0, 2 * np.pi, 80)
    phi = np.linspace(0, np.pi, 40)
    theta, phi = np.meshgrid(theta, phi)
    r = 5 * R200
    xx = r * np.sin(phi) * np.cos(theta)
    yy = r * np.sin(phi) * np.sin(theta)
    zz = r * np.cos(phi)
    ax.plot_surface(xx, yy, zz, color = 'blue', alpha = 0.15, linewidth = 0)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    grid_style = {'color': 'gray', 'linestyle': ':', 'linewidth': 0.2, 'alpha': 0.3}
    ax.xaxis._axinfo["grid"].update(grid_style)
    ax.yaxis._axinfo["grid"].update(grid_style)
    ax.zaxis._axinfo["grid"].update(grid_style)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('x [Mpc]', fontsize = 14, labelpad = 10)
    ax.set_ylabel('y [Mpc]', fontsize = 14, labelpad = 10)
    ax.set_zlabel('z [Mpc]', fontsize = 14, labelpad = 10)
    ax.set_title(f'Mock {cluster_id} 3D Distribution')
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'3Dplot_cluster_{cluster_id}.pdf'), dpi = 250)
    plt.close(fig) 


# In[39]:


cluster_names = {
    0: 'Abell 85', 1: 'Abell 119', 2: 'Abell 133', 3: 'Abell 147', 4: 'Abell 151', 5: 'Abell 168',
    6: 'Abell 194', 7: 'Abell 496', 8: 'Abell 500', 9: 'Abell 548', 10: 'Abell 754', 11: 'Abell 780',
    12: 'Abell 957', 13: 'Abell 970', 14: 'Abell 1069', 15: 'Abell 1520', 16: 'Abell 1631', 
    17: 'Abell 1644', 18: 'Abell 2399', 19: 'Abell 2415', 20: 'Abell 2457', 21: 'Abell 2717', 
    22: 'Abell 2734', 23: 'Abell 2870', 24: 'Abell 2877', 25: 'Abell 3223', 26: 'Abell 3266',
    27: 'Abell 3301', 28: 'Abell 3341', 29: 'Abell 3376', 30: 'Abell 3389', 31: 'Abell 3391', 
    32: 'Abell 3395', 33: 'Abell 3490', 34: 'Abell 3497', 35: 'Abell 3526', 36: 'Abell 3565', 
    37: 'Abell 3571', 38: 'Abell 3574', 39: 'Abell 3581', 40: 'Abell 3651', 41: 'Abell 3667',
    42: 'Abell 3716', 43: 'Abell 3809', 44: 'Abell 4038', 45: 'Abell 4059', 46: 'Abell S560',
    47: 'Antlia', 48: 'Fornax', 49: 'Hydra (A1060)', 50: 'IIZw108', 51: 'MKW4', 52: 'MKW8'}

################################################
##########     PLOT MOCKS 3D + PPS     ###############
################################################

# 3D Cluster plots
# PRELIMINARY PPS w/ substructures, run with results in hand (Metrics_per_cluster.csv) to include Purity & Completeness in title

# df_metrics = pd.read_csv(d+'RF\\Metrics_per_cluster.csv')

output_combined_dir = 'Combined_3D_PPS_clusters'
os.makedirs(output_combined_dir, exist_ok = True)

dff = dff.drop(columns = ['3d_true_members', 'specz_members', 'photz_members_lsdr10', 'subs_labels_0'], errors = 'ignore')
dff  = dff.merge(df_subs, on = ['id', 'mock_id'], how = 'left')                    # merge with substructures catalogue (and photz and specz memb., not needed)
cluster_catalogue['Cluster'] = cluster_catalogue['Cluster'].astype(int)

# Iterate over mocks
for i, row in cluster_catalogue.iterrows():
    cluster_id = int(row['Cluster'])
    cluster_real_name = cluster_names.get(cluster_id)   # switch from mock_id to cluster name
    title = f'{cluster_real_name}'
    
    if 'df_metrics' in locals():        # metrics, if available
        metric_row = df_metrics[df_metrics['mock_id'] == cluster_id]   
        if not metric_row.empty:
            pur = metric_row.iloc[0]['purity']
            comp = metric_row.iloc[0]['completeness']
            title += f' | P: {pur:.2f} | C: {comp:.2f}'

    z_cl = row['redshift']
    m200 = row['M_200_1e14Mo']
    r200 = row['R_200_mpc']
    sigma200 = row['sigma_true_r200']
    n_memb = int(row['all_members'])
    
    props_text = (f'$z = {z_cl:.3f}$\n'                                                                                   # cluster properties box
                            f'$M_{{200}} = {m200:.2f} \\times 10^{{14}} M_{{\\odot}}$\n'
                            f'$r_{{200}} = {r200:.2f}$ Mpc\n'
                            f'$\\sigma_{{v}} = {sigma200:.0f}$ km/s\n'
                            f'$N_{{mem}} (< 5r_{{200}}) = {n_memb}$')
                
    x_cent, y_cent, z_cent = row['X_cent'], row['Y_cent'], row['Z_cent']
    R200 = row['R_200_mpc']
    sigma_true_r200 = row['sigma_true_r200']

    df_mock = dff[dff['mock_id'] == cluster_id].copy()  # galaxies in mock

    df_mock['x0'] = df_mock['x'] - x_cent      # center coords
    df_mock['y0'] = df_mock['y'] - y_cent
    df_mock['z0'] = df_mock['z'] - z_cent

    lim_3d = 5 * R200 * 1.2
    mask_lim = ((df_mock['x0'] >= -lim_3d) & (df_mock['x0'] <= lim_3d) &
                          (df_mock['y0'] >= -lim_3d) & (df_mock['y0'] <= lim_3d) &
                          (df_mock['z0'] >= -lim_3d) & (df_mock['z0'] <= lim_3d))
    
    df_plot_3d = df_mock[mask_lim]
    memb_3d = df_plot_3d['true_member'] == 1
    nonmemb_3d = df_plot_3d['true_member'] == 0

    lim_pps = (df_mock['R_norm'] <= 5) & (df_mock['V_norm'].abs() <= 4)
    subset_pps = df_mock[lim_pps]
    
    nonmemb_pps = subset_pps['true_member'] == 0
    memb_pps = (subset_pps['true_member'] == 1) & (subset_pps['subs_labels_0'] == -1)
    mask_subs = (subset_pps['true_member'] == 1) & (subset_pps['subs_labels_0'] != -1)
    mask_subs_interloper = (subset_pps['true_member'] == 0) & (subset_pps['subs_labels_0'] != -1)
    
    fig = plt.figure(figsize = (18, 8))
    gs = fig.add_gridspec(1, 2) 
    
    # -------------------      3D        ------------------------
    
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    ax1.scatter(df_plot_3d.loc[nonmemb_3d, 'x0'], df_plot_3d.loc[nonmemb_3d, 'y0'], df_plot_3d.loc[nonmemb_3d, 'z0'], color = 'gray', 
                        s = 5, alpha = 0.7, label = 'Interlopers')
    
    ax1.scatter(df_plot_3d.loc[memb_3d, 'x0'], df_plot_3d.loc[memb_3d, 'y0'], df_plot_3d.loc[memb_3d, 'z0'], color = 'red', s = 8,
                        alpha = 0.7, label = 'Members') 

    # 5R200 Sphere
    theta = np.linspace(0, 2 * np.pi, 80)
    phi = np.linspace(0, np.pi, 40)
    theta, phi = np.meshgrid(theta, phi)
    r_sphere = 5 * R200
    xx = r_sphere * np.sin(phi) * np.cos(theta)
    yy = r_sphere * np.sin(phi) * np.sin(theta)
    zz = r_sphere * np.cos(phi)
    ax1.plot_surface(xx, yy, zz, color = 'blue', alpha = 0.15, linewidth = 0)

    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('w')
    ax1.yaxis.pane.set_edgecolor('w')
    ax1.zaxis.pane.set_edgecolor('w')
    grid_style = {'color': 'gray', 'linestyle': ':', 'linewidth': 0.2, 'alpha': 0.3}
    ax1.xaxis._axinfo['grid'].update(grid_style)
    ax1.yaxis._axinfo['grid'].update(grid_style)
    ax1.zaxis._axinfo['grid'].update(grid_style)

    ax1.set_xlim(-lim_3d, lim_3d)
    ax1.set_ylim(-lim_3d, lim_3d)
    ax1.set_zlim(-lim_3d, lim_3d)
    ax1.set_xlabel('x [Mpc]', fontsize = 16, labelpad = 10)
    ax1.set_ylabel('y [Mpc]', fontsize = 16, labelpad = 10)
    ax1.set_zlabel('z [Mpc]', fontsize = 16, labelpad = 10)
    ax1.legend(fontsize = 14, loc = 'upper right', fancybox = True)
    ax1.set_box_aspect([1,1,1]) 
    
    # -------------          PPS          ------------------------
    
    ax2 = fig.add_subplot(gs[0, 1]) 
    
    ax2.scatter(subset_pps.loc[nonmemb_pps, 'R_norm'], subset_pps.loc[nonmemb_pps, 'V_norm'],  marker = 'x', s = 5, alpha = 0.3, color = 'gray', label = 'Interlopers')
    ax2.scatter(subset_pps.loc[memb_pps, 'R_norm'], subset_pps.loc[memb_pps, 'V_norm'], s = 6, alpha = 0.5, color = 'crimson', label = 'Members') 

    
    # Color substructures within the cluster (found in 3D; previously done with HDBSCAN by Franco)    
    unique_subs = sorted(subset_pps.loc[mask_subs, 'subs_labels_0'].unique())                       # member substructures
    cmap = plt.get_cmap('gist_rainbow', len(unique_subs))
    
    for idx, sub_id in enumerate(unique_subs):
        sub_data = subset_pps[(subset_pps['subs_labels_0'] == sub_id) & (subset_pps['true_member'] == 1)]
        ax2.scatter(sub_data['R_norm'], sub_data['V_norm'], s = 10, alpha = 0.8, color = cmap(idx), marker = 'o', edgecolor = 'k', linewidth = 0.8)

    unique_subs = sorted(subset_pps.loc[mask_subs_interloper, 'subs_labels_0'].unique())    # interloper substructures
    cmap = plt.get_cmap('gist_rainbow', len(unique_subs))
    
    for idx, sub_id in enumerate(unique_subs):
        sub_data = subset_pps[(subset_pps['subs_labels_0'] == sub_id) & (subset_pps['true_member'] == 0)]
        ax2.scatter(sub_data['R_norm'], sub_data['V_norm'], s = 10, alpha = 0.8, color = cmap(idx), marker = 'x', linewidth = 1.2)

    # cluster properties
    ax2.text(0.01, 0.99, props_text, transform = ax2.transAxes, fontsize = 12, va = 'top', bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.7))
    
    ax2.set_xlabel(r'$r_{proj} / r_{200}$', fontsize = 22)
    ax2.set_ylabel(r'$\Delta v / \sigma_{200}$', fontsize = 22)
    ax2.tick_params(labelsize = 13)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(-4, 4)

    # custom legend
    handles = [Line2D([0], [0], marker = 'x', color = 'w', label = 'Interlopers', markeredgecolor = 'gray', markeredgewidth = 2, markersize = 6, alpha = 0.5),
                       Line2D([0], [0], marker = 'o', color = 'w', label = 'Members', markerfacecolor = 'crimson', markersize = 6, alpha = 0.7),
                       Line2D([0], [0], marker = 'o', color = 'w', label = 'Member subs.', markeredgecolor = 'k', markeredgewidth = 1.5, markersize = 6),
                       Line2D([0], [0], marker = 'x', color = 'w', label = 'Interloper subs.', markeredgecolor = 'k', markeredgewidth = 2, markersize = 6)]
    
    ax2.legend(handles = handles, fontsize = 12, loc = 'upper right', framealpha = 0.8, fancybox = True)
    
    plt.suptitle(title, fontsize = 22, y = 1.04)
    plt.tight_layout() 
    plt.savefig(os.path.join(output_combined_dir, f'Combined_Plot_Mock_{cluster_id}.pdf'), dpi = 250, bbox_inches = 'tight')
    plt.close(fig)


# ## Labelling

# In[11]:


features = ['V_norm', 'R_norm', 'log_local_density', 'Mvir', 'r_mag']     # features to use
RF_features = dff[features]


# In[12]:


names = {'V_norm': r'$V_{pec}/\sigma_{200}$',
                'R_norm': r'$r_{proj}/r_{200}$',
                'log_local_density': r'log $\Sigma_{10}$', 
                'Mvir': r'$M_{200}$',
                'r_mag': r'$m_{r}$'}

plt.figure(figsize = (10,10))
ax = sns.heatmap(RF_features.rename(columns = names).corr(), annot = True, cbar = True, cmap = 'coolwarm',  cbar_kws = {'shrink': .9}, 
                                vmin = -1, vmax = 1, square = True, annot_kws = {'size': 12})

ax.tick_params(labelsize = 14, length = 0, pad = 7)
cbar = ax.collections[0].colorbar
cbar.set_label('Correlation', fontsize = 15, rotation = 270, labelpad = 15)
cbar.ax.tick_params(labelsize = 12)

plt.gca().set_aspect('equal')
plt.savefig('Correlation Matrix.pdf', dpi = 300, bbox_inches = 'tight')
plt.show()


# In[14]:


dff['true_member'].value_counts()


# In[15]:


X = RF_features
y = dff['true_member']
groups = dff['mock_id'].values

# cv = StratifiedGroupKFold(n_splits = 3, shuffle=True, random_state=41)    # recommended only if training with a low amount of clusters (e.g., < 4)
cv = LeaveOneGroupOut()
threshold = 0.45    # threshold to consider membership (P >= 0.45   ->  member)

print(f'Unique groups: {len(np.unique(groups))}') 


# ## Model Training

# ### Benchmark - RF 'naked'

# In[61]:


# Random Forest Classifier with no hyperparameter optimization, default configuration with exception of the class weighting

benchmark = RandomForestClassifier(random_state = 1, class_weight = 'balanced', n_jobs = -1)


# In[20]:


cv_results = cross_validate(benchmark, X, y, groups = groups, cv = cv, scoring = ['accuracy', 'precision', 'recall', 'f1'], return_train_score = True, n_jobs = -1) # cross validation

print('CV mean ± std:')
for m in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
    print(f' {m}: {cv_results[m].mean():.3f} ± {cv_results[m].std():.3f}')

y_prob =  cross_val_predict(benchmark, X, y, groups = groups, cv = cv, method = 'predict_proba', n_jobs = -1)[:,1]   # probabilities for membership


# In[21]:


y_pred = (y_prob >= threshold).astype(int)          # predictions, threshold dependant
print('Classification report:')
print(classification_report(y, y_pred, digits = 3))

cm = confusion_matrix(y, y_pred)


# In[22]:


plot_confusion_matrix(cm, ['Interloper', 'Member'], model_name = 'Benchmark', cmap = 'Blues')


# In[23]:


TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0] , cm[1][1]
purity = TP / (TP + FP)
completeness = TP / (TP + FN)

print('Purity/Precision : ', precision_score(y, y_pred))
print('Completeness/Recall: ', recall_score(y, y_pred))
print('F1 : ', f1_score(y, y_pred))
print('ROC AUC: ', roc_auc_score(y, y_prob))
print('AP (PR AUC): ', average_precision_score(y, y_prob))

# ROC  curve
plot_roc_curve(y, y_prob, threshold, model_name = 'Benchmark')


# In[25]:


plot_learning_curve(benchmark, X, y, groups = groups, cv = cv, scoring = 'f1', model_name = 'Benchmark')


# In[26]:


plot_feature_importances(benchmark, X, y, model_name = 'Benchmark')


# In[27]:


# Save trained model as .pkl, allows to reuse the model for analysis later without retraining
joblib.dump(benchmark, 'RF_benchmark.pkl')

# Create results dataframe with labels, and predictions
results = X.copy()
results['y_true'] = y.values
results['y_pred'] = y_pred
results['y_prob'] = y_prob

# Re-attach identification columns (id, mock_id) from the master dataframe
results = results.join(dff[['id', 'mock_id']], how = 'left')

# Classify each point as TP, TN, FP, FN
conditions = [
    (results['y_true'] == 1) & (results['y_pred'] == 1),
    (results['y_true'] == 0) & (results['y_pred'] == 0),
    (results['y_true'] == 0) & (results['y_pred'] == 1),
    (results['y_true'] == 1) & (results['y_pred'] == 0)]

labels = ['TP', 'TN', 'FP', 'FN']
results['class'] = np.select(conditions, labels, default = 'None')

# Save result ot a parquet file (load easily with pd.read_parquet())
results.to_parquet('RF_benchmark.parquet', index=False)


# In[28]:


plot_phase_space(results, 'Benchmark')


# In[29]:


plot_purity_and_completeness(y, y_pred, results, 'Benchmark')


# ### RF with hyperparameter optimization

# Final Parameters (recommended to avoid the gridsearch):
# 
# {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 20, 'max_features': None, 
# 
# 'max_leaf_nodes': 80, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 50, 
# 
# 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 300, 'n_jobs': None, 'oob_score': False, 
# 
# 'random_state': 14, 'verbose': 0, 'warm_start': False}

# In[192]:


# Hyperparameter grid
parameters = { 
              'n_estimators': [100, 200, 300], 
              'min_impurity_decrease': [0.0, 0.01, 0.1], 
              'max_depth': [5, 10, 20], 
              'min_samples_split': [10, 20, 50], 
              'max_leaf_nodes': [20, 50, 80], 
              'max_features': [None], 
              'class_weight': ['balanced'] }

model = GridSearchCV(estimator = RandomForestClassifier(random_state = 14), param_grid = parameters, scoring = 'f1', cv = cv, n_jobs = -1, verbose = 2, return_train_score = True)


# In[ ]:


# Option 1: Run gridsearch. BEWARE: might take a long time depending on your CPU capabilitites. If you do not possess enough time, you can run
# gridsearch with an alternative CV such as StratifiedKFold first to find the optimal parameters and then run LOGO CV with the new parameters.

get_ipython().run_line_magic('%time', '')
model.fit(X, y, groups = groups)
 
print('Best score:', model.best_score_)
print('Best parameters:', model.best_params_)

RF_optimized = model.best_estimator_
y_prob = cross_val_predict(RF_optimized, X, y, groups = groups, cv = cv, method = 'predict_proba', n_jobs = -1)[:,1] 


# In[17]:


# Option 2: Use parameters from previous gridsearch (listed above in this section)

RF_optimized =  joblib.load(d+'RF\\RF_gridsearch.pkl')
best_params = RF_optimized.get_params()
RF_optimized = RandomForestClassifier(**best_params)

cv_results = cross_validate(RF_optimized, X, y, groups = groups, cv = cv, scoring = ['accuracy', 'precision', 'recall', 'f1'], return_train_score = True, n_jobs = -1)  # Cross-validation

print('CV mean ± std:')
for m in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
    print(f' {m}: {cv_results[m].mean():.3f} ± {cv_results[m].std():.3f}')

y_prob = cross_val_predict(RF_optimized, X, y, groups = groups, cv = cv, method = 'predict_proba', n_jobs = -1)[:,1]


# In[82]:


y_pred = (y_prob >= threshold).astype(int) 
cm = confusion_matrix(y,y_pred) 


# In[83]:


plot_confusion_matrix(cm, ['Interloper', 'Member'], model_name = 'RF_optimized', cmap = 'Blues')


# In[84]:


TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0] , cm[1][1]
purity = TP / (TP + FP)
completeness = TP / (TP + FN)

print('Purity/Precision : ', precision_score(y, y_pred))
print('Completeness/Recall: ', recall_score(y, y_pred))
print('F1 : ', f1_score(y, y_pred))
print('ROC AUC: ', roc_auc_score(y, y_prob))
print('AP (PR AUC): ', average_precision_score(y, y_prob))

plot_roc_curve(y, y_prob, threshold, model_name = 'RF_optimized')


# In[31]:


plot_learning_curve(RF_optimized, X, y, groups = groups, ylim = (0.6, 1.0), cv = cv, scoring = 'f1', model_name = 'RF_optimized')


# In[85]:


importance = plot_feature_importances(RF_optimized, X, y, model_name = 'RF_optimized')


# In[86]:


# Save trained model as .pkl, allows to reuse the model for analysis later without retraining
joblib.dump(RF_optimized, 'RF_Final.pkl')

# Create results dataframe with labels, and predictions
results = X.copy()
results['y_true'] = y.values
results['y_pred'] = y_pred
results['y_prob'] = y_prob

# Re-attach identification columns (id, mock_id) from the master dataframe
results = results.join(dff[['id', 'mock_id']], how = 'left')

# Classify each point as TP, TN, FP, FN
conditions = [
    (results['y_true'] == 1) & (results['y_pred'] == 1),
    (results['y_true'] == 0) & (results['y_pred'] == 0),
    (results['y_true'] == 0) & (results['y_pred'] == 1),
    (results['y_true'] == 1) & (results['y_pred'] == 0)]

labels = ['TP', 'TN', 'FP', 'FN']
results['class'] = np.select(conditions, labels, default = 'None')

# Save result ot a parquet file (load easily with pd.read_parquet())
results.to_parquet('RF_Final.parquet', index = False)


# In[87]:


plot_phase_space(results, 'RF_optimized')


# In[88]:


plot_purity_and_completeness(y, y_pred, results, 'RF_optimized')


# In[89]:


plot_3d_cm_per_cluster(dff, cluster_catalogue, results, 'RF_optimized')


# In[90]:


plot_3d_cm_stacked(dff, cluster_catalogue, results, 'RF_optimized')


# In[91]:


plot_3d_cm_obs_coords(dff, cluster_catalogue, results, 'RF_optimized')


# ### Oversampling - SMOTE

# In[81]:


# SMOTE performs oversampling of the minority class, here we oversample until reaching a 20% of the majority class

# 'Fake' SMOTE, just for plot
sm = SMOTE(random_state = 52, sampling_strategy = 0.2)
X_res, y_res = sm.fit_resample(X, y)

print(f'Oversampling: {len(X)} -> {len(X_res)}')
print(f'New synthetic data points: {len(X_res) - len(X)}')

# Identify synthetic indexes
synthetic_idx = np.arange(len(X), len(X_res))

X_res_df = pd.DataFrame(X_res, columns = X.columns)
X_res_df['synthetic'] = 0
X_res_df.loc[synthetic_idx, 'synthetic'] = 1
X_res_df['true_member'] = y_res


# In[82]:


plt.figure(figsize = (8, 6))

plt.scatter(dff.loc[nonmembers, 'R_norm'], dff.loc[nonmembers, 'V_norm'], s = 2, color = 'gray', alpha = 0.3, label = 'Interlopers', zorder = 1, rasterized = True)
plt.scatter(dff.loc[members, 'R_norm'], dff.loc[members, 'V_norm'], s = 2, color = 'crimson', alpha = 0.5, label = 'Members', zorder = 2, rasterized = True)
plt.scatter(X_res_df.loc[X_res_df['synthetic'] == 1, 'R_norm'], X_res_df.loc[X_res_df['synthetic'] == 1, 'V_norm'], s = 1, color = 'dodgerblue', alpha = 0.5, 
                   label = 'SMOTE Synthetic', zorder = 3, rasterized = True)

plt.xlabel(r'$r_{proj} / r_{200}$', fontsize = 17)
plt.ylabel(r'$\Delta v / \sigma_{200}$', fontsize = 17)
plt.xlim(0, 5)
plt.ylim(-4, 4)
leg = plt.legend(fontsize = 12, loc = 'upper right', fancybox = True, framealpha = 0.8)
for handle in leg.legend_handles: 
    handle._sizes = [25]
    handle.set_alpha(0.8) 

plt.tight_layout()
plt.savefig('PPSwSMOTE.png', dpi = 250)
plt.show()


# In[41]:


#  Pipeline takes a list of tuples, each of them is a step with its name and the object:    steps = [('name' , object), ()...]

pipe = Pipeline([('smote', SMOTE(random_state = 52, sampling_strategy = 0.2)),
                            ('rf', RandomForestClassifier(random_state = 14, class_weight = 'balanced'))])

cv_results = cross_validate(pipe, X, y, groups = groups, cv = cv, scoring = ['accuracy', 'precision', 'recall', 'f1'], return_train_score = True, n_jobs = -1)

print('CV mean ± std:')
for m in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
    print(f'{m}: {cv_results[m].mean():.3f} ± {cv_results[m].std():.3f}')

y_prob = cross_val_predict(pipe, X, y, groups = groups, cv = cv, method = 'predict_proba', n_jobs = -1)[:,1]


# In[42]:


y_pred = (y_prob >= threshold).astype(int)
print('Classification report (CV predictions):')
print(classification_report(y, y_pred, digits = 3))

cm = confusion_matrix(y, y_pred)


# In[43]:


plot_confusion_matrix(cm, ['Interloper', 'Member'], model_name = 'SMOTE_nogrid', cmap = 'Blues')


# In[44]:


TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0] , cm[1][1]
purity = TP / (TP + FP)
completeness = TP / (TP + FN)

print('Purity/Precision : ', precision_score(y, y_pred))
print('Completeness/Recall: ', recall_score(y, y_pred))
print('F1 : ', f1_score(y, y_pred))
print('ROC AUC: ', roc_auc_score(y, y_prob))
print('AP (PR AUC): ', average_precision_score(y, y_prob))

plot_roc_curve(y, y_prob, threshold, model_name = 'SMOTE_nogrid')


# In[45]:


plot_learning_curve(pipe, X, y,  groups = groups, cv = cv, scoring = 'f1', model_name = 'SMOTE_nogrid')


# In[46]:


plot_feature_importances(pipe.named_steps['rf'], X, y, model_name = 'SMOTE_nogrid')


# In[47]:


joblib.dump(pipe, 'SMOTE_nogrid.pkl')

results = X.copy()
results['y_true'] = y.values
results['y_pred'] = y_pred
results['y_prob'] = y_prob

results = results.join(dff[['id', 'mock_id']], how = 'left')

conditions = [
    (results['y_true'] == 1) & (results['y_pred'] == 1),
    (results['y_true'] == 0) & (results['y_pred'] == 0),
    (results['y_true'] == 0) & (results['y_pred'] == 1),
    (results['y_true'] == 1) & (results['y_pred'] == 0)]

labels = ['TP', 'TN', 'FP', 'FN']
results['class'] = np.select(conditions, labels, default = 'None')

results.to_parquet('SMOTE_nogrid.parquet', index = False)


# In[48]:


plot_phase_space(results, 'SMOTE_nogrid')


# In[49]:


plot_purity_and_completeness(y, y_pred, results, 'SMOTE_nogrid')


# In[50]:


plot_3d_cm_per_cluster(dff, cluster_catalogue, results, 'SMOTE_nogrid')


# In[51]:


plot_3d_cm_stacked(dff, cluster_catalogue, results, 'SMOTE_nogrid')


# ### SMOTE with hyperparameters

# In[178]:


# Option 1: Run gridsearch

pipe = Pipeline([('smote', SMOTE(random_state = 52, sampling_strategy = 0.2)), 
                            ('rf', RandomForestClassifier(random_state = 14, class_weight = balanced))]) 

# Hyperparameter grid
parameters = { 
              'rf__n_estimators': [100, 200, 300], 
              'rf__min_impurity_decrease': [0.0, 0.01, 0.1], 
              'rf__max_depth': [5, 10, 20], 
              'rf__min_samples_split': [10, 20, 50], 
              'rf__max_leaf_nodes': [20, 50, 80], 
              'rf__max_features': [None], 
              'rf__class_weight': ['balanced'] }

model = GridSearchCV(estimator = pipe, param_grid = parameters, scoring = 'f1', cv = cv, n_jobs = -1, verbose = 2, return_train_score = True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "model.fit(X, y, groups = groups)\n \nprint('Best score:', model.best_score_)\nprint('Best parameters:', model.best_params_)\n\nRF_pipeline = model.best_estimator_\ny_prob = cross_val_predict(RF_pipeline, X, y, groups = groups, cv = cv, method = 'predict_proba', n_jobs = -1)[:,1]\n")


# In[ ]:


# Option 2: No gridsearch, use hypereparameters from previous run

RF_pipeline =  joblib.load(d+'RF\\SMOTE_gridsearch.pkl')
rf_step = RF_pipeline.named_steps['rf']
best_params = rf_step.get_params()

RF_pipeline = Pipeline([('smote', SMOTE(random_state = 52, sampling_strategy = 0.2)),
                                        ('rf', RandomForestClassifier(**best_params)) ])

cv_results = cross_validate(RF_pipeline, X, y, groups = groups, cv = cv, scoring = ['accuracy', 'precision', 'recall', 'f1'], return_train_score = True, n_jobs = -1)

print('CV mean ± std:')
for m in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
    print(f' {m}: {cv_results[m].mean():.3f} ± {cv_results[m].std():.3f}')

y_prob = cross_val_predict(RF_pipeline, X, y, groups = groups, cv = cv, method = 'predict_proba', n_jobs = -1)[:,1]


# In[ ]:


y_pred = (y_prob >= threshold).astype(int)

print('Classification Report:')
print(classification_report(y, y_pred, digits = 3))

cm = confusion_matrix(y,y_pred) 


# In[ ]:


plot_confusion_matrix(cm, ['Interloper', 'Member'], model_name = 'SMOTE_gridsearch', cmap = 'Blues')


# In[ ]:


TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0] , cm[1][1]
purity = TP / (TP + FP)
completeness = TP / (TP + FN)

print('Purity/Precision : ', precision_score(y, y_pred))
print('Completeness/Recall: ', recall_score(y, y_pred))
print('F1 : ', f1_score(y, y_pred))
print('ROC AUC: ', roc_auc_score(y, y_prob))
print('AP (PR AUC): ', average_precision_score(y, y_prob))

plot_roc_curve(y, y_prob, threshold, model_name = 'SMOTE_gridsearch')


# In[ ]:


plot_learning_curve(RF_pipeline, X, y, groups = groups, cv = cv, scoring = 'f1', model_name = 'SMOTE_gridsearch')


# In[ ]:


plot_feature_importances(RF_pipeline.named_steps['rf'], X, y, model_name = 'SMOTE_gridsearch')


# In[ ]:


joblib.dump(RF_pipeline, 'SMOTE_gridsearch.pkl') 

results = X.copy()
results['y_true'] = y.values
results['y_pred'] = y_pred
results['y_prob'] = y_prob

results = results.join(dff[['id', 'mock_id']], how = 'left')

conditions = [
    (results['y_true'] == 1) & (results['y_pred'] == 1),
    (results['y_true'] == 0) & (results['y_pred'] == 0),
    (results['y_true'] == 0) & (results['y_pred'] == 1),
    (results['y_true'] == 1) & (results['y_pred'] == 0)]

labels = ['TP', 'TN', 'FP', 'FN']
results['class'] = np.select(conditions, labels, default = 'None')

results.to_parquet('SMOTE_gridsearch.parquet', index = False)


# In[ ]:


plot_phase_space(results, 'SMOTE_gridsearch')


# In[ ]:


plot_purity_and_completeness(y, y_pred, results, 'SMOTE_gridsearch')


# In[ ]:


plot_3d_cm_per_cluster(dff, cluster_catalogue, results, 'SMOTE_gridsearch')


# In[ ]:


plot_3d_cm_stacked(dff, cluster_catalogue, results, 'SMOTE_gridsearch')


# ### Undersampling - RUS

# In[172]:


# RandomUnderSampler, randomly undersamples the majority class, here we undersample until the minority class reaches 20% of the majority class

pipe = Pipeline([('undersample', RandomUnderSampler(random_state = 52, sampling_strategy = 0.2)),
                            ('rf', RandomForestClassifier(random_state = 14, class_weight = 'balanced'))])

cv_results = cross_validate(pipe, X, y, groups = groups, cv = cv, scoring = ['accuracy', 'precision', 'recall', 'f1'], return_train_score = True, n_jobs = -1)

print('CV mean ± std:')
for m in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
    print(f'{m}: {cv_results[m].mean():.3f} ± {cv_results[m].std():.3f}')

y_prob = cross_val_predict(pipe, X, y, groups = groups, cv = cv, method = 'predict_proba', n_jobs = -1)[:,1]


# In[173]:


y_pred = (y_prob >= threshold).astype(int)
print('Classification report (CV predictions):')
print(classification_report(y, y_pred, digits = 3))

cm = confusion_matrix(y, y_pred)


# In[174]:


plot_confusion_matrix(cm, ['Interloper', 'Member'], model_name = 'RF_RUS', cmap = 'Blues')


# In[175]:


TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0] , cm[1][1]
purity = TP / (TP + FP)
completeness = TP / (TP + FN)

print('Purity/Precision : ', precision_score(y, y_pred))
print('Completeness/Recall: ', recall_score(y, y_pred))
print('F1 : ', f1_score(y, y_pred))
print('ROC AUC: ', roc_auc_score(y, y_prob))
print('AP (PR AUC): ', average_precision_score(y, y_prob))

plot_roc_curve(y, y_prob, threshold, model_name = 'RF_RUS')


# In[182]:


plot_learning_curve(pipe, X, y, groups = groups, cv = cv, scoring = 'f1', model_name = 'RF_RUS')


# In[176]:


plot_feature_importances(pipe.named_steps['rf'], X, y, model_name = 'RF_RUS')


# In[183]:


joblib.dump(pipe, 'RUS.pkl') 

results = X.copy()
results['y_true'] = y.values
results['y_pred'] = y_pred
results['y_prob'] = y_prob

results = results.join(dff[['id', 'mock_id']], how = 'left')

conditions = [
    (results['y_true'] == 1) & (results['y_pred'] == 1),
    (results['y_true'] == 0) & (results['y_pred'] == 0),
    (results['y_true'] == 0) & (results['y_pred'] == 1),
    (results['y_true'] == 1) & (results['y_pred'] == 0)]

labels = ['TP', 'TN', 'FP', 'FN']
results['class'] = np.select(conditions, labels, default = 'None')

results.to_parquet('RUS.parquet', index = False)


# In[178]:


plot_phase_space(results, 'RF_RUS')


# In[179]:


plot_purity_and_completeness(y, y_pred, results, 'RF_RUS')   


# In[180]:


plot_3d_cm_per_cluster(dff, cluster_catalogue, results, 'RF_RUS')


# In[181]:


plot_3d_cm_stacked(dff, cluster_catalogue, results, 'RF_RUS')


# ### Undersampling with  hyperparameters

# In[ ]:


# Option 1: Run gridsearch

pipe = Pipeline([('undersample', RandomUnderSampler(random_state = 52, sampling_strategy = 0.2)),     
                            ('rf', RandomForestClassifier(random_state = 14, n_jobs = -1))])

parameters = {
    'rf__n_estimators': [100, 200, 300],
    'rf__min_impurity_decrease': [0.0, 0.01],
    'rf__max_depth': [5, 10, 20],
    'rf__min_samples_split': [10, 20, 50],
    'rf__max_leaf_nodes': [20, 50, 80],
    'rf__max_features': [None, 'sqrt'],
    'rf__class_weight': ['balanced']}

model = GridSearchCV(estimator = pipe, param_grid = parameters, scoring = 'f1', cv = cv, n_jobs = -1, verbose = 2, return_train_score = True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "model.fit(X, y, groups = groups)\n\nprint('Best score:', model.best_score_)\nprint('Best parameters:', model.best_params_)\n\nRF_pipeline = model.best_estimator_\ny_prob = cross_val_predict(RF_pipeline, X, y, groups = groups, cv = cv, method = 'predict_proba', n_jobs = -1)[:,1]  \n")


# In[67]:


# Option 2: Use parameters from preious run

RF_pipeline =  joblib.load(d+'RF\\RFundersampling.pkl')
rf_step = RF_pipeline.named_steps['rf']
best_params = rf_step.get_params()

RF_pipeline = Pipeline([('undersample', RandomUnderSampler(random_state=52, sampling_strategy=0.2)),  
                                        ('rf', RandomForestClassifier(**best_params)) ])

cv_results = cross_validate(RF_pipeline, X, y, groups = groups, cv = cv, scoring = ['accuracy', 'precision', 'recall', 'f1'], return_train_score = True, n_jobs = -1)

print('CV mean ± std:')
for m in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
    print(f' {m}: {cv_results[m].mean():.3f} ± {cv_results[m].std():.3f}')

y_prob = cross_val_predict(RF_pipeline, X, y, groups = groups, cv = cv, method = 'predict_proba', n_jobs = -1)[:,1] 


# In[68]:


y_pred = (y_prob >= threshold).astype(int)

print("Classification Report:")
print(classification_report(y, y_pred, digits = 3))

cm = confusion_matrix(y,y_pred) 


# In[69]:


plot_confusion_matrix(cm, ['Interloper', 'Member'], model_name = 'RUS_gridsearch', cmap = 'Blues')


# In[70]:


TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0] , cm[1][1]
purity = TP / (TP + FP)
completeness = TP / (TP + FN)

print('Purity/Precision : ', precision_score(y, y_pred))
print('Completeness/Recall: ', recall_score(y, y_pred))
print('F1 : ', f1_score(y, y_pred))
print('ROC AUC: ', roc_auc_score(y, y_prob))
print('AP (PR AUC): ', average_precision_score(y, y_prob))

plot_roc_curve(y, y_prob, threshold, model_name = 'RUS_gridsearch')


# In[54]:


plot_learning_curve(RF_pipeline, X, y, groups = groups, cv = cv, scoring = 'f1', model_name = 'RUS_gridsearch')


# In[71]:


plot_feature_importances(RF_pipeline.named_steps['rf'], X, y, model_name = 'RUS_gridsearch')


# In[56]:


joblib.dump(RF_pipeline, 'RUS_gridsearch.pkl')  

results = X.copy()
results['y_true'] = y.values
results['y_pred'] = y_pred
results['y_prob'] = y_prob

results = results.join(dff[['id', 'mock_id']], how = 'left')

conditions = [
    (results['y_true'] == 1) & (results['y_pred'] == 1),
    (results['y_true'] == 0) & (results['y_pred'] == 0),
    (results['y_true'] == 0) & (results['y_pred'] == 1),
    (results['y_true'] == 1) & (results['y_pred'] == 0)]

labels = ['TP', 'TN', 'FP', 'FN']
results['class'] = np.select(conditions, labels, default = 'None')

results.to_parquet('RUS_gridsearch.parquet', index = False)


# In[57]:


plot_phase_space(results, 'RUS_gridsearch')


# In[114]:


plot_purity_and_completeness(y, y_pred, results, 'RUS_gridsearch')


# In[59]:


plot_3d_cm_per_cluster(dff, cluster_catalogue, results, 'RUS_gridsearch')


# In[60]:


plot_3d_cm_stacked(dff, cluster_catalogue, results, 'RUS_gridsearch')


# ## Extra Plots

# In[22]:


# PPS stacked for all clusters + 3D view of a single cluster, colored with confusion matrix

target_cluster_id = 0 
model_name = 'RF_Optimized' 
results = pd.read_parquet(d+'RF\\RF_Final.parquet')

df_working = dff.copy()
df_working['class_pred'] = results['class'].values

cluster_info = cluster_catalogue[cluster_catalogue['Cluster'] == target_cluster_id].iloc[0]
x_cent, y_cent, z_cent = cluster_info['X_cent'], cluster_info['Y_cent'], cluster_info['Z_cent']
R200 = cluster_info['R_200_mpc']

df_mock = df_working[df_working['mock_id'] == target_cluster_id].copy()

df_mock['x0'] = df_mock['x'] - x_cent
df_mock['y0'] = df_mock['y'] - y_cent
df_mock['z0'] = df_mock['z'] - z_cent

lim = 5 * R200 * 1.3
mask_in_box = ((df_mock['x0'].abs() <= lim) & 
                            (df_mock['y0'].abs() <= lim) & 
                            (df_mock['z0'].abs() <= lim))

df_plot = df_mock[mask_in_box]

# PLOT
fig = plt.figure(figsize = (20, 9))
gs = gridspec.GridSpec(1, 2, width_ratios = [1.2, 1], wspace = 0.001)

# =============================================================================
# LEFT PANEL: PROJECTED PHASE SPACE 
# =============================================================================
ax1 = fig.add_subplot(gs[0])

mask_TP = results['class'] == 'TP'
mask_FP = results['class'] == 'FP'
mask_TN = results['class'] == 'TN'
mask_FN = results['class'] == 'FN'

ax1.scatter(results.loc[mask_TN, 'R_norm'], results.loc[mask_TN, 'V_norm'], s = 2, alpha = 0.3, color = c_tn, label = 'TN', zorder = 1)
ax1.scatter(results.loc[mask_TP, 'R_norm'], results.loc[mask_TP, 'V_norm'], s = 5, alpha = 0.8, color = c_tp, label = 'TP', zorder = 2)
ax1.scatter(results.loc[mask_FN, 'R_norm'], results.loc[mask_FN, 'V_norm'], s = 4, alpha = 0.5, color = c_fn, label = 'FN', zorder = 3)
ax1.scatter(results.loc[mask_FP, 'R_norm'], results.loc[mask_FP, 'V_norm'], s = 4, alpha = 0.5, color = c_fp, label = 'FP', zorder = 4)    

# Contours
kde_args = {'ax': ax1, 'levels': 5, 'thresh': 0.3, 'linewidths': 2.5}
sns.kdeplot(x = results.loc[mask_TP, 'R_norm'], y = results.loc[mask_TP, 'V_norm'], color = 'darkgreen', zorder = 7, **kde_args)
sns.kdeplot(x = results.loc[mask_FN, 'R_norm'], y = results.loc[mask_FN, 'V_norm'], color = 'darkred', zorder = 5, **kde_args)
sns.kdeplot(x = results.loc[mask_FP, 'R_norm'], y = results.loc[mask_FP, 'V_norm'], color = 'darkorange', zorder = 6, **kde_args)

ax1.set_xlabel(r'$r_{proj} / r_{200}$', fontsize = 18)
ax1.set_ylabel(r'$\Delta v / \sigma_{200}$', fontsize = 18)
ax1.set_xlim(0, 5)
ax1.set_ylim(-4, 4)

legend_elements = [Line2D([0], [0], color = c_tn, marker = 'o', linestyle = 'None', markersize = 7, label = 'TN', alpha = 0.5),
                                  Line2D([0], [0], color = c_tp, marker = 'o', markersize = 7, label = 'TP', linestyle = '-'),
                                  Line2D([0], [0], color = c_fn, marker = 'o', markersize = 7, label = 'FN', linestyle = '-'),
                                  Line2D([0], [0], color = c_fp, marker = 'o', markersize = 7, label = 'FP', linestyle = '-')]
ax1.legend(handles = legend_elements, fontsize = 14, loc = 1, framealpha = 0.9)

# =============================================================================
# RIGHT PANEL: 3D CLUSTER
# =============================================================================
ax2 = fig.add_subplot(gs[1], projection='3d')

tp = df_plot[df_plot['class_pred'] == 'TP']
fp = df_plot[df_plot['class_pred'] == 'FP']
fn = df_plot[df_plot['class_pred'] == 'FN'] 
tn = df_plot[df_plot['class_pred'] == 'TN'] 

ax2.scatter(tn['x0'], tn['y0'], tn['z0'], c = c_tn, s = 5, alpha = 0.4, label = f'TN: {len(tn)}', zorder = 1)
ax2.scatter(tp['x0'], tp['y0'], tp['z0'], c = c_tp, s = 10, alpha = 0.6, label = f'TP: {len(tp)}', zorder = 2)
ax2.scatter(fn['x0'], fn['y0'], fn['z0'], c = c_fn, s = 15, marker = '^', alpha = 0.8, label = f'FN: {len(fn)}', zorder = 11)
ax2.scatter(fp['x0'], fp['y0'], fp['z0'], c = c_fp, s = 15, marker = 'x', alpha = 0.9, label = f'FP: {len(fp)}', zorder = 12)

# 5R200 Sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
r_sphere = 5 * R200
x_sph = r_sphere * np.cos(u) * np.sin(v)
y_sph = r_sphere * np.sin(u) * np.sin(v)
z_sph = r_sphere * np.cos(v)
ax2.plot_wireframe(x_sph, y_sph, z_sph, color = 'black', alpha = 0.1, linewidth = 0.5)

ax2.set_xlabel('x [Mpc]', fontsize = 14)
ax2.set_ylabel('y [Mpc]', fontsize = 14)
ax2.set_zlabel('z [Mpc]', fontsize = 14)
ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.set_zlim(-lim, lim)

ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor('w')
ax2.yaxis.pane.set_edgecolor('w')
ax2.zaxis.pane.set_edgecolor('w')
grid_style = {'color': 'gray', 'linestyle': ':', 'linewidth': 0.5, 'alpha': 0.3}
ax2.xaxis._axinfo['grid'].update(grid_style)
ax2.yaxis._axinfo['grid'].update(grid_style)
ax2.zaxis._axinfo['grid'].update(grid_style)

ax2.set_title(f'Mock A85 3D Distribution', fontsize = 16)
ax2.set_box_aspect([1,1,1])
ax2.legend(loc = 'upper right', fontsize = 12)
# plt.tight_layout()
plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
plt.savefig(f'PPS_and_3DCluster{target_cluster_id}.pdf', dpi = 300, bbox_inches = 'tight')
plt.show()


# In[41]:


# PPS stacked for all clusters + Observational 3D space (RA, DEC, z) view of a single cluster, colored with confusion matrix

results =  pd.read_parquet(d+'RF\\RF_gridsearch.parquet')
target_cluster_id = 0 
model_name = 'RF_Optimized' 

row = cluster_catalogue[cluster_catalogue['Cluster'] == target_cluster_id].iloc[0]   # cluster 0 info
RA_cl, DEC_cl, z_cl = row['RA'], row['Dec'], row['redshift']
R200_mpc = row['R_200_mpc']

df_working = dff.copy()
df_working['class_pred'] = results['class'].values

fig = plt.figure(figsize = (18, 8))
gs = gridspec.GridSpec(1, 2, width_ratios = [1.12, 1], wspace = -0.12)

# =============================================================================
# LEFT PANEL: PROJECTED PHASE SPACE
# =============================================================================
ax1 = fig.add_subplot(gs[0])

mask_TP = results['class'] == 'TP'
mask_FP = results['class'] == 'FP'
mask_TN = results['class'] == 'TN'
mask_FN = results['class'] == 'FN'

ax1.scatter(results.loc[mask_TN, 'R_norm'], results.loc[mask_TN, 'V_norm'], s = 2, alpha = 0.3, color = c_tn, label = 'TN', zorder = 1)
ax1.scatter(results.loc[mask_TP, 'R_norm'], results.loc[mask_TP, 'V_norm'], s = 5, alpha = 0.8, color = c_tp, label = 'TP', zorder = 2)
ax1.scatter(results.loc[mask_FN, 'R_norm'], results.loc[mask_FN, 'V_norm'], s = 4, alpha = 0.5, color = c_fn, label = 'FN', zorder = 3)
ax1.scatter(results.loc[mask_FP, 'R_norm'], results.loc[mask_FP, 'V_norm'], s = 4, alpha = 0.5, color = c_fp, label = 'FP', zorder = 4)    

# Contours
kde_args = {'ax': ax1, 'levels': 5, 'thresh': 0.3, 'linewidths': 2.5}
sns.kdeplot(x=results.loc[mask_TP, 'R_norm'], y = results.loc[mask_TP, 'V_norm'], color = 'darkgreen', zorder = 7, **kde_args)
sns.kdeplot(x=results.loc[mask_FN, 'R_norm'], y = results.loc[mask_FN, 'V_norm'], color = 'darkred', zorder = 5, **kde_args)
sns.kdeplot(x=results.loc[mask_FP, 'R_norm'], y = results.loc[mask_FP, 'V_norm'], color = 'darkorange', zorder = 6, **kde_args)

ax1.set_xlabel(r'$r_{proj} / r_{200}$', fontsize = 25)
ax1.set_ylabel(r'$\Delta v / \sigma_{200}$', fontsize = 25)
ax1.set_xlim(0, 5)
ax1.set_ylim(-4, 4)
ax1.tick_params(labelsize=13)

legend_elements = [Line2D([0], [0], color = c_tn, marker = 'o', linestyle = 'None', markersize = 7, label = 'TN', alpha = 0.5),
                                  Line2D([0], [0], color = c_tp, marker = 'o', markersize = 7, label = 'TP', linestyle = '-'),
                                  Line2D([0], [0], color = c_fn, marker = 'o', markersize = 7, label = 'FN', linestyle = '-'),
                                  Line2D([0], [0], color = c_fp, marker = 'o', markersize = 7, label = 'FP', linestyle = '-')]
ax1.legend(handles = legend_elements, fontsize = 15, loc = 1, framealpha = 0.9)

# =============================================================================
# RIGHT PANEL: 3D OBSERVATIONAL SPACE 
# =============================================================================
ax2 = fig.add_subplot(gs[1], projection = '3d')

df_mock = df_working[df_working['mock_id'] == target_cluster_id].copy()

# coords
df_mock['dra'] = df_mock['RA'] - RA_cl
df_mock['ddec'] = df_mock['DEC'] - DEC_cl
df_mock['dz'] = df_mock['redshift_S_1'] - z_cl

da = cosmo.angular_diameter_distance(z_cl).value
r5_deg = np.rad2deg((5 * R200_mpc) / da)   # 5r200 in deg

hz = cosmo.H(z_cl).value
r5_dz = (hz * (5 * R200_mpc)) / c_km_s      # 5r200 in z

# Box limits, selected to visualize FOG
lim_ang = r5_deg * 3.0   
lim_z = r5_dz * 4.0

mask_in_box = ((df_mock['dra'].abs() <= lim_ang) & 
                            (df_mock['ddec'].abs() <= lim_ang) & 
                            (df_mock['dz'].abs() <= lim_z))

df_plot = df_mock[mask_in_box]

tp = df_plot[df_plot['class_pred'] == 'TP']
fp = df_plot[df_plot['class_pred'] == 'FP']
fn = df_plot[df_plot['class_pred'] == 'FN'] 
tn = df_plot[df_plot['class_pred'] == 'TN']

ax2.scatter(tn['dra'], tn['ddec'], tn['dz'], c = c_tn, s = 5, alpha = 0.3, label = f'TN: {len(tn)}', zorder = 1)
ax2.scatter(tp['dra'], tp['ddec'], tp['dz'], c = c_tp, s = 8, alpha = 0.6, label = f'TP: {len(tp)}', zorder = 2)
ax2.scatter(fn['dra'], fn['ddec'], fn['dz'], c = c_fn, s = 12, marker = '^', alpha = 0.8, label = f'FN: {len(fn)}', zorder = 11)
ax2.scatter(fp['dra'], fp['ddec'], fp['dz'], c = c_fp, s = 12, marker = 'x', alpha = 0.9, label = f'FP: {len(fp)}', zorder = 12)
        
# 5R200 Sphere
phi, theta = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sph = r5_deg * np.cos(phi) * np.sin(theta)
y_sph = r5_deg * np.sin(phi) * np.sin(theta)
z_sph = r5_dz * np.cos(theta)
ax2.plot_wireframe(x_sph, y_sph, z_sph, color = 'k', alpha = 0.15, linewidth = 0.5)

ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor('w')
ax2.yaxis.pane.set_edgecolor('w')
ax2.zaxis.pane.set_edgecolor('w')
grid_style = {'color': 'gray', 'linestyle': ':', 'linewidth': 0.5, 'alpha': 0.3}
ax2.xaxis._axinfo["grid"].update(grid_style)   
ax2.yaxis._axinfo["grid"].update(grid_style)
ax2.zaxis._axinfo["grid"].update(grid_style)

ax1.set_facecolor('none') 
ax2.set_facecolor('none') 
ax1.patch.set_alpha(0.0)
ax2.patch.set_alpha(0.0)

ax2.set_xlabel(r'$\Delta  \alpha$ [deg]', fontsize = 18)
ax2.set_ylabel(r'$\Delta \delta$ [deg]', fontsize = 18)
ax2.set_zlabel(r'$\Delta z$', labelpad= 18, fontsize = 18)
ax2.tick_params(axis = 'z', pad = 10)
ax2.set_xlim(-lim_ang, lim_ang)
ax2.set_ylim(-lim_ang, lim_ang)
ax2.set_zlim(-lim_z, lim_z)
ax2.set_title(f'Mock A85 Observational Space', fontsize = 18)

leg = ax2.legend(loc = (0.1, 0.82), fontsize = 13)
for lh in leg.legend_handles:
    lh.set_sizes([40])
    lh.set_alpha(1)

ax2.set_box_aspect([1, 1, 1.333])  # since limits are different, we need to adjust so that the sphere doesn't stretch
ax2.view_init(elev = 10, azim = -60)

pos2 = ax2.get_position()
y_shift = 0.1  
ax2.set_position([pos2.x0, pos2.y0 - y_shift, pos2.width, pos2.height])

plt.subplots_adjust(left = 0.05, right = 1.05, top = 0.95, bottom = 0.09)
plt.savefig(f'PPS_and_Obs3D_Cluster{target_cluster_id}.pdf', dpi = 300)
plt.show()

