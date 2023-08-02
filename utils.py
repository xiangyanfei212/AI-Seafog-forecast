import os
import gc
import time
import itertools
import numpy as np
import pandas as pd
import pingouin as pg
import networkx as nx
import seaborn as sns
import lightgbm as lgb
from random import randint
from collections import Counter
from datetime import date,datetime

# %% matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# %% cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# %% imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# %% scipy
from scipy import special
from scipy import optimize
import scipy.stats as stats
from scipy.misc import derivative

# %% sklearn
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve,PrecisionRecallDisplay

name_dict = {'PRES_GDS3_SFC':   'Pressure',
             'PRMSL_GDS3_MSL':  'Pressure reduced to MSL',
             'HGT_GDS3_SFC':    'Geopotential height (SFC)',
             'HGT_GDS3_0DEG':   'Geopotential height (0DEG)',
             'HGT_GDS3_HTFL':   'Geopotential height (HTFL)',
             'HGT_GDS3_CEIL':   'Geopotential height (CEIL)',
             'TMP_GDS3_SFC':    'Temperature (SFC)',
             'TEMP_WTEMP':      'Temperature - sea Temperature',
             'TMP_GDS3_HTGL':   'Temperature (HTGL)',
             'DPT_GDS3_HTGL':   'Dew point temperature (HTGL)',
             'PLI_GDS3_SPDY':   'Parcel lifted index (to 500 hPa)',
             'U_GRD_GDS3_HTGL': 'u-component of wind (HTGL)',
             'U_GRD_GDS3_HYBL': 'u-component of wind (HYBL)',
             'U_GRD_GDS3_SPDY': 'u-component of wind (SPDY)',
             'V_GRD_GDS3_HTGL': 'v-component of wind (HTGL)',
             'V_GRD_GDS3_HYBL': 'v-component of wind (HYBL)',
             'V_GRD_GDS3_SPDY': 'v-component of wind (SPDY)',
             'SPF_H_GDS3_HTGL': 'Specific humidity (HTGL)',
             'SPF_H_GDS3_HYBL': 'Specific humidity (HYBL)',
             'SPF_H_GDS3_SPDY': 'Specific humidity (SPDY)',
             'R_H_GDS3_HTGL':   'Relative humidity (HTGL)',
             'R_H_GDS3_HYBL':   'Relative humidity (HYBL)',
             'P_WAT_GDS3_EATM': 'Precipitable water',
             'L_CDC_GDS3_LCY':  'Low level cloud cover',
             'M_CDC_GDS3_MCY':  'Mid level cloud cover',
             'H_CDC_GDS3_HCY':  'High level cloud cover',
             'T_CDC_GDS3_EATM': 'Total cloud cover',
             'WTMP_GDS3_SFC':   'Water temperature',
             'SFC_R_GDS3_SFC':  'Surface roughness',
             'MSLET_GDS3_MSL':  'Mean sea level pressure (ETA model)',
             'LFT_X_GDS3_ISBY': 'Surface lifted index',
             'POP_GDS3_SFC':    'Probability of precipitation',
             'TCOLW_GDS3_EATM': 'Total column-integrated cloud water'}


time_dict = {'(t0)': '(t-6)',
             '(t1)': '(t-5)',
             '(t2)': '(t-4)',
             '(t3)': '(t-3)',
             '(t4)': '(t-2)',
             '(t5)': '(t-1)',
             '(t6)': '(t-0)'}


def get_all_dataset(sample_dir, years_range, spot_hour):
    
    sample = pd.DataFrame()
    for year in years_range:
        print(year)
       
        sample_file = os.path.join(sample_dir, f'{year}_spot{spot_hour}_wrf61_prev18.csv')
   
        print(f'Loading from {sample_file}')
        sample_df = pd.read_csv(sample_file, index_col=0)
        
        sample = pd.concat([sample, sample_df], axis=0)
        
    sample.drop_duplicates(inplace=True)
    
    return sample

def get_area_border(area_name):

    HB_loc = [42, 33, 118, 126] # lat_N,lat_S,lon_W,lon_E
    CJ_loc = [33, 28, 119, 126]
    Zhu_loc = [24, 18, 110, 118]

    if area_name == 'Huang_Bo':
        [lat_N, lat_S, lon_W, lon_E] = HB_loc
    elif area_name == 'CJ':
        [lat_N, lat_S, lon_W, lon_E] = CJ_loc
    elif area_name == 'Zhu':
        [lat_N, lat_S, lon_W, lon_E] = Zhu_loc

    return [lat_N, lat_S, lon_W, lon_E]


def show_station_loc(area_name, df, id_col, lat_col, lon_col):

    [lat_N, lat_S, lon_W, lon_E] = get_area_border(area_name)

    plt.figure(figsize=(12, 8))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_xticks(range(-180, 180, 1), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 90, 1), crs=ccrs.PlateCarree())

    # extent = [100, 135, 18, 54]
    extent = [lon_W, lon_E, lat_S, lat_N]
    ax.set_extent(extent)
    ax.stock_img()  # 将参考底图图像添加到地图,如果没有这条命令，底图是没有背景色的

    # 画经纬度网格
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.9, color='k', alpha=0.3, linestyle='--')
    gl.top_labels = False  # 关闭顶端的经纬度标签
    gl.right_labels = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mticker.FixedLocator(np.arange(lon_W, lon_E + 1, 1)) # 手动设置x轴刻度
    gl.ylocator = mticker.FixedLocator(np.arange(lat_S, lat_N + 1, 1))

    loc_df = df.drop_duplicates(subset=[id_col, lon_col, lat_col])
    
    size = 7
    ax.scatter(loc_df[lon_col][:], loc_df[lat_col][:], s=size, c='r', transform=ccrs.PlateCarree())

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    for index, row in loc_df.iterrows():
        ax.annotate(int(row[id_col]), (row[lon_col], row[lat_col]))

    plt.show()


def significance_test(sam, co_var, tgt_var, sig_thres):

    cause_var = [tgt_var] + co_var
    result_var = [tgt_var] + co_var
    sig_df = pd.DataFrame(columns=result_var, index=cause_var)

    for cv in cause_var:
        for rv in result_var:
            
            # p值越小，表示相关系数越显著，一般p值在500个样本以上时有较高的可靠性。
            r,p = stats.pearsonr(sam[cv], sam[rv])
 
            sig_df.at[cv, rv] = p
    
    return sig_df

def partial_corr(sample, cov_variables, tgt_var, sig_threshold):
    
    vis_partial_corr = pd.DataFrame()
    for cause in cov_variables:
        covar = cov_variables.copy(); covar.remove(cause)
    
        pc = pg.partial_corr(data=sample, x=cause, y=tgt_var, covar=covar)
        
        sub_partial_corr = pd.DataFrame({'cause': cause, 
                                         'r':pc['r'].values,
                                         'p':pc['p-val'].values
                                    })
        vis_partial_corr = pd.concat([vis_partial_corr, sub_partial_corr])
    return vis_partial_corr

    
def plot_partial_corr_bar(org_abs_par_corr, title, savepath):
    
    abs_par_corr = org_abs_par_corr.copy()
    plt.style.use(['science', 'no-latex'])
    
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 15), sharey=True)

    abs_par_corr = org_abs_par_corr.copy()
    abs_par_corr = abs_par_corr[abs_par_corr['abs_partial_corr'] != 0]
    sns.barplot(data=abs_par_corr, x='abs_partial_corr', y='variables', ax=axs, palette="Blues_d")

    plt.xlabel('Absolute Partial Correlation', fontsize=20)
    plt.ylabel('Variables', fontsize=20)

    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)

    plt.title(title, fontsize=25)
    plt.savefig(savepath, dpi=200)
    plt.show()
    
    
def draw_heatmap(org_corr_df, title:str, save_prefix:str, vmin, vmax):
    
    ## correlation
    corr_df = org_corr_df.copy()[['variables', 'r']]
    
    corr_df['variable'] = [i.split(' ')[0] for i in corr_df['variables']]
    for (wrf_name, long_name) in name_dict.items():
        corr_df.loc[corr_df.variable == wrf_name, 'variable'] = long_name
        
    corr_df['lagged_time'] = [i.split(' ')[1] for i in corr_df['variables']]
    for (org_time, new_time) in name_dict.items():
        corr_df.loc[corr_df.lagged_time == wrf_name, 'lagged_time'] = new_time
        
    corr_df.drop(columns=['variables'], inplace=True)

    corr_df = corr_df.groupby(['variable', 'lagged_time']).mean()
    corr_df = corr_df.unstack(level=0)
    corr_df.fillna(0, inplace=True)
    corr_df = corr_df.T

    col_order = ['(t0)', '(t1)', '(t2)', '(t3)', '(t4)', '(t5)', '(t6)']
    corr_df = corr_df[col_order]
    # print(corr_df)
    ##
    
    ## significance
    sig_df = org_corr_df.copy()[['variables', 'sig']]    
    sig_df['sig'] = sig_df['sig'].astype(float)
    sig_df['variable'] = [i.split(' ')[0] for i in sig_df['variables']]
    for (wrf_name, long_name) in name_dict.items():
        sig_df.loc[sig_df.variable == wrf_name, 'variable'] = long_name
        
    sig_df['lagged_time'] = [i.split(' ')[1] for i in sig_df['variables']]
    for (org_time, new_time) in name_dict.items():
        sig_df.loc[sig_df.lagged_time == wrf_name, 'lagged_time'] = new_time
        
    sig_df.drop(columns=['variables'], inplace=True)
    
    sig_df = sig_df.groupby(['variable', 'lagged_time']).mean()
    sig_df = sig_df.unstack(level=0)
    sig_df.fillna(0, inplace=True)
    sig_df = sig_df.T

    col_order = ['(t0)', '(t1)', '(t2)', '(t3)', '(t4)', '(t5)', '(t6)']
    sig_df = sig_df[col_order]
    # print(sig_df)
    ##

    variable_type = [ corr_df.index[t][1] for t in range(len(corr_df.index))]
    # print(variable_type)

    plt.style.use(['science', 'no-latex'])
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(8, 15))
    
    ax.tick_params(bottom=False,top=False,left=False,right=False)  #移除全部刻度线
    
    col_list = [v[1] for v in sig_df.index.tolist()]    
    row_list = sig_df.columns.tolist()
    widthx = 0.75
    widthy = 0.45
    # 显著性
    for ci,c in enumerate(col_list):
        for ri, r in enumerate(row_list):
            sig = sig_df.values[ci, ri]
            # print(c,r, sig)
            if sig < 0.05:
                # print(sig)
                ax.text(ri+widthx,ci+widthy, 
                        '**',
                        ha = 'center',
                        color = 'black',
                        fontsize=8,
                       )
            
    
    sns.heatmap(corr_df, 
                cmap="RdBu",
                ax = ax,
                annot=True,
                vmin= vmin, vmax=vmax,
                linewidth=0.3, 
                # square = True,
                # annot=True,
                # cbar_kws={"shrink": .8}
                cbar=False
               )

    # xticks
    ax.xaxis.tick_top()
    plt.yticks(np.arange(len(variable_type)) + .5, labels=variable_type)
    plt.xticks(np.arange(len(col_order)) + .5, labels=['(t-6)', '(t-5)', '(t-4)', '(t-3)', '(t-2)', '(t-1)', '(t-0)'])

    # axis labels
    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)

    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    ax.tick_params(which="major", bottom=False, top=False, left=False, right=False)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=18)

    # title
    plt.title(title, fontsize=18, pad=20)

    ax.figure.colorbar(sm)

    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['bottom'].set_color('none')

    save_path = save_prefix + '.pdf'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=-0, dpi=500)
    
    save_path = save_prefix + '.svg'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=-0, dpi=500)
    
    save_path = save_prefix + '.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=-0, dpi=500)

    plt.show()

def FSL(T, Td, rh):
    """
    NOAA's FSL algorithm

    parameters:
        T:  Temperatue (K)
        Td: Dew point temperature（K）
        rh: Relative humidity (%)
    return:
        visibility (km)

    vis = a * (T - Td) / (RH^1.75)
        when a = 6000, vis units is mile
        when a = 9656.1, vis units is km

    if T = Td, RH=100%, vis = 0
    """
    a = 9656.1
    vis = a * (T - Td) / np.power(rh, 1.75)

    return vis

def calc_vis_via_fsl(df):

    T_var = [f'TMP_GDS3_HTGL (t6)']
    T = df[T_var].values

    Td_var = [f'DPT_GDS3_HTGL (t6)']
    Td = df[Td_var].values

    RH_var = [f'R_H_GDS3_HTGL (t6)']
    RH = df[RH_var].values

    fsl_var = [f'fsl_vis (t6)']
    fsl_vis = FSL(T, Td, RH)
    fsl_vis = np.clip(fsl_vis, 0, 30)

    return fsl_vis, fsl_var


def get_dataset(sample_dir, year_range, spot_hour, wrf_length, vis_before_wrf_pre_seq_length, stations):
    
    print(f'Selecting stations: {stations}')
    
    sample = pd.DataFrame()
    for year in year_range:
        sample_file = os.path.join(sample_dir, f'{year}_spot{spot_hour}_wrf{wrf_length}_prev{vis_before_wrf_pre_seq_length}_samples.csv')
        print(f'Loading from {sample_file}')
        sample_df = pd.read_csv(sample_file, index_col=0)
        
        if stations != 'all':
            sample_df = sample_df[sample_df['ori_ID (t6)'].isin(stations)]
        
        sample = pd.concat([sample, sample_df], axis=0)
        
    sample.drop_duplicates(inplace=True)
    return sample

def find_opt_thres(pr_thres, result):
    
    f1_list = []
    far_list = []
    pod_list = []
    ets_list = []
    hss_list = []
    csi_list = []

    thres_list = np.arange(0, np.max(pr_thres), np.max(pr_thres)/30)
    print('thres_list:', thres_list)
    
    for thres in thres_list:
        # print('\n','-'*20, f'Thres:{thres}', '-'*20)
        
        pre_cls = np.where(result['pred_probs'] >= thres, 1, 0)
        # print('pre_cls:', pre_cls)
        
        f1, pod, far, csi, ets, hss, acc = show_confusion_matrix_and_cls_metrics(pre_cls, result['gt'], show_img=False)
        
        # print('\nF1: ', f1, '\nPOD: ', pod, '\nFAR: ', far, '\nCSI: ', csi, '\nETS: ', ets, '\nHSS: ', hss)
        f1_list.append(f1); pod_list.append(pod); far_list.append(far); csi_list.append(csi);
        ets_list.append(ets); hss_list.append(hss)

    metric_df = {'thres':thres_list ,'F1': f1_list, 'POD': pod_list, 'FAR': far_list, 'CSI': csi_list, 'ETS':ets_list, 'HSS': hss_list}
    metric_df = pd.DataFrame(metric_df)
    ax = sns.lineplot(data=metric_df[['F1','POD','FAR','CSI','ETS','HSS']])

    xticks = [round(x,4) for x in thres_list]
    # print(xticks)
    # ax.set_xticklabels(xticks)
    plt.xticks(np.arange(0,30,4), xticks[::4])
    plt.xticks(size='small',rotation=68,fontsize=10)
    plt.show()

    # print(metric_df)
    # print(metric_df.describe())

    # Find the best thres in PR Curves
    print('\n----------------------Max ETS------------------------')
    opt_row = metric_df.sort_values(by="ETS", ascending=False).head(1)
    opt_thres = opt_row['thres'].values[0]
    
    # opt_row = metric_df.groupby(by='ETS').max()
    print(opt_row)
    
    return opt_thres

def get_feature_importance(gbm):
    print('feature importance (gain)')
    imp = pd.DataFrame({
                    'feature': gbm.feature_name(),
                    'importance': gbm.feature_importance('gain')
                }).sort_values(by='importance')
    print(imp)
    
    plt.figure(figsize=(16, 16))
    sns.barplot(data=imp.sort_values(by="importance", ascending=False).head(50), x='importance', y='feature')
    plt.title("Feature Importance")

    # nfeats = 20
    # importance_types = ['gain', 'split']
    # fig, ax = plt.subplots(2,1,figsize=(14,30))
    # for i, imp_i in enumerate(importance_types):
    #     lgb.plot_importance(
    #             gbm,
    #             ax=ax[i],
    #             max_num_features=nfeats,
    #             importance_type=imp_i,
    #             xlabel=imp_i,
    #             )
    plt.show()
    plt.close()
    return


def show_PR_curves(recall, precision, PR_AUC):
    # plot the precision-recall curves
    plt.plot(recall, precision, marker='.', label='LGBM')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title(f'PR_AUC = {PR_AUC}')
    plt.show()
    

def FAR(cm):
    # tn, fp, fn, tp = cm
    # print('FAR')
    # print(cm.ravel())
    correctnegatives, falsealarms, misses, hits = cm.ravel()
    
    if hits + falsealarms == 0:
        return 1
    return falsealarms / (hits + falsealarms)

def MAR(cm):
    # tn, fp, fn, tp = cm
    correctnegatives, falsealarms, misses, hits = cm.ravel()
    return misses / (hits + misses)

def TS(cm):
    '''
    TS: 风险评分ThreatScore;
    CSI: critical success index 临界成功指数;
    '''
    # tn, fp, fn, tp = conf_matrix.ravel()
    correctnegatives, falsealarms, misses, hits = cm.ravel()
    return hits/(hits + falsealarms + misses)

def ETS(cm):
    '''
    ETS - Equitable Threat Score
    '''
    correctnegatives, falsealarms, misses, hits = cm.ravel()
    num = (hits + falsealarms) * (hits + misses)
    den = hits + misses + falsealarms + correctnegatives
    Dr = num / den
    ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)
    return ETS

def POD(cm):
    correctnegatives, falsealarms, misses, hits = cm.ravel()
    return hits / (hits + misses)

def HSS(cm):
    '''
    HSS - Heidke skill score
    '''
    correctnegatives, falsealarms, misses, hits = cm.ravel()

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses**2 + falsealarms**2 + 2*hits*correctnegatives + (misses + falsealarms)*(hits + correctnegatives))

    return HSS_num / HSS_den

def ACC(cm):
    correctnegatives, falsealarms, misses, hits = cm.ravel()
    return (correctnegatives + hits) / (correctnegatives + falsealarms + misses + hits)
    # (TP + TN) / (TP + TN + FP + FN)
    
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)
    
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j], 
                horizontalalignment = 'center', fontsize=14,
                color = 'white' if cm[i,j] > thresh else 'black')
        
        plt.tight_layout()
        plt.ylabel('Truth', fontsize=14)
        plt.xlabel('Prediction', fontsize=14)
        
def show_confusion_matrix_and_cls_metrics(pre, truth, 
                                          title='Confusion matrix', 
                                          show_img=True,
                                          save_path_prefix=None):
    f1 = f1_score(truth, pre)
    # print('f1_score: ', f1)

    cm = confusion_matrix(truth, pre)
    # print('show_confusion_matrix_and_cls_metrics: ', cm)
    
    if show_img:
        f = plt.figure(1)
        plot_confusion_matrix(cm, classes=['Fog-free', 'Fog'], title=title)
        if save_path_prefix:
            print(f'saving fig to {save_path_prefix}')
            plt.savefig(save_path_prefix + '.pdf', dpi=400)
            plt.savefig(save_path_prefix + '.svg', dpi=400)
            plt.savefig(save_path_prefix + '.png', dpi=400)
        plt.show()

    far = FAR(cm)
    pod = POD(cm)
    far = FAR(cm)
    ets = ETS(cm)
    hss = HSS(cm)
    csi = TS(cm)
    acc = ACC(cm)

    return f1, pod, far, csi, ets, hss, acc

def anly_by_lead_hours_in_24hours(station, pre_result_df, threshold, delta_time):
    """
    分析预报时效24小时内的逐小时结果
    """
    
    # leads_range = np.arange(1,61,6)
    leads_range = np.arange(0,25,delta_time)
    # print(len(leads_range), leads_range)
    
    if station != 'all':
        sta_res = pre_result_df[pre_result_df['ori_ID (t6)']==station]
    else:
        sta_res = pre_result_df
        
    met_col = ['lead_hours', 'F1','POD', 'FAR', 'CSI', 'ETS', 'HSS', 'ACC']
    LGBM_metrics = pd.DataFrame(columns=met_col)
    FSL_metrics = pd.DataFrame(columns=met_col)
    WRF_metrics = pd.DataFrame(columns=met_col)

    for l,lh in enumerate(leads_range):
            
        res = sta_res[sta_res['lead_hours'] == lh]
        if len(res) == 0:
            print(f'this lead time has not results in lead_hours={lh}~{lh+delta_time}h, skip!')
            met_df = {
                'lead_hours': lh,
                'F1': 0, 'POD':0, 'FAR':0, 
                'CSI':np.nan, 'ETS':np.nan, 'HSS':0, 'ACC':0,
            }
            met_df = pd.DataFrame(met_df, index=[0])
            LGBM_metrics = pd.concat([LGBM_metrics, met_df], axis=0)
            FSL_metrics = pd.concat([FSL_metrics, met_df], axis=0)
            WRF_metrics = pd.concat([WRF_metrics, met_df], axis=0)
            continue
            
        
        fog_num = res['gt'].value_counts().shape[0]
        if fog_num == 1:
            print(f'{station}, There is no Fog in lead_hours={lh} h', res['gt'].value_counts())
            met_df = {
                'lead_hours': lh,
                'F1': 0, 'POD':0, 'FAR':0, 
                'CSI':np.nan, 'ETS':np.nan, 'HSS':0, 'ACC':0,
            }
            met_df = pd.DataFrame(met_df, index=[0])
            LGBM_metrics = pd.concat([LGBM_metrics, met_df], axis=0)
            FSL_metrics = pd.concat([FSL_metrics, met_df], axis=0)
            WRF_metrics = pd.concat([WRF_metrics, met_df], axis=0)
            continue

        pre_cls = np.where(res['pred_probs'].values >= threshold, 1, 0)

                
        # print(f'\n------------------------LGBM lead_hours:{ls}~{le}h------------------------')
        f1, pod, far, csi, ets, hss, acc = show_confusion_matrix_and_cls_metrics(pre_cls, 
                                                                                 res['gt'].values, 
                                                                                 show_img=False)
        met_df = {
            'lead_hours': lh,
            'F1': f1, 'POD':pod, 'FAR':far, 
            'CSI':csi, 'ETS':ets, 'HSS':hss, 'ACC':acc,
        }
        met_df = pd.DataFrame(met_df, index=[0])
        LGBM_metrics = pd.concat([LGBM_metrics, met_df], axis=0)
        del met_df

        #  print(f'\n------------------------FSL lead_hours:{ls}~{le}h------------------------')
        f1, pod, far, csi, ets, hss, acc = show_confusion_matrix_and_cls_metrics(res['fsl_vis_cls'].values, 
                                                                                 res['gt'].values, 
                                                                                 show_img=False)
        met_df = {
            'lead_hours': lh,
            'F1': f1, 'POD':pod, 'FAR':far, 
            'CSI':csi, 'ETS':ets, 'HSS':hss, 'ACC':acc,
        }
        met_df = pd.DataFrame(met_df, index=[0])
        FSL_metrics = pd.concat([FSL_metrics, met_df], axis=0)
        del met_df

        #  print(f'\n------------------------WRF lead_hours:{ls}~{le}h------------------------')
        f1, pod, far, csi, ets, hss, acc = show_confusion_matrix_and_cls_metrics(res['wrf_vis_cls'].values, 
                                                                                 res['gt'].values, 
                                                                                 show_img=False)
        met_df = {
            'lead_hours': lh,
            'F1': f1, 'POD':pod, 'FAR':far, 
            'CSI':csi, 'ETS':ets, 'HSS':hss, 'ACC':acc,
        }
        met_df = pd.DataFrame(met_df, index=[0])
        WRF_metrics = pd.concat([WRF_metrics, met_df], axis=0)
        del met_df
        

    # fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.3
    x = leads_range    
    x_ticks = leads_range
    # y_ticks = np.arange(0, 0.4, 0.05)
    
    colors1 = plt.get_cmap('tab20c')([0,4,8])

    fig, ax = plt.subplots(figsize=(20, 8))
    colors1 = plt.get_cmap('tab20c')([0,4,8])
    ax.bar(x-width, LGBM_metrics['ETS'].values, width, color=colors1[0], label='LGBM')
    ax.bar(x, WRF_metrics['ETS'].values, width, color=colors1[1], label='WRF')
    ax.bar(x+width, FSL_metrics['ETS'].values, width, color=colors1[2], label='FSL')
    
    ax.set_ylabel('ETS')
    ax.set_ylim(0, 0.3)
    
    ax.set_xticks(x_ticks)
    ax.set_xlabel('Lead Hours')
    
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    
    lgb_avg_ets = round(LGBM_metrics['ETS'].mean(),4)
    lgb_avg_csi = round(LGBM_metrics['CSI'].mean(),4)
    wrf_avg_ets = round(WRF_metrics['ETS'].mean(),4)
    wrf_avg_csi = round(WRF_metrics['CSI'].mean(),4)
    fsl_avg_ets = round(FSL_metrics['ETS'].mean(),4)
    fsl_avg_csi = round(FSL_metrics['CSI'].mean(),4)
    
    plt.title(f'Station: {station}, Visibility <= 1km, lead_hours=1~24\nUse Optimal Threshold:{threshold}\nlgb_avg_ETS={lgb_avg_ets},lgb_avg_CSI={lgb_avg_csi}\nwrf_avg_ETS={wrf_avg_ets},wrf_avg_CSI={wrf_avg_csi}\nfsl_avg_ETS={fsl_avg_ets},fsl_avg_CSI={fsl_avg_csi}')
    plt.legend()
    plt.show()
    plt.close()       

    print('-------------------LGBM-------------------')
    print(LGBM_metrics)
    print('-------------------FSL-------------------')
    print(FSL_metrics)
    print('-------------------WRF-------------------')
    print(WRF_metrics)
    
    return LGBM_metrics, WRF_metrics, FSL_metrics

def anly_station_by_station(pre_result_df, threshold, pre_thres_list, lead_hours, delta_time):
    
    col = ['lead_hours', 'ETS', 'station_id', 'method']
    met_results = pd.DataFrame(columns = col)
    
    thres_lst = []
    for sid, sta_res in pre_result_df.groupby(by='ori_ID (t6)'):
        print(f'-----------------------{sid}---------------------------')
        
        if threshold == 'optimal':
            print('Finding the optimal threshold from subset of results.....')
            thres = find_opt_thres(pre_thres_list, sta_res.sample(frac=0.2))
        else:
            thres = threshold
        thres_lst.append(thres)
        
        if lead_hours == 24:
            lgb_met, wrf_met, fsl_met = anly_by_lead_hours_in_24hours(sid, sta_res, thres, delta_time)
        elif lead_hours == 60:
            lgb_met, wrf_met, fsl_met = anly_by_lead_hours_in_60hours(sid, sta_res, thres, delta_time)
    
        sub = lgb_met[['ETS', 'POD', 'FAR', 'HSS', 'lead_hours']]
        sub['station_id'] = sid; sub['method'] = 'LGBM'
        met_results = pd.concat([met_results, sub], axis=0)
    
        sub = wrf_met[['ETS', 'POD', 'FAR', 'HSS', 'lead_hours']]
        sub['station_id'] = sid; sub['method'] = 'WRF'
        met_results = pd.concat([met_results, sub], axis=0)
    
        sub = fsl_met[['ETS', 'POD', 'FAR', 'HSS', 'lead_hours']]
        sub['station_id'] = sid; sub['method'] = 'FSL'
        met_results = pd.concat([met_results, sub], axis=0)
    
    thres_lst = np.unique(np.array(thres_lst)).tolist()
    
    return met_results, thres_lst



def visual_by_box_01(met_results_df, opt_thres_lst):
    '''
    opt_thres_lst: optimal probability thresholds of all stations
    '''
    fontdict = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 18}
    sns.set_context("paper")  #配置paper的类型
    sns.set(style="ticks")
    
    met_results_df.dropna(inplace=True)
    met_results_df['ETS'] = met_results_df['ETS'].astype(float)
    met_results_df.sort_values(by='lead_hours', inplace=True)
    
    f, ax = plt.subplots(figsize=(20,8))

    # palette = sns.diverging_palette(255, 20, n=12)[::4]

    # 绘制分组小提琴图
    sns.boxplot(x = "lead_hours", # 指定x轴的数据
                y = "ETS", # 指定y轴的数据
                hue = "exp", # 指定分组变量
                data = met_results_df, # 指定绘图的数据集
                # palette = palette,
                linewidth = 1,
                saturation = 1,
                fliersize = 0,
            )

    # 去掉图像上边框和右边框
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)
    
    avg_ets_dict = {}
    for exp, exp_gp in met_results_df.groupby('exp'):
        avg_ets = exp_gp['ETS'].mean()
        avg_ets_dict[exp] = avg_ets

#     lgb_avg_ets = met_results_df[met_results_df['method'] == 'LGBM']['ETS'].mean()
#     wrf_avg_ets = met_results_df[met_results_df['method'] == 'WRF']['ETS'].mean()
#     fsl_avg_ets = met_results_df[met_results_df['method'] == 'FSL']['ETS'].mean()

    if len(opt_thres_lst) != 1:
        max_thres = round(np.max(opt_thres_lst),3); min_thres = round(np.min(opt_thres_lst),2)
        title_str = f'Thresholds: {min_thres}~{max_thres}'
    else:
        title_str = f'Thresholds: {opt_thres_lst[0]}'
        
    # plt.title(f'{title_str}\nLGBM_ETS={round(lgb_avg_ets,3)}, NMM_ETS={round(wrf_avg_ets,3)}, FSL_ETS={round(fsl_avg_ets,3)}', 
    #          fontdict=fontdict)
    plt.legend(loc = 'upper right', ncol = 1)
    plt.xlabel('Lead Hours', fontdict=fontdict)
    plt.ylabel('ETS', fontdict=fontdict)
    plt.xticks(fontproperties='Times New Roman', size=16)
    plt.yticks(fontproperties='Times New Roman', size=16)

    plt.show()
    
def visual_by_box(met_results_df, opt_thres_lst, lead_time=24):
    '''
    opt_thres_lst: optimal probability thresholds of all stations
    '''
    met_results_df = met_results_df[(met_results_df['lead_hours'] > 1) & (met_results_df['lead_hours'] <= lead_time)]
    met_results_df.dropna(inplace=True)
    met_results_df['ETS'] = met_results_df['ETS'].astype(float)
    met_results_df.sort_values(by='lead_hours', inplace=True)
    
    # 将Seaborn重置为默认设置
    plt.rcParams['font.size'] = 30
    sns.set(style="white", context="notebook")

    f, ax = plt.subplots(figsize=(24, 10))

    # 指定hue顺序
    hue_order = ["WRF", "FSL", 'LGBM']

    # 绘制分组小提琴图
    sns.boxplot(x = "lead_hours", # 指定x轴的数据
                y = "ETS", # 指定y轴的数据
                hue = "method", # 指定分组变量
                data = met_results_df, # 指定绘图的数据集
                hue_order = hue_order,
                linewidth = 2,
                saturation = 1,
                fliersize = 0,
                palette=sns.color_palette("Blues")[::2],
                # palette={"WRF": "#287885", "FSL": "#9AC9DB", 'LGBM':"#C82423"},
            )

    # 去掉图像上边框和右边框
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)

    lgb_avg_ets = met_results_df[met_results_df['method'] == 'LGBM']['ETS'].mean()
    wrf_avg_ets = met_results_df[met_results_df['method'] == 'WRF']['ETS'].mean()
    fsl_avg_ets = met_results_df[met_results_df['method'] == 'FSL']['ETS'].mean()

    if len(opt_thres_lst) != 1:
        max_thres = round(np.max(opt_thres_lst),3); min_thres = round(np.min(opt_thres_lst),2)
        title_str = f'Thresholds: {min_thres}~{max_thres}'
    else:
        title_str = f'Thresholds: {opt_thres_lst[0]}'
        
    plt.title(f'{title_str}\nLGBM_ETS={round(lgb_avg_ets,3)}, NMM_ETS={round(wrf_avg_ets,3)}, FSL_ETS={round(fsl_avg_ets,3)}')
    plt.legend(loc = 'upper right', ncol = 1)
    plt.xlabel('Lead Time (Hour)')
    plt.ylabel('ETS')
    
    plt.ylim(-0.05, 0.4)

    plt.show()
    
def anly_by_lead_hours_in_hours(station, pre_result_df, threshold, lead_time, delta_time, metric_name='ETS', limy=[0,0.2]):
    
    leads_range = np.arange(1,lead_time+1, delta_time)[1:]
    
    if station != 'all':
        sta_res = pre_result_df[pre_result_df['ori_ID (t6)']==station]
    else:
        sta_res = pre_result_df
        
    met_col = ['lead_hours', 'F1','POD', 'FAR', 'CSI', 'ETS', 'HSS', 'ACC']
    LGBM_metrics = pd.DataFrame(columns=met_col)
    FSL_metrics = pd.DataFrame(columns=met_col)
    WRF_metrics = pd.DataFrame(columns=met_col)

    for l,lh in enumerate(leads_range):
        
        res = sta_res[(sta_res['lead_hours'] >= lh) & (sta_res['lead_hours'] < lh+delta_time)]
    
        if len(res) == 0:
            print(f'this lead time has not results in lead_hours={lh}~{lh+delta_time}h, skip!')
            met_df = {
                'lead_hours': lh,
                'F1': 0, 'POD':0, 'FAR':0, 
                'CSI':np.nan, 'ETS':np.nan, 'HSS':0, 'ACC':0,
            }
            met_df = pd.DataFrame(met_df, index=[0])
            LGBM_metrics = pd.concat([LGBM_metrics, met_df], axis=0)
            FSL_metrics = pd.concat([FSL_metrics, met_df], axis=0)
            WRF_metrics = pd.concat([WRF_metrics, met_df], axis=0)
            continue
            
        fog_num = res['gt'].value_counts().shape[0]
        if fog_num == 1:
            print(f'{station}, There is no Fog in lead_hours={lh}~{lh+delta_time}h', res['gt'].value_counts())
            met_df = {
                'lead_hours': lh,
                'F1': 0, 'POD':0, 'FAR':0, 
                'CSI':np.nan, 'ETS':np.nan, 'HSS':0, 'ACC':0,
            }
            met_df = pd.DataFrame(met_df, index=[0])
            LGBM_metrics = pd.concat([LGBM_metrics, met_df], axis=0)
            FSL_metrics = pd.concat([FSL_metrics, met_df], axis=0)
            WRF_metrics = pd.concat([WRF_metrics, met_df], axis=0)
            continue
            
        pre_cls = np.where(res['pred_probs'].values >= threshold, 1, 0)

        # print(f'\n------------------------LGBM lead_hours:{ls}~{le}h------------------------')
        f1, pod, far, csi, ets, hss, acc = show_confusion_matrix_and_cls_metrics(pre_cls, 
                                                                                 res['gt'].values, 
                                                                                 show_img=False)
        met_df = {
            'lead_hours': lh,
            'F1': f1, 'POD':pod, 'FAR':far, 
            'CSI':csi, 'ETS':ets, 'HSS':hss, 'ACC':acc,
        }
        met_df = pd.DataFrame(met_df, index=[0])
        LGBM_metrics = pd.concat([LGBM_metrics, met_df], axis=0)
        del met_df

        #  print(f'\n------------------------FSL lead_hours:{ls}~{le}h------------------------')
        f1, pod, far, csi, ets, hss, acc = show_confusion_matrix_and_cls_metrics(res['fsl_vis_cls'].values, 
                                                                                 res['gt'].values, 
                                                                                 show_img=False)
        met_df = {
            'lead_hours': lh,
            'F1': f1, 'POD':pod, 'FAR':far, 
            'CSI':csi, 'ETS':ets, 'HSS':hss, 'ACC':acc,
        }
        met_df = pd.DataFrame(met_df, index=[0])
        FSL_metrics = pd.concat([FSL_metrics, met_df], axis=0)
        del met_df

        #  print(f'\n------------------------WRF lead_hours:{ls}~{le}h------------------------')
        f1, pod, far, csi, ets, hss, acc = show_confusion_matrix_and_cls_metrics(res['wrf_vis_cls'].values, 
                                                                                 res['gt'].values, 
                                                                                 show_img=False)
        met_df = {
            'lead_hours': lh,
            'F1': f1, 'POD':pod, 'FAR':far, 
            'CSI':csi, 'ETS':ets, 'HSS':hss, 'ACC':acc,
        }
        met_df = pd.DataFrame(met_df, index=[0])
        WRF_metrics = pd.concat([WRF_metrics, met_df], axis=0)
        del met_df
        
    LGBM_metrics.fillna(0, inplace=True)
    FSL_metrics.fillna(0, inplace=True)
    WRF_metrics.fillna(0, inplace=True)
    
    fig, ax = plt.subplots(figsize=(22, 10), dpi=300)
    
    colors1 = plt.get_cmap('tab20c')([0,4,8])
    color1 = [40/255, 120/255, 181/255, 1]
    color2 = [154/255, 201/255, 219/255, 1]
    color3 = [248/255, 172/255 ,140/255, 1]
    # colors_list.append(0.8)
    # print(colors_list)
    
    width = 0.6
    x = leads_range
    
    # print(LGBM_metrics['ETS'].values)
    ax.bar(x-width, WRF_metrics[metric_name].values, width, color='#2878B5', label='NMM')
    ax.bar(x, FSL_metrics[metric_name].values, width, color='#9AC9DB', label='FSL')
    ax.bar(x+width, LGBM_metrics[metric_name].values, width, color='#C82423', label='ML')
    
    ax.set_ylabel(metric_name, fontsize=22,  fontweight='bold')
    ax.set_ylim(limy[0], limy[1])
    
    ax.set_xticks(leads_range)
    ax.set_xlabel('Lead Hours', fontsize=22, fontweight='bold')

    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    
    lgb_avg_ets = round(LGBM_metrics['ETS'].mean(),4)
    wrf_avg_ets = round(WRF_metrics['ETS'].mean(),4)
    fsl_avg_ets = round(FSL_metrics['ETS'].mean(),4)
    
    lgb_avg_pod = round(LGBM_metrics['POD'].mean(),4)
    wrf_avg_pod = round(WRF_metrics['POD'].mean(),4)
    fsl_avg_pod = round(FSL_metrics['POD'].mean(),4)
    
    lgb_avg_far = round(LGBM_metrics['FAR'].mean(),4)
    wrf_avg_far = round(WRF_metrics['FAR'].mean(),4)
    fsl_avg_far = round(FSL_metrics['FAR'].mean(),4)
    
    lgb_avg_hss = round(LGBM_metrics['HSS'].mean(),4)
    wrf_avg_hss = round(WRF_metrics['HSS'].mean(),4)
    fsl_avg_hss = round(FSL_metrics['HSS'].mean(),4)
    
    plt.xticks(fontsize=20, fontweight='bold',) # 设置x轴刻度字体大小
    plt.yticks(fontsize=20, fontweight='bold',) # 设置y轴刻度字体大小

    plt.title(f'Station: {station}, Visibility <= 1km, Threshold:{threshold}\nlgb_avg_ETS={lgb_avg_ets}, wrf_avg_ETS={wrf_avg_ets}, fsl_avg_ETS={fsl_avg_ets}', 
              fontsize=16, 
              fontweight='bold',
              y=1.01)
    plt.legend(fontsize=16)
    plt.show()
    
    plt.close()       

    print('-------------------LGBM-------------------')
    print(f'avg POD={lgb_avg_pod}, FAR={lgb_avg_far}, ETS={lgb_avg_ets}, HSS={lgb_avg_hss}')
    print(LGBM_metrics)
    print('-------------------FSL-------------------')
    print(f'avg POD={fsl_avg_pod}, FAR={fsl_avg_far}, ETS={fsl_avg_ets}, HSS={fsl_avg_hss}')
    print(FSL_metrics)
    print('-------------------WRF-------------------')
    print(f'avg POD={wrf_avg_pod}, FAR={wrf_avg_far}, ETS={wrf_avg_ets}, HSS={wrf_avg_hss}')
    print(WRF_metrics)
    
    return LGBM_metrics, WRF_metrics, FSL_metrics

class FocalLoss:

    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better

def create_lgb(X_train: pd.DataFrame, y_train: pd.DataFrame, 
               X_val: pd.DataFrame, y_val: pd.DataFrame, 
               features, model_params, 
               use_focal_loss, fl_alpha=0, fl_gamma=0, init_model=False):

    """ Takes as input the training set composed of X_train and y_train. 
    It returns an xgboost model tuned with the specified training set. 
    model_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'learning_rate': 0.02,
            'max_depth': 6,
            'num_leaves': 64,
            'num_thread': 16,
            'feature_fraction': 0.8,
            'feature_fraction_seed': 66,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'bagging_seed': 100,
            'verbose':-1,
        }
        
    """
    
    # study = optuna.create_study(direction='maximize')
    # study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
    # params=study.best_params
    
    print('\nStrating train...')

    trn_data = lgb.Dataset(X_train, y_train, feature_name = features)
    val_data = lgb.Dataset(X_val, y_val, reference = trn_data)
    
    if not use_focal_loss:
        
        if init_model == None:
            model = lgb.train(
                model_params,
                train_set = trn_data,
                valid_sets = val_data,
                verbose_eval = 300, 
                num_boost_round = 15000,
                early_stopping_rounds = 300, 
            )
            print('Train Finished')

            return model, 0
           
        else:
            print(f'Load pretrained model from {init_model}')
            model = lgb.train(
                model_params,
                train_set = trn_data,
                valid_sets = val_data,
                verbose_eval = 300, 
                num_boost_round = 15000,
                early_stopping_rounds = 300, 
                init_model = init_model,
            )
            print('Train Finished')

            return model, 0
    
    else:
        print(f'\nUse focal loss (alpha={fl_alpha}, gamma={fl_gamma})....\n')
        fl = FocalLoss(alpha=fl_alpha, gamma=fl_gamma)

        if init_model == None:
            model = lgb.train(
                model_params,
                fobj = fl.lgb_obj,
                feval = fl.lgb_eval,
                train_set = trn_data,
                valid_sets = val_data,
                verbose_eval = 300, 
                num_boost_round = 15000,
                early_stopping_rounds = 300, 
                init_model = init_model)
        else:
            print(f'Load pretrained model from {init_model}')
            model = lgb.train(
                model_params,
                fobj = fl.lgb_obj,
                feval = fl.lgb_eval,
                train_set = trn_data,
                valid_sets = val_data,
                verbose_eval = 300, 
                num_boost_round = 15000,
                early_stopping_rounds = 300)
            
        
        print('Train Finished')
        return model, fl
    
    
# the underbagging algorithm
def easyensemble(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                 X_val: pd.DataFrame, y_val: pd.DataFrame, 
                 features: list,
                 target: str, 
                 model_params: dict,
                 n_estimators: int,
                 resample_type: str,  
                 sampling_ratio:float,
                 result_dir: str,
                 model_save_path_prefix: str,
                 use_focal_loss: bool,
                 alpha=0, beta=0):
    
    """ Simple implementation of easyensemble but with XGBoost as learners instead of AdaBoost.
    Takes as input a training set, the different features and the number of XGBoost model. """
    
    print('Starting easyensemble...')
    
    if use_focal_loss:
        print('Using focal loss...')
        
    models_path = []
    
    for estimator in range(n_estimators):
        
        print(f'\n\n== estimator: {estimator} ==')
        
        # %% resampling
        if resample_type == 'under_sample':
            rs = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=randint(1,1000))
            X_train_, y_train_ = rs.fit_resample(X_train, y_train)
            print(f'\nAfter {resample_type}({sampling_ratio}): ', sorted(Counter(y_train_).items()))
        elif resample_type == 'over_sample':
            rs = RandomOverSampler(sampling_strategy=sampling_ratio, random_state=randint(1,1000))
            X_train_, y_train_ = rs.fit_resample(X_train, y_train)
            print(f'\nAfter {resample_type}({sampling_ratio}): ', sorted(Counter(y_train_).items()))
        elif resample_type == 'None':
            X_train_, y_train_ = X_train, y_train
        
        # %% if model have created, skip
        model_path = os.path.join(result_dir, 'model', f'{model_save_path_prefix}_{estimator}.txt')
        models_path.append(model_path)
#         if os.path.exists(model_path):
#             print(f'Having created {model_path}, skip!')
#             continue
        
        # %% create model and train
        model, focal_loss = create_lgb(X_train_, y_train_, X_val, y_val, 
                                       features, model_params, 
                                       use_focal_loss, alpha, beta)
        
        # %% Save model
        model.save_model(model_path)
        print(f'\nSaving model to {model_path}')
         
    return models_path, focal_loss
