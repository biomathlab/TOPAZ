import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from ripser import ripser
import scipy

import concurrent.futures
from scipy.integrate import ode
import glob
import imageio as io
from itertools import repeat

from Scripts.DorsognaNondim_Align import *
from Scripts.crocker import *

def filtering_df(filt_df, FRAME_LIST, track_len=10, max_frame=128, min_speed=None):
    
    if min_speed is None:
        #make a histogram of velocities (displacement over some number of frames) of all cells at all times
        object = filt_df.copy()
        mags_xy = []
        for idx in np.arange(track_len,max_frame):
            p_t1=object.particle.values[object.frame==idx-track_len]
            p_t2=object.particle.values[object.frame==idx]

            object2= object[(object["particle"].isin(p_t1)) & (object["particle"].isin(p_t2))]

            frame_t1 = object2[object2['frame']==idx-track_len]
            frame_t2 = object2[object2['frame']==idx]
            delta_x = frame_t2.x.values-frame_t1.x.values
            delta_y = frame_t2.y.values-frame_t1.y.values

            mag_xy = np.sqrt(np.square(delta_x)+np.square(delta_y))
            mags_xy.append(mag_xy)

        mags_xy_data=np.concatenate(mags_xy,axis=0)
        a1,b1,c1=scipy.stats.lognorm.fit(mags_xy_data)
        rv = scipy.stats.lognorm(a1,b1,c1)
        min_speed=rv.ppf(.25)

    new_data = pd.DataFrame(columns=['x','y','vx','vy','angle','mag','frame'])

    num_cells_in_frame = []
    num_cells_in_frame_all = []

    for frame_idx in FRAME_LIST:
        p_t1=filt_df.particle.values[filt_df.frame==frame_idx-track_len]
        p_t2=filt_df.particle.values[filt_df.frame==frame_idx]
        object2= filt_df[(filt_df["particle"].isin(p_t1)) & (filt_df["particle"].isin(p_t2))]
        frame_t2 = object2[object2['frame']==frame_idx]
        frame_t1 = object2[object2['frame']==frame_idx-track_len]
        delta_x = frame_t2.x.values-frame_t1.x.values
        delta_y = frame_t2.y.values-frame_t1.y.values
        angle_xy = np.arctan2(delta_y,delta_x)
        deg_xy = np.mod(np.degrees(angle_xy),360)
        mag_xy = np.sqrt(np.square(delta_x)+np.square(delta_y))
        x = frame_t2.x.values
        y = frame_t2.y.values
        vx = frame_t2.vx.values
        vy = frame_t2.vy.values
        particle = frame_t2.particle.values
        count=0
        all_count = 0
        for idx,val in enumerate(x):
            all_count +=1
            if mag_xy[idx]>min_speed:
                count+=1
                df1 = pd.DataFrame({'x':[x[idx]],'y':[y[idx]],'vx':[vx[idx]],'vy':[vy[idx]],'angle':[deg_xy[idx]],'mag':[mag_xy[idx]],'frame':[frame_idx],'particle':[particle[idx]]})
                new_data=pd.concat([new_data,df1],ignore_index=True,axis=0)
        num_cells_in_frame.append(count)
        num_cells_in_frame_all.append(len(p_t2))
        
    return new_data, min_speed

def run_ABC_crocker(C_idx,L_idx,W_idx,T0,TF,DT,in_num_agents):
    #VANILLA CROCKER
    #Which DataFrame columns to use as dimensions
    DATA_COLS = ('x','y','angle')

    #Make time vector
    time_vec = np.arange(T0,TF+DT,DT)
    #Initial conditions
    rng = np.random.default_rng()

    num_agents = in_num_agents

    #List of frame values to use, must be aligned for direct comparison
    true_FRAME_LIST = range(20,120,1)
    pred_FRAME_LIST = range(10,120,1) #starts at 10 because of angle computation
    #compute the data for the crocker plot
    #compute the data for the crocker plot
    if 'angle' in DATA_COLS:
        PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
    if 'vx' in DATA_COLS:
        PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
    
    #Stochastic diffusivity parameter
    SIGMA = 0 #0.05
    #alpha
    ALPHA = 1.0
    BETA = 0.5

    Cs = np.linspace(0.1,3.0,30)
    Ls = np.linspace(0.1,3.0,30)
    Ws = np.linspace(0.0,0.1,11)

    Cval = Cs[C_idx-1]
    Lval = Ls[L_idx-1]
    Wval = Ws[W_idx]

    # Get true data frame:
    true_PATH = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/df.pkl'

    true_df = pd.read_pickle(true_PATH)
    if 'angle' in DATA_COLS:
        true_df, _ = filtering_df(true_df, pred_FRAME_LIST, track_len=10, max_frame=126, min_speed=0)

    true_crocker = compute_crocker_custom(true_df,true_FRAME_LIST,PROX_VEC,data_cols=DATA_COLS,betti=[0,1])
    plot_crocker_highres_split(true_crocker,
                               PROX_VEC,
                               [50,150,250,350,450],
                               true_crocker,
                               PROX_VEC,
                               [50,150,250,350,450],
                               [Cval,Lval,Wval],
                                           save_path='./Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/true_crocker.png')

    # load median values
    median_path = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/medians.npy'
    medians = np.load(median_path,allow_pickle=True)

    # Get ABC results:
    ABC_path = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/df_ABC.pkl'
    ABC_df = pd.read_pickle(ABC_path)
    if 'angle' in DATA_COLS:
        ABC_df, _ = filtering_df(ABC_df, pred_FRAME_LIST, track_len=10, max_frame=126, min_speed=0)

    ABC_crocker = compute_crocker_custom(ABC_df,true_FRAME_LIST,PROX_VEC,data_cols=DATA_COLS,betti=[0,1])
    plot_crocker_highres_split(ABC_crocker,
                               PROX_VEC,
                               [50,150,250,350,450],
                               ABC_crocker,
                               PROX_VEC,
                               [50,150,250,350,450],
                               medians,
                               save_path='./Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/ABC_crocker.png')


    # get the difference in the true and simulated (median value) crocker plots 
    crocker_diff = true_crocker - ABC_crocker
    diff_path = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/crocker_differences.npy'
    np.save(diff_path, crocker_diff)


    



