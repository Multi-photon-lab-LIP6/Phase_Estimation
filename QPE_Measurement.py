import time
import itertools
import random
import TT
from basis import MEAS_WP_ANGLES
import motors_control
import numpy as np
import Voting as Vote
import os
import counts_statistics as cs
import glob
import tqdm
import scipy as sp

def Unitary(angle,angles):

    a = angle[0]
    b = angle[1]
    y = angle[2]
    
    f = (1/2)*(-np.cos(2*(a-b))-np.cos(2*(b-y))) - np.real(np.exp(-1j*angles/2))
    g = (1/2)*(np.sin(2*(a - b)) + np.sin(2*(b - y)))
    h = (1/2)*(-np.sin(2*(a - b)) - np.sin(2*(b - y)))
    v = (1/2)*(-np.cos(2*(a-b))- np.cos(2*(b - y))) - np.real(np.exp(1j*angles/2))

    K = (1/2)*(-np.cos(2*b) + np.cos(2*(a - b + y))) - np.imag(np.exp(-1j*angles/2))
    m = (1/2)*(-np.sin(2*b) + np.sin(2*(a - b + y)))
    z = (1/2)*(-np.sin(2*b) + np.sin(2*(a - b + y)))
    e = (1/2)*(np.cos(2*b) - np.cos(2*(a - b + y))) - np.imag(np.exp(1j*angles/2))
    
def solving(angle,angles):
    result = sp.least_squares(Unitary,angle,method='trf',args=[angles],max_nfev=1000000000)
    QWP1 = result.x[0]
    HWP1 = result.x[1]
    QWP2 = result.x[2]
    return(QWP1*180/np.pi,HWP1*180/np.pi,QWP2*180/np.pi)


def main():
    varying_angle_1 = np.linspace(0,np.pi,num = 50)
    varying_angle_2 = np.pi/4
    varying_angle_3 = np.pi/2
    varying_angle_4 = np.pi/3
    
    for i in tqdm(range(len(varying_angle_1))):
    # Angle per qubit
        angle = np.array([varying_angle_4,varying_angle_3,varying_angle_2,varying_angle_1[i]])
    ##################################################################
    ####################### TIME TAGGER ##############################
    ##################################################################
    
    CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]
    TRIGGER = [0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.12]
    DELAY = [0,  2750, -26650, -33020, -32230, -33210, 2720, 4990]
    AQUISITION_TIME = int(1.5*60E12) # in picosecond
    
    N_REP = 1
    tt = TT.Swabian(CHANNELS, TRIGGER, DELAY, "QPE",f"{AQUISITION_TIME}")
    """Defining the coincidence channels we want to save
    If you change the order, make sure it matches with the analysis code
    """
    GROUPS = [(1,3,5,7),(1,3,5,8),(1,3,6,7),(1,3,6,8),
            (1,4,5,7),(1,4,5,8),(1,4,6,7),(1,4,6,8),
            (2,3,5,7),(2,3,5,8),(2,3,6,7),(2,3,6,8),
            (2,4,5,7),(2,4,5,8),(2,4,6,7),(2,4,6,8)] 
    COINCIDENCE_WINDOW = 200 # in picosecond

    ##################################################################
    ##################### DEFINING THE PLAYERS #######################
    ##################################################################

    players = ["arya" , "bran", "cersei", "dany"]
    sample=["SQWP1","SHWP","SQWP2"]

    # Create new device, Connect, begin polling, and enable
    arya, bran, cersei, dany = motors_control.players_init(players, sample)
    
    ##################################################################
    ####################### ANGLE OF WAVE PLATES #####################
    ##################################################################
    
    arya_SWP = [-18.60208456082875, 70.01563655846527, -17.553712134270178]
    bran_SWP = [-18.60208456082875, 70.01563655846527, -17.553712134270178]
    cersei_SWP = [-18.60208456082875, 70.01563655846527, -17.553712134270178]
    dany_SWP = [-18.60208456082875, 70.01563655846527, -17.553712134270178]
    
    ##################################################################
    ##################### VERIFICATION TASK ##########################
    ##################################################################
    for i in range(len(angle)):
                  
        # Reset the angle 
        print("Setting samples's angles")
        # Choosing the Agent to verify :
        
        angle_arya = Vote.solving(arya_SWP,angle[0])
        bran_arya = Vote.solving(bran_SWP,angle[1])
        cersei_arya = Vote.solving(cersei_SWP,angle[2])
        dany_arya = Vote.solving(dany_SWP,angle[3])
        
        arya.set.set_sample_angles(angle_arya)
        bran.set_sample_angles(bran_arya)
        cersei.set_sample_angles(cersei_arya)
        dany.set_sample_angles(dany_arya)
        
        arya.set_meas_basis('x')
        cersei.set_meas_basis('x')
        bran.set_meas_basis('x')
        dany.set_meas_basis('x')
        
        print("Gathering the counts for bad (no bad words) analysis")
        tt.measure(AQUISITION_TIME,N_REP, GROUPS, COINCIDENCE_WINDOW, count_singles=False, data_filename=f"\ABCD=xxxx_{angle[3]}_.txt", save_raw=True, save_params=True)
    
    with open ('Angle_file', mode ='w') as file :
        file.write(str(angle))
        file.close()
        
    tt.free_swabian()
    arya.off()
    bran.off()
    cersei.off()
    dany.off()
    

if __name__ == "__main__":
    main()


