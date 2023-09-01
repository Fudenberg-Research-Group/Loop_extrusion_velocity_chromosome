import glob
import ast 

import datetime

SEPARATION_WT = 100
STALL_WT = 0.9
STEPS_WT = 200
Tad=1000


already_processed = []
for fname  in glob.glob('/home1/start/polychrom/pds5/draft-extrusion/Loop_extrusion_velocity/sims/*_STALL*'):
    already_processed.append(fname.split('/')[-1])

with open(str(datetime.date.today())+'_runfile.txt','w') as f:
    for LIFETIME_WT in [16.6, 50, 150]:
        for velocity_multiplier in [0.33, 1, 3]:
            for RSTALL_WT in [0.9]:
                for LSTALL_WT in [0.0]:
                    if LSTALL_WT==RSTALL_WT:
                        continue
                    for RC_WT in [0.0]:
                        for LC_WT in [0.9]:
                            if LC_WT==RC_WT:
                                continue
                            for stalldist_WT in [1]:
                                for SEPARATION_WT in [50, 100, 300, 500]:
                                    paramset = (
                                       'LIFETIME_'+str(LIFETIME_WT)+
                                       '_SEPARATION_'+str(SEPARATION_WT)+
                                       '_RSTALL_'+str(RSTALL_WT) +
                                       '_LSTALL_'+str(LSTALL_WT)+
                                       '_RC_'+str(RC_WT)+
                                       '_LC_'+str(LC_WT)+
                                       '_stalldist_'+str(stalldist_WT)+
                                       '_STEPS_'+str(STEPS_WT)+
                                       '_velocitymultiplier_'+str(velocity_multiplier)
                                       +'_Tad_'+str(Tad)
                                      )
                                    if paramset not in already_processed:
                                        f.write( paramset +'\n')
                                    else:
                                        print('already done')
                                
            
            
