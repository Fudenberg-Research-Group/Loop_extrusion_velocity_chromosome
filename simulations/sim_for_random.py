
#salloc --partition=debug --gres=gpu --mem-per-cpu=2GB --cpus-per-task=8

import ast
import pickle
import os
import time
import numpy as np
import polychrom

from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
from polychrom.lib.extrusion import  bondUpdater

import openmm 
import os 
import shutil


import warnings
import h5py 
import glob
import numpy
import sys

import pyximport; 

#pyximport.install() # need to pass setup_args now!
pyximport.install(setup_args={"include_dirs":numpy.get_include()}, reload_support=True)

from LEF_Dynamics import LEFTranslocatorDirectional

# -------defining parameters----------
#  -- basic loop extrusion parameters
N1 = 1500   # number of monomers
M = 10 # number of replicas of the simulation
N = N1 * M  # total number of monomers

#STALL_WT = 0.9
#LIFETIME_WT = 50
#SEPARATION_WT = 100 

# --- polymer parameters 
#STEPS_WT = 200   # MD steps per step of cohesin
stiff = 1
dens = 0.2
box = (N / dens) ** 0.33  # density = 0.1.

smcStepsPerBlock = 1  # now doing 1 SMC step per block 


filename = sys.argv[-1]

print(filename)


file=open('Testing.txt','a')
file.write('%s dddd\n'%filename)
file.close()


params = [ast.literal_eval(i) for i in filename.split('/')[-1].split('_')[1::2]]
LIFETIME_WT, SEPARATION_WT, RSTALL_WT, LSTALL_WT,RC_WT,LC_WT,stalldist_WT,STEPS_WT, velocity_multiplier, Tad = params
print(params,stalldist_WT)
if velocity_multiplier != 1:
    print('non unity velocity multiplier')
    LIFETIME = int(LIFETIME_WT * velocity_multiplier)
    RSTALL =   1 - (1 - (RSTALL_WT))**( 1/velocity_multiplier)
    LSTALL =   1 - (1 - (LSTALL_WT))**( 1/velocity_multiplier)
    RSTALL_C =   1 - (1 - (RC_WT))**( 1/velocity_multiplier)
    LSTALL_C =   1 - (1 - (LC_WT))**( 1/velocity_multiplier)
    stalldist=stalldist_WT
    steps = int(STEPS_WT / velocity_multiplier)
else:
    LIFETIME = int(LIFETIME_WT)
    RSTALL = RSTALL_WT
    LSTALL= LSTALL_WT
    RSTALL_C=RC_WT
    LSTALL_C=LC_WT
    stalldist=stalldist_WT
    steps = int(STEPS_WT)
SEPARATION = int(SEPARATION_WT)
TADsize = Tad
folder_name = (
               'LIFETIME_'+str(LIFETIME_WT)+
               '_SEPARATION_'+str(SEPARATION_WT)+
               '_RSTALL_'+str(RSTALL_WT) +
               '_LSTALL_'+str(LSTALL_WT)+
               '_RC_'+str(RC_WT)+
               '_LC_'+str(LC_WT)+
               '_stalldist_'+str(stalldist_WT)+
               '_STEPS_'+str(STEPS_WT)+
                '_velocitymultiplier_'+str(velocity_multiplier)+
                '_Tad_'+str(Tad)
               )
folder_name = '/'.join(filename.split('/')[:-1])+'/'+folder_name
print(folder_name)

if os.path.exists(folder_name):
    print("already exist")

trajectoryLength=15000
num_dummy_steps = 5000
#stall_dist=15    #This is the index distance between neighboring stall points 
print('distance between boundaries is %s'%stalldist)
birthArray = np.zeros(N1, dtype=np.double) + 0.1
deathArray = np.zeros(N1, dtype=np.double) + 1. / LIFETIME
stallDeathArray = np.zeros(N1, dtype=np.double) + 1 / LIFETIME
pauseArray = np.zeros(N1, dtype=np.double)

#stallList = [500-stalldist//2]
#stallList_c=[500+stalldist//2+1]#coupled stall list
#print(stallList,stallList_c)
#stallLeftArray = np.zeros(N1, dtype = np.double)
#stallRightARray = np.zeros(N1, dtype = np.double)
stallList=[15, 50,115, 175,519,670,830,1100,1180,1270,1350,1430]
stallList_c=[15, 50,115, 175,519,670,830,1100,1180,1270,1350,1430] #for layout_a
#stallList_c=[40,70,135,275,510,650,730,870,1185,1240,1330,1390] #for layout_b
stallRight=np.zeros(N1,dtype=np.double)
stallLeft=np.zeros(N1,dtype=np.double)
stallRight_c=np.zeros(N1,dtype=np.double)
stallLeft_c=np.zeros(N1,dtype=np.double)

for i in stallList:
    stallRight[i]=RSTALL
    stallLeft[i] = LSTALL
for j in stallList_c:
    stallRight_c[j]=RSTALL_C
    stallLeft_c[j] = LSTALL_C

stallRightARray=stallRight+stallRight_c
stallLeftArray=stallLeft+stallLeft_c

#print(stallRightARray,stallLeftArray)

LEFNum = N // SEPARATION
LEFTran = LEFTranslocatorDirectional(
    np.tile(birthArray,M), 
    np.tile(deathArray,M),
    np.tile(stallLeftArray,M),
    np.tile(stallRightARray,M),
    np.tile(pauseArray,M),
    np.tile(stallDeathArray,M),
    LEFNum)
folder = folder_name
if not os.path.exists(folder):
    os.mkdir(folder)

with h5py.File(folder_name+"/LEFPositions.h5", mode='w') as myfile:
    dset = myfile.create_dataset("positions", 
                                 shape=(trajectoryLength, LEFNum, 2), 
                                 dtype=np.int32, 
                                 compression="gzip")
    LEFTran.steps(num_dummy_steps)    
    bsteps = 50 
    bins = np.linspace(0, trajectoryLength, bsteps, dtype=int)
    for st,end in zip(bins[:-1], bins[1:]):
        cur = []
        for i in range(st, end):
            LEFTran.steps(1)
            LEFs = LEFTran.getLEFs()
            cur.append(np.array(LEFs).T)
        cur = np.array(cur)
        dset[st:end] = cur
    myfile.attrs["N"] = N
    myfile.attrs["LEFNum"] = LEFNum
#del dset

#print(LEFNum, N, N1)
#print(dset)

# -------defining parameters----------
#  -- reload loop extrusion parameters
myfile = h5py.File(folder_name+"/LEFPositions.h5", mode='r')

N = myfile.attrs["N"]
LEFNum = myfile.attrs["LEFNum"]
LEFpositions = myfile["positions"]
Nframes = LEFpositions.shape[0]
#myfile.close()
# initialize positions
data = grow_cubic(N, int(box) - 2)  # creates a compact conformation 
block = 0  # starting block 

# new parameters because some things changed 
saveEveryBlocks = 10   # save every 10 blocks (saving every block is now too much almost)
restartSimulationEveryBlocks = 100

# parameters for smc bonds
smcBondWiggleDist = 0.2
smcBondDist = 0.5

# assertions for easy managing code below 
assert (Nframes % restartSimulationEveryBlocks) == 0 
assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
simInitsTotal  = (Nframes) // restartSimulationEveryBlocks 


tstp = 70 # timestep for integrator in fs
tmst = 0.01 # thermostat for integrator

milker = polychrom.lib.extrusion.bondUpdater(LEFpositions)

reporter = HDF5Reporter(folder=folder, max_data_length=100, overwrite=True, blocks_only=False)

for iteration in range(simInitsTotal):
    # simulation parameters are defined below 
    a = Simulation(
            platform="cuda",
            integrator='langevin',  timestep=tstp, collision_rate=tmst,
            error_tol=0.01, 
            GPU = "0", 
            N = len(data),
            reporters=[reporter],
            PBCbox=[box, box, box],
            precision="mixed")  # timestep not necessary for variableLangevin
    ############################## New code ##############################
    a.set_data(data)  # loads a polymer, puts a center of mass at zero

    a.add_force(
        forcekits.polymer_chains(
            a,
            chains=[(0, None, 0)],

                # By default the library assumes you have one polymer chain
                # If you want to make it a ring, or more than one chain, use self.setChains
                # self.setChains([(0,50,1),(50,None,0)]) will set a 50-monomer ring and a chain from monomer 50 to the end

            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                'bondLength':1.0,
                'bondWiggleDistance':0.1, # Bond distance will fluctuate +- 0.05 on average
             },

            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                'k':1.5
                # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
                # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
            },

            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                'trunc':1.5, # this will let chains cross sometimes
                'radiusMult':1.05, # this is from old code
                #'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
            },
            except_bonds=True,
    ))
    # ------------ initializing milker; adding bonds ---------
    # copied from addBond
    kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
    bondDist = smcBondDist * a.length_scale

    activeParams = {"length":bondDist,"k":kbond}
    inactiveParams = {"length":bondDist, "k":0}
    milker.setParams(activeParams, inactiveParams)

    # this step actually puts all bonds in and sets first bonds to be what they should be
    milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                blocks=restartSimulationEveryBlocks)

    # If your simulation does not start, consider using energy minimization below
    if iteration==0:
        a.local_energy_minimization() 
    else:
        a._apply_forces()

    for i in range(restartSimulationEveryBlocks):        
       # print("restart#",i)
        if i % saveEveryBlocks == (saveEveryBlocks - 1):  
            a.do_block(steps=steps)
        else:
            a.integrator.step(steps)  # do steps without getting the positions from the GPU (faster)
        if i < restartSimulationEveryBlocks - 1: 
            curBonds, pastBonds = milker.step(a.context)  # this updates bonds. You can do something with bonds here
    data = a.get_data()  # save data and step, and delete the simulation
    del a

    reporter.blocks_only = True  # Write output hdf5-files only for blocks

    time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

reporter.dump_data()



myfile.close()



  
 
    




    
    
    
    
    
    
    
    
