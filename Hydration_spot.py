import matplotlib.pyplot as plt
import numpy as np
import mdshare
import mdtraj as md
#import os
import sys
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import alphashape
import timeit
import trimesh 
import argparse
import datetime
from itertools import groupby
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans 


######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
# Code structure: 
#    Initialization: read input paramters and load trajectrory 
# 1: Selecting the $\alpha$-Carbon atoms of the relevant part of the protein that encloses water
# 2: Building the convex hull surface with the coordinates from step 1
# 3: Collecting the position of the water molecules that lie within the convex hull for longer than a pre-defined lifetime  
# 4: Clustering the collected data to determine areas of high water density and their centers

# SAMPLE COMMAND FOR RUNNING THE SCRIPT
# python3 Hydration_spot.py --traj  TRAJ_short.trr --top system.pdb --alpha 0.5  --deltat 1 --min_time 3 --ncluster 20
#
#
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################


######################################################################################################################
######################################################################################################################
######################################################################################################################
#
#                                   AUXILIARY FUNCTIONS 
#
######################################################################################################################
######################################################################################################################
######################################################################################################################

def initialize ():

    parser = argparse.ArgumentParser(description='Calculate hydratation spot inside the protein')      
# files
    parser.add_argument('--traj',dest='trajectory',type=str,default='traj.trr',required=True,help='Trajectory: xtc trr gro pdb')
    parser.add_argument('--top',dest='topology',type=str,default='system.pdb',required=True,help='Structure: gro pdb ')
# parameters  
    parser.add_argument('--alpha',dest='alpha',type=float,required=True,help='Alpha parameter for bulding convex hull which depents on the distance between the selected atoms, eg. 0.5')
    parser.add_argument('--deltat',dest='time_step',type=float,required=True,help='Time step between input frames (in unit of time values: fs, ps, ns, us, ms, s)')
    parser.add_argument('--min_time',dest='min_time',type=float,required=True,help='Pre-defined lifetime for intrested water molecules')
    parser.add_argument('--ncluster',dest='ncluster',type=int,required=True,help='Number of cluster centers')
    parser.add_argument('--cluster',dest='cluster',type=bool,default=False,help='If True, code will just do clustering')

    args=parser.parse_args()
    trajectory = args.trajectory
    topology = args.topology
    alpha = args.alpha
    time_step= args.time_step 
    min_time = args.min_time
    ncluster = args.ncluster
    cluster = args.cluster
    return trajectory, topology, alpha, time_step, min_time, ncluster, cluster


def load_traj (trajectory,topol):

    begin_time = datetime.datetime.now()
# Loading Trajectory
    print(begin_time) 
    print ('------- Loading Trajectory ---------')
    traj = md.load(trajectory, top=topol)

# C_alpha and water selection 
    topology = traj.topology
    C_alpha=topology.select('name CA')
    water=topology.select("water and name O")
    Nwat = len(water)
    Nstep = len(traj)

    print ('Total water:', Nwat)
    print ('Total Step:', Nstep)

    print('Time:',datetime.datetime.now()-begin_time) 
    return traj, C_alpha, water, Nwat, Nstep 

def plot_alpha_shape(alpha_shape):
# plotting alpha_shape

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces, alpha=0.4)
    plt.show()

def plot_alpha_shape_with_point(alpha_shape,Win,Wout):

# plotting alpha_shape with waters 
    N_Win = len(Win)
    N_Wout = len(Wout)

    x_in = [Win[i][0] for i in range(N_Win)]
    y_in = [Win[i][1] for i in range(N_Win)]
    z_in = [Win[i][2] for i in range(N_Win)]

    x_out = [Wout[i][0] for i in range(N_Wout)]
    y_out = [Wout[i][1] for i in range(N_Wout)]
    z_out = [Wout[i][2] for i in range(N_Wout)]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces, alpha=0.4)
    ax.scatter(x_in,y_in,z_in, color='red')
    ax.scatter(x_out,y_out,z_out, color='blue', alpha=0.03)
    plt.show()


def check_water(Nwat,alpha_shape,XYZ_wat,debug):

# Check whether water id inside or outside of the convex hull
    Win = []
    Wout = []
    index = [0] * Nwat
    for i in range(Nwat):
        wat = XYZ_wat[i] # analysis point
#        if alpha_shape.contains([wat]) == True:
        if alpha_shape.contains([wat]):
           Win.append(wat)
           index[i] = 1
        else:
           Wout.append(wat)

    if debug:
       N_Win = len(Win)
       N_Wout = len(Wout)
       print ('Water inside:', N_Win)
       print ('Water outside:', N_Wout)
       plot_alpha_shape_with_point(alpha_shape, Win, Wout)

    return index 
 

def build_alpha_shape(alpha,traj,C_alpha,water):

# Build the convex hull and get index of water inside it

    begin_time = datetime.datetime.now()
    print ('------- Building Alpha Shape -------')
    alpha_input = alpha
    Nstep = len(traj) 
    Nwat = len(water) 
    W_index = []

    original_stdout = sys.stdout # Save a reference to the original standard outputoriginal_stdout
    with open('OUTPUT', 'w') as f:
         sys.stdout = f         # Change the standard output to the file we created.
       
         for i in range(Nstep):
             XYZ_wat=traj.xyz[i, water,:]   # water coordinates
             XYZ_CA=traj.xyz[i, C_alpha,:]  # Calpha coordinates 
       
             print ('----- Builgind Alpha shape step:',i)
             alpha_shape = alphashape.alphashape(XYZ_CA, alpha)
             watertight = alpha_shape.is_watertight
             if watertight:
                print('Mesh is watertight:',watertight,'alpha=',alpha)
             else:
                print('WARNING: Mesh is not watertight, decreasing alpha value')
             while not watertight:
                 alpha = alpha-0.01
                 alpha_shape = alphashape.alphashape(XYZ_CA, alpha)
                 watertight = alpha_shape.is_watertight
                 print('Mesh is watertight:',watertight,'alpha=',alpha)
       
#             plot_alpha_shape(alpha_shape)
             alpha= alpha_input   # set again to  initial value input parameter  
             W_index.append(check_water(Nwat,alpha_shape,XYZ_wat,False)) 

    f.close()
    sys.stdout = original_stdout 

    print('Time:',datetime.datetime.now()-begin_time) 
    return W_index 


def extract(lst,nelem):
# extract nth  element of each sublist in a list of lists

    return [item[nelem] for item in lst]


def monotoneRanges(water_residence, min_count):

    g = groupby(enumerate(water_residence), lambda x:x[1])
    l = [(x[0], list(x[1])) for x in g if x[0] == 1]    # check if elements are equal 1 
    output = [(x[0], len(x[1]), x[1][0][0]) for x in l]

    idx = [[output[i][2],output[i][2]+(output[i][1]-1)] for i in range(len(output)) if output[i][1] >= min_count]
    return idx


def lifetime(Nwat,min_time,time_step,water,W_index,traj):
# Life time analysis 

    begin_time = datetime.datetime.now()
    print ('------- Water Life Time Analysis ---')

    life_time = []
    Active_water = []
    Active_water_XYZ = []
    min_count = round(min_time/time_step)
    for j in range(Nwat):
            atom_idx = water[j] #  is the index of first O atom of water input parameter 
            check_lifetime = monotoneRanges(extract(W_index,j), min_count)
            life_time.append(check_lifetime) 
            if check_lifetime: 
               for frame in range(len(check_lifetime)):
                   Sframe_idx = check_lifetime[frame][0]
                   Eframe_idx = check_lifetime[frame][1]
                   for P in range(Sframe_idx,Eframe_idx+1):
                        Active_water_XYZ.append((tuple(traj.xyz[P, atom_idx,:])))

    XYZ_W_active = (np.array(Active_water_XYZ))*10. 

    with open('Active_water.xyz', 'w') as f:
       f.write("%d\n" % len(XYZ_W_active))
       f.write("\n" )
       for item in range(len(XYZ_W_active)):
                f.write("x %.4f\t %.4f\t %.4f \n" % (XYZ_W_active[item,0],XYZ_W_active[item,1],XYZ_W_active[item,2]))
       f.close()

    print('Time',datetime.datetime.now()-begin_time) 
    return XYZ_W_active


def read_active_water(filename):
    """Read XYZ file and return atom names and coordinates

    Args:
        filename:  Name of xyz data file

    Returns:
        coords: Cartesian coordinates for every frame.
    """
    print ('------- Reading Active_water.xyz ---')
    coors = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                natm = int(line)  # Read number of atoms
                next(f)     # Skip over comments
                for i in range(natm):
                    line = next(f).split()
                    coors.append(
                        [float(line[1]), float(line[2]), float(line[3])])
            except (TypeError, IOError, IndexError, StopIteration):
                raise ValueError('Incorrect XYZ file format')
    coors = np.array(coors)
    return coors


def clustering(Ncluster,XYZ_W_active, cluster):

    begin_time = datetime.datetime.now()
    print ('------- KMeans clustering  ---------')
# Clustering 

    kmeans = KMeans(n_clusters=Ncluster, random_state=0).fit(XYZ_W_active)
    center= kmeans.cluster_centers_
 
    with open('Water_centroid.xyz', 'w') as f:
        f.write("%d\n" % len(center))
        f.write("\n" )
        for itemm in range(len(center)):
                f.write("C %.4f\t %.4f\t %.4f \n" % (center[itemm,0],center[itemm,1],center[itemm,2]))
    f.close()

    print('Time',datetime.datetime.now()-begin_time) 
    print(datetime.datetime.now()) 
    return 


def main():

    trajectory, topology, alpha, time_step, min_time, ncluster, cluster = initialize()       # initialization 
    if not cluster:
       traj, C_alpha, water, Nwat, Nstep = load_traj(trajectory,topology)                                  # load trajectory 
       W_index = build_alpha_shape(alpha,traj,C_alpha,water)                                               # build alpha shape           
       XYZ_W_active = lifetime(Nwat,min_time,time_step,water,W_index,traj)                           # water life time analysis 
       clustering(ncluster,XYZ_W_active, cluster)                                                          # kmeans clustering 
    else:
       XYZ_W_active = read_active_water('Active_water.xyz')                                                # read from Active_water_XYZ
       clustering(ncluster,XYZ_W_active, cluster)                                                          # kmeans clustering 

######################################################################################################################

if __name__ == '__main__':
   main()


