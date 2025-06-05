import numpy as np
import matplotlib.pyplot as plt
import yt
from yt.units import *
import h5py
import matplotlib.colors as colors
import pandas as pd
import sys
import cmasher
import os
import shutil
try:
    from sklearn.decomposition import PCA
except:
    print("Import Error: Please install sklearn using [pip install scikit-learn] in terminal.")
    sys.exit(1)
from scipy.stats import binned_statistic
from yt.utilities.exceptions import YTFieldNotFound

"""
This Python file produces radial profiles along the R and z coordinate, where the original xyz coordinates from 
the YT dataframe are transformed into a new basis where the angular momentum vector L acts as the z-coordinate vector,
and the radial coordinate is calculated from the transformed x and y basis vectors. This way, we have a cylindrical
coordinate system that represents the star forming disk.
The output of this file is 8 radial profiles: both a profile along the radial- and z-direction for the YT fields for
number density, temperature, H2 mass fraction and H+ mass fraction.
The radial profiles are saved as 8 .npy files.

The script can be used by entering 'python RadialProfiles.py' in the terminal.
"""
#---------------------------------------------------------------------------------------------------------------------------------------------------------

print('\n')
print('Greetings, esteemed user of this humble python script.')
print('\n')
while True:
    digits = input('Are your filenames of the form: Krome_Chem_rmhd_hdf5_plt_cnt_(step) ? Type either "y" or "n": ')

    if digits == 'y': 
        step = input('Please type a 4-digit number, for example: 0480. This is the file-step of the plt/part file you wish for me to use. Please enter now: ')
        while True:
            if len(step) == 4 and step.isdigit():
                print('Thank you!')
                filename = f'Krome_Chem_rmhd_hdf5_plt_cnt_{step}' 
                print('Checking if the file exists..')
                if os.path.exists(filename):
                    print(f'It "{filename}" exists!')
                    break
                else:
                    print(f'The file "{filename}" could not be found. Make sure you are in the correct folder. The script will now close.')
                    sys.exit(1)
            else:
                step = input('That was not a four-digit number. Try again:')
        break

    elif digits == 'n':
        while True:
            filename = input('Please type the filename: ')
            print(f'Thank you!')
            print('Checking if the file exists..')
            if os.path.exists(filename):
                print(f'It "{filename}" exists!')
                while True:
                    step = input('Please type a 4-digit number, for example: 0480. This is the file-step of the file you just entered. Please enter the number: ')
                    if len(step) == 4 and step.isdigit():
                        print('Input is of correct format, the script will now proceed.')
                        break
                    else:
                        step = input('That was not a four-digit number. Try again:')
                break
            else:
                print(f'The file "{filename}" could not be found. Please enter the filename again.')
        break
    else:
        print('Neither y or n was entered. Try again.')
    



# Make the dataset and add number density and plasma beta fields

print('\n')
print('Loading data..')

#---------------------------------------------------------------------------------------------------------------------------------------------------------

"""
Functions:
"""

def com(dataset, unit): #calculates com at this timestep in cm or another desired unit
    #Here, data is the dd variable, where dd = ds.all_data()
    data = dataset.all_data()
    cx=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_position_x'])/np.sum(data['all', 'particle_mass']))
    cy=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_position_y'])/np.sum(data['all', 'particle_mass']))
    cz=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_position_z'])/np.sum(data['all', 'particle_mass']))
    if unit == None: 
        return np.array([cx,cy,cz],dtype=float)
    else:
        return np.array([cx,cy,cz])/(1.0*unit).to("cm").value

def vcom(dataset): #calculates velocity com of sink particles at this timestep
    #Here, data is the dd variable
    data = dataset.all_data()
    cx=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_velocity_x'])/np.sum(data['all', 'particle_mass']))
    cy=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_velocity_y'])/np.sum(data['all', 'particle_mass']))
    cz=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_velocity_z'])/np.sum(data['all', 'particle_mass']))
    return np.array([cx,cy,cz])

def disk_normal_PCA(ds): #This gives an indication of the direction of L using Principal Component Analysis.
    dd = ds.all_data()
    mask_disk = (dd[("flash", "numdens")] > 2e10) & (dd[("flash", "temp")] < 1.1e3)
    x_disk = np.array(dd[("flash", "x")][mask_disk])
    y_disk = np.array(dd[("flash", "y")][mask_disk])
    z_disk = np.array(dd[("flash", "z")][mask_disk])
    sinks_com = com(ds,None)
    positions = np.vstack((x_disk, y_disk, z_disk)).T  
    pca = PCA(n_components=3)
    pca.fit(positions - sinks_com)
    disk_norm = pca.components_[-1]  
    disk_norm /= np.linalg.norm(disk_norm)
    #print('pca normal =',disk_norm)
    return disk_norm

def _numdens(field,data):
    return((data["density"]/mass_hydrogen_cgs)*(data['h   ']/1 + data['h2  ']/2 + 0.2492/4 + data['hp  ']/1 + 
                                               data['hd  ']/3 + data['d   ']/2))

# We might need a reference file.. Let's go through another round of while loops and try except statements.. yay!
print('\n')
print(f'Loading data for file {step}..')
ds = yt.load(filename)
dd = ds.all_data()
print('Data loaded.')
nosinks = False

try:
    sinks_com = com(ds, None)
except YTFieldNotFound:
    nosinks = True
    print('It seems your current file does not have any sink particles, so a reference file is needed. \
          Please provide the first plt_cnt file where the first star has appeared.')
    while True:
        if digits == 'y':
            ref_step = input('Please provide the 4-digit step of this reference file.')
            while True:
                if len(ref_step) == 4 and ref_step.isdigit():
                    print('Thank you!')
                    filename = f'Krome_Chem_rmhd_hdf5_plt_cnt_{ref_step}' 
                    print('Checking if the file exists..')
                    if os.path.exists(filename):
                        print(f'It "{filename}" exists!')
                        ref_ds = yt.load(filename)
                        ref_ds.add_field(("flash", "numdens"),  function=_numdens, units="cm**-3",sampling_type="cell")
                        try:
                            sinks_com = com(ref_ds, None)
                            break
                        except YTFieldNotFound:
                            print('It seems this file does not have any sink particles. \
                                Please provide the first plt_cnt file where the first star has appeared.')
                        
                    else:
                        print(f'The file "{filename}" could not be found. Make sure you are in the correct folder. The script will now close.')
                        sys.exit(1)
                else:
                    step = input('That was not a four-digit number. Try again:')
            break
        elif digits == 'n':
            while True:
                filename = input('Please type the filename: ')
                print(f'Thank you!')
                print('Checking if the file exists..')
                if os.path.exists(filename):
                    print(f'It "{filename}" exists!')
                    ref_ds = yt.load(filename)
                    ref_ds.add_field(("flash", "numdens"),  function=_numdens, units="cm**-3",sampling_type="cell")
                    try:
                        sinks_com = com(ref_ds, None)
                        break
                    except YTFieldNotFound:
                        print('It seems this file does not have any sink particles. \
                            Please provide the first plt_cnt file where the first star has appeared.')
                else:
                    print('It seems this file does not exist. Please enter again.')
            break
    ref_com = com(ref_ds,None)
    ref_vcom = vcom(ref_ds)
    ref_pca = disk_normal_PCA(ref_ds)


def L_vector(ds):
    dd = ds.all_data()
    nosinkies = False

    #define cylindrical selection
    try:
        # Try everything that might access sink fields
        sinks_com = com(ds, None)
        sinks_vcom = vcom(ds)
        disk_normal_pca = disk_normal_PCA(ds)
    except:
        # Use fallback values if sinks are not present
        sinks_com = ref_com
        sinks_vcom = ref_vcom
        disk_normal_pca = ref_pca  # You'll need to define this
        nosinkies = True
    group = ds.disk(sinks_com, disk_normal_pca, (0.01, "pc"), (250, "au"))

    #gas contribution
    x = np.array(group[("flash", "x")])-sinks_com[0]
    y = np.array(group[("flash", "y")])-sinks_com[1]
    z = np.array(group[("flash", "z")])-sinks_com[2]
    r = np.vstack((x, y, z)).T
    m = np.array(group[('gas', 'cell_mass')])
    vx = np.array(group[('gas', 'velocity_x')])-sinks_vcom[0]
    vy = np.array(group[('gas', 'velocity_y')])-sinks_vcom[1]
    vz = np.array(group[('gas', 'velocity_z')])-sinks_vcom[2]
    v = np.vstack((vx, vy, vz)).T

    #sink contribution 
    if nosinkies == False:
        sink_x = np.array(dd['all','particle_position_x'])-sinks_com[0]
        sink_y = np.array(dd['all','particle_position_y'])-sinks_com[1]
        sink_z = np.array(dd['all','particle_position_z'])-sinks_com[2]
        sink_r =  np.vstack((sink_x, sink_y, sink_z)).T
        sink_m = np.array(dd[('all', 'particle_mass')])
        sink_vx = np.array(dd['all', 'particle_velocity_x'])-sinks_vcom[0] 
        sink_vy = np.array(dd['all', 'particle_velocity_y'])-sinks_vcom[1]
        sink_vz = np.array(dd['all', 'particle_velocity_z'])-sinks_vcom[2]
        sink_v = np.vstack((sink_vx, sink_vy, sink_vz)).T
        L_sinks = np.cross(sink_r,sink_v)*sink_m[:,None]
        L_gas =  np.cross(r, v)*m[:, None]
        Lvec = np.sum(L_gas,axis=0)+np.sum(L_sinks,axis=0)
        Lvec /= np.linalg.norm(Lvec)
    else:
        L_gas =  np.cross(r, v)*m[:, None]
        Lvec = np.sum(L_gas,axis=0)
        Lvec /= np.linalg.norm(Lvec)
    
    return Lvec

def matrix_Lbasis(ds): #Creates the new basis with angular momentum vector as z.
    L = L_vector(ds)
    random_vector = np.array([0, 1, 0])  
    b1 = random_vector - np.dot(random_vector, L) * L
    b2 = np.cross(L, b1)
    b2 /= np.linalg.norm(b2) #'x-axis'
    #print(np.dot(L,b1))
    #print(np.dot(L,b2))

    return np.array([b2,b1,L])

field_units = pd.DataFrame([["numdens", None,r"$n_{\rm{H}} \rm{(cm}^{-3}\rm{)}$", "log"],["temp", "K",r"$T\rm{(K)}$", None],\
                            ["magnetic_field_magnitude","G","B (G)", "log"],["hp  ",None, r"$X_{H^{+}}$", "log"],[ "h2  ",None,r"$X_{H_2}$", "log"]], columns = ["field", "unit","label","scale"])

def Radial_Profile(ds,field):
    #Create disk and yt profile
    disk_radius = 0.005 #0.005 pc ~= 1000 AU
    if nosinks == True:
        sinks_com = ref_com
    else:
        sinks_com = com(ds, None)
    L = L_vector(ds)
    disk = ds.disk(sinks_com, L, (disk_radius, "pc"), (200, "au"))
    unit = np.array(field_units[field_units["field"] == field]["unit"])[0]
    if unit != None:
        values = disk[field].to(unit).v
    else:
        values = disk[field].v
    cell_mass = disk["cell_mass"].v/1.989e33 #solar mass

    """
    First, we transform positions into disk coordinate system and extract R values.
    Without this procedure, you would take: radii = disk["radius"].to("au").v
    """
    sinks_com_yt = yt.YTArray(sinks_com, 'cm').to('au')
    x = np.array(disk['x'].to('au').v) - sinks_com_yt[0].v
    y = disk['y'].to('au').v - sinks_com_yt[1].v
    z = disk['z'].to('au').v - sinks_com_yt[2].v
    pos = np.vstack([x,y,z])
    A = matrix_Lbasis(ds)
    pos_Lcoords = np.dot(A,pos)
    radii = np.sqrt(pos_Lcoords[0,:]**2+pos_Lcoords[1,:]**2) 
    
    #Prepare data, ascending order for radius
    data_for_binning = pd.DataFrame({'R': radii,'val':values*cell_mass, 'mass':cell_mass})
    sorted_data = data_for_binning.sort_values(by='R')
    radii = np.array(sorted_data['R'])
    massw_vals = np.array(sorted_data['val'])
    cell_mass = np.array(sorted_data['mass'])

    #Create mass weighted bin values using scipy binned statistics
    bins = 60
    #bin_edges = np.logspace(np.log10(radii.min()), np.log10(radii.max()), bins + 1)
    bin_edges = np.linspace(radii.min(), radii.max(), bins + 1)
    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    weighted_values = binned_statistic(radii, massw_vals, bins=bin_edges, statistic = "sum")[0]
    mass_per_bin = binned_statistic(radii, cell_mass, statistic = "sum", bins=bin_edges)[0]
    values_massweighted = weighted_values/mass_per_bin #oops, there are a few nans
    values_massweighted = np.nan_to_num(values_massweighted, nan=0.0)
    n = values_massweighted
    return r, n

def Radial_Profile_Zaxis(ds,field): 
    #Create disk and yt profile
    disk_radius = 0.005 #0.005 pc ~= 1000 AU
    if nosinks == True:
        sinks_com = ref_com
    else:
        sinks_com = com(ds, None)
    L = L_vector(ds)
    disk = ds.disk(sinks_com, L, (disk_radius, "pc"), (300, "au"))

    #First, we transform positions into disk coordinate system
    sinks_com_yt = yt.YTArray(sinks_com, 'cm').to('au')
    x = np.array(disk['x'].to('au').v) - sinks_com_yt[0].v
    y = disk['y'].to('au').v - sinks_com_yt[1].v
    z = disk['z'].to('au').v - sinks_com_yt[2].v
    pos = np.vstack([x,y,z])
    #print(np.shape(pos))
    A = matrix_Lbasis(ds)
    pos_Lcoords = np.dot(A,pos)
    #print(np.shape(pos_Lcoords))
    radii = pos_Lcoords[2,:]

    unit = np.array(field_units[field_units["field"] == field]["unit"])[0]
    if unit != None:
        values = disk[field].to(unit).v
    else:
        values = disk[field].v
    cell_mass = disk["cell_mass"].v/1.989e33 #solar mass
    
    #Prepare data, ascending order for radius
    data_for_binning = pd.DataFrame({'R': radii,'val':values*cell_mass, 'mass':cell_mass})
    sorted_data = data_for_binning.sort_values(by='R')
    radii = np.array(sorted_data['R'])
    #print(radii[0:15])
    massw_vals = np.array(sorted_data['val'])
    cell_mass = np.array(sorted_data['mass'])

    #Create mass weighted bin values using scipy binned statistics
    bins = 60
    bin_edges = np.linspace(radii.min(), radii.max(), bins + 1)
    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    weighted_values = binned_statistic(radii, massw_vals, bins=bin_edges, statistic = "sum")[0]
    mass_per_bin = binned_statistic(radii, cell_mass, statistic = "sum", bins=bin_edges)[0]
    values_massweighted = weighted_values/mass_per_bin #oops, there are a few nans
    values_massweighted = np.nan_to_num(values_massweighted, nan=0.0)
    n = values_massweighted
    
    return r, n



"""
Script: 
"""


mass_hydrogen_cgs = yt.units.mass_hydrogen_cgs
ds.add_field(("flash", "numdens"),  function=_numdens, units="cm**-3",sampling_type="cell")

for field in field_units["field"]:
    print("Creating radial profiles for field ["+field+"]..")
    R,n_R = Radial_Profile(ds,field)
    Rprof = np.vstack((R,n_R)).T
    Z,n_Z = Radial_Profile_Zaxis(ds,field)
    Zprof = np.vstack((Z,n_Z)).T
    np.save(f'Rprof_{step}_{field}.npy',Rprof)
    np.save(f'Zprof_{step}_{field}.npy',Zprof)

print('Complete!')
