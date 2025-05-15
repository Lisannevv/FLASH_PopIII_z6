import numpy as np
import matplotlib.pyplot as plt
import yt
from yt.units import *
from scipy.stats import binned_statistic
import h5py
import matplotlib.colors as colors
import pandas as pd
import cmasher
import matplotlib
import os
import shutil
import sys
try:
    from yt.visualization.volume_rendering.api import off_axis_projection
    from yt.visualization.fixed_resolution import FixedResolutionBuffer
except: 
    print("Import Error: Please install yt using [pip install yt] in terminal.")

try:
    from sklearn.decomposition import PCA
except:
    print("Import Error: Please install sklearn using [pip install scikit-learn] in terminal.")
    sys.exit(1)

#----------------------------------------------------------------------------------------------------

print('\n')
print('Hello! This Python script will create projections or slices of your FLASH star formation simulation data, \
where the basis vectors are those of the star forming disk, and then save these as 2D numpy files for you to plot as you wish.')
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

def _numdens(field,data):
    return((data["density"]/mass_hydrogen_cgs)*(data['h   ']/1 + data['h2  ']/2 + 0.2492/4 + data['hp  ']/1 + 
                                               data['hd  ']/3 + data['d   ']/2))

def _plasma_beta(field,data):
    return(8*np.pi*data[("gas","pressure")]/data[("gas","magnetic_field_strength")]**2)

def _jeans_length(field,data):
    kB = YTQuantity(1.3807e-16, 'g*cm**2/K/s**2')
    G = YTQuantity(6.6743e-8, 'cm**3/g/s**2')
    mu = data[('gas','mean_molecular_weight')]
    dens = data[('flash','dens')].in_units('g/cm**3')
    T = data[('flash','temp')].in_units('K')
    return np.sqrt(15*kB*T/(4*np.pi*G*mu*mass_hydrogen_cgs*dens))

def _jeans_number(field,data):
    kB = YTQuantity(1.3807e-16, 'g*cm**2/K/s**2')
    G = YTQuantity(6.6743e-8, 'cm**3/g/s**2')
    mu = data[('gas','mean_molecular_weight')]
    dens = data[('flash','dens')].in_units('g/cm**3')
    T = data[('flash','temp')].in_units('K')
    Lj = np.sqrt(15*kB*T/(4*np.pi*G*mu*mass_hydrogen_cgs*dens))
    dx = data[('flash','dx')]
    return Lj/dx

ds = yt.load(filename)
dd = ds.all_data()
ds.add_field(("flash", "numdens"),  function=_numdens, units="cm**-3",sampling_type="cell")
ds.add_field(("flash", "plasma_beta"),  function=_plasma_beta, units="dimensionless",sampling_type="cell")
ds.add_field(("flash", "jeans_length"),  function=_jeans_length, units="cm",sampling_type="cell")
ds.add_field(("flash", "jeans_number"),  function=_jeans_number, units="dimensionless",sampling_type="cell")


#------------------------------------------------------------------------------------------------------

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

def L_vector(ds):
    #Define cylindrical selection of star forming disk using PCA angular momentum vector
    sinks_com = com(ds,None)
    sinks_vcom = vcom(ds)
    disk_normal_pca = disk_normal_PCA(ds)
    group = ds.disk(sinks_com, disk_normal_pca, (0.01, "pc"), (250, "au"))

    #Gas contribution
    x = np.array(group[("flash", "x")])-sinks_com[0]
    y = np.array(group[("flash", "y")])-sinks_com[1]
    z = np.array(group[("flash", "z")])-sinks_com[2]
    r = np.vstack((x, y, z)).T
    m = np.array(group[('gas', 'cell_mass')])
    vx = np.array(group[('gas', 'velocity_x')])-sinks_vcom[0]
    vy = np.array(group[('gas', 'velocity_y')])-sinks_vcom[1]
    vz = np.array(group[('gas', 'velocity_z')])-sinks_vcom[2]
    v = np.vstack((vx, vy, vz)).T

    #Sink particles contribution 
    sink_x = np.array(dd['all','particle_position_x'])-sinks_com[0]
    sink_y = np.array(dd['all','particle_position_y'])-sinks_com[1]
    sink_z = np.array(dd['all','particle_position_z'])-sinks_com[2]
    sink_r =  np.vstack((sink_x, sink_y, sink_z)).T
    sink_m = np.array(dd[('all', 'particle_mass')])
    sink_vx = np.array(dd['all', 'particle_velocity_x'])-sinks_vcom[0] 
    sink_vy = np.array(dd['all', 'particle_velocity_y'])-sinks_vcom[1]
    sink_vz = np.array(dd['all', 'particle_velocity_z'])-sinks_vcom[2]
    sink_v = np.vstack((sink_vx, sink_vy, sink_vz)).T

    #Calculate and return L
    L_gas =  np.cross(r, v)*m[:, None]
    L_sinks = np.cross(sink_r,sink_v)*sink_m[:,None]
    Lvec = np.sum(L_gas,axis=0)+np.sum(L_sinks,axis=0)
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

def proj_2darr_offaxis(ds, field, normal_vector,north_vector):
    #width = (0.01, "pc") 
    width = (0.01*pc).to("cm").value
    res = [1000, 1000]    
    center = com(ds, None)
    weight = ("flash", "dens")
    if field == "plasma_beta":
        field = ("flash","plasma_beta")
    proj = off_axis_projection(ds,center=center,normal_vector=normal_vector,north_vector=north_vector,width=width,resolution=res,item=field,weight=weight)
    return np.array(proj)

def slice_2darr_offaxis(ds,field,disk_normal,north_vector):
    wi = 0.01
    center = com(ds, None)
    slc = yt.OffAxisSlicePlot(ds, disk_normal, field, center=center, width=(wi, "pc"),north_vector=north_vector)
    arr = slc.plots[field].image._A
    return arr

def get_field():
    print('What field do you want to make a rendering of? Please provide a tuple with the group ("gas","flash","io","index", etc.) and the field name ("temp", "numdens", "h2  ", etc.).')
    print('Some examples are ("flash","temp") and ("gas","magnetic_field_strength").')
    while True:
        field = input('Enter: ')
        try:
            while True:
                field = eval(field) # Turn input into tuple
                if field in ds.derived_field_list:
                    print(f"The field {field} exists in the dataset!")
                    break
                else: 
                    field = input(f"The field {field} does not exist in the dataset. Try typing the field again:")
            break        
                    
        except Exception as e:
            print(f"There was an error interpreting the field input, please submit the desired field again. Make sure you input a tuple. Error: {e}")
    return field

def get_axis():
    print('What axis would you like for the projection to be made along?')
    while True:
        axis = input('Choose 0,1, or 2 with 2 being L, or type "all" if all axes are desired: ')
        if axis.isdigit() and int(axis) in [0,1,2]:
            print('Input correct.')
            return int(axis)
        elif axis == "all":
            print('Input correct.')
            return axis
        else:
            print('Input incorrect, please try again.')
    

def make_projections(field,axis):
    basis = matrix_Lbasis(ds)
    field_name = field[1]
    if axis == 2:
        print(f'Creating a projection of the desired field along axis {axis}..')
        vec = basis[2]
        north = basis[1]
        proj = np.transpose(proj_2darr_offaxis(ds, field[1], vec, north))
        print('Saving..')
        np.save(f'proj_{axis}_'+str(field_name)+'_'+str(step)+'.npy',proj)
    elif axis in [0,1]:
        print(f'Creating a projection of the desired field along axis {axis}..')
        vec = basis[2]
        vec = basis[axis]
        north = basis[2]
        if axis == 0:
            proj = np.transpose(proj_2darr_offaxis(ds, field[1], vec, north))
        else:
            proj = np.flip(np.transpose(proj_2darr_offaxis(ds, field[1], vec, north)),axis=1)
        print('Saving..')
        np.save(f'proj_{axis}_'+str(field_name)+'_'+str(step)+'.npy',proj)
    elif axis == "all":
        for axis in [0,1,2]:
            print(f'Creating a projection of the desired field along axis {axis}..')
            vec = basis[2]
            north = basis[1]
            if axis == 1:
                proj = np.flip(np.transpose(proj_2darr_offaxis(ds, field[1], vec, north)),axis=1)
            else:
                proj = np.transpose(proj_2darr_offaxis(ds, field[1], vec, north))
            print("Saving..")
            np.save(f'proj_{axis}_'+str(field_name)+'_'+str(step)+'.npy',proj)
    return

def make_slices(field,axis):
    basis = matrix_Lbasis(ds)
    field_name = field[1]
    if axis == 2:
        print(f'Creating a slice of the desired field along axis {axis}..')
        vec = basis[2]
        north = basis[1]
        slice = slice_2darr_offaxis(ds, field, vec, north)
        print('Saving..')
        np.savez(f"slice_{axis}_{field_name}_{step}.npz", data=slice)
    elif axis in [0,1]:
        print(f'Creating a slice of the desired field along axis {axis}..')
        vec = basis[axis]
        north = basis[2]
        if axis == 0:
            slice = slice_2darr_offaxis(ds, field, vec, north)
        else:
            slice = np.flip(slice_2darr_offaxis(ds, field, vec, north),axis=1)
        print('Saving..')
        np.savez(f"slice_{axis}_{field_name}_{step}.npz", data=slice)
    elif axis == "all":
        for axis in [0,1,2]:
            print(f'Creating a slice of the desired field along axis {axis}..')
            if axis == 1:
                vec = basis[axis]
                north = basis[2]
                slice = np.flip(slice_2darr_offaxis(ds, field, vec, north),axis=1)
            elif axis == 0:
                vec = basis[axis]
                north = basis[2]
                slice = slice_2darr_offaxis(ds, field, vec, north)
            else:
                vec = basis[axis]
                north = basis[1]
                slice = slice_2darr_offaxis(ds, field, vec, north)
            print("Saving..")
            np.savez(f"slice_{axis}_{field_name}_{step}.npz", data=slice)
    return

def make_rendering():
    
    while True:
        arrtype = input('Do you want to create a slice or a projection? Type "slice" or "proj": ')
        if arrtype == 'slice':
            field = get_field()
            axis = get_axis()
            if axis == 'all':
                make_slices(field,axis)
            else:
                make_slices(field,int(axis))
            break
        elif arrtype == 'proj':
            field = get_field()
            axis = get_axis()
            if axis == 'all':
                make_projections(field,axis)
            else:
                make_projections(field,int(axis))
            break
        else:
            print('Incorrect input. Please try again.')
    print('Complete!')
    return

def sink_positions(dataset): # data = dd
    w = 0.01
    # Here we calculate the positions of the stars in pixels given the com
    data = dataset.all_data()
    xpos = np.array(data[('io', 'particle_posx')]) #in cm
    ypos = np.array(data[('io', 'particle_posy')]) #1 cm = 3,24078e-19 pc
    zpos = np.array(data[('io', 'particle_posz')])
    c = com(dataset,None)
    pix = w/1000 * 3.08567758128e18 # cm
    xpos = (xpos-c[0])/pix 
    ypos = (ypos-c[1])/pix
    zpos = (zpos-c[2])/pix 
    pos = np.array([xpos, ypos, zpos])
    mask = (pos >= -500) & (pos <= 500)
    valid_mask = mask.all(axis=0)  
    filtered_pos = pos[:, valid_mask] *2.06265 # AU/pix
    return filtered_pos


#---------------------------------------------------------------------------------------------------------


print('Before we go further, would you like to save the sink positions at this step?')
while True:
    pos = input('Type "y" or "n": ')
    if pos == 'y':
        print('Calculating sink positions..')
        matrix = matrix_Lbasis(ds)
        matrix[:,0]=[-matrix[0,0],-matrix[1,0],-matrix[2,0]]
        sinkpos = np.dot(matrix,sink_positions(ds))
        print('Saving..')
        np.save('sinkpos_py_'+str(step)+'.npy',sinkpos)
        print('Positions saved. Proceeding..')
        break
    elif pos == 'n':
        print('Okay! Proceeding..')
        break
    else:
        print("Incorrect input, please try again.")

print('\n')
make_rendering()

while True:
    print('\n')
    print('Would you like to make another rendering?')
    answer = input('Type "y" or "n": ')

    if answer == 'y':
        make_rendering()
    elif answer == 'n':
        print('Okay! The script will now close.')
        sys.exit(1)
    else:
        print('Incorrect input, please try again.')


