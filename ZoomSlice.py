import numpy as np
import matplotlib.pyplot as plt
import yt
from yt.units import *
import h5py
import matplotlib.colors as colors
import pandas as pd
import cmasher
import os
import shutil
from sklearn.decomposition import PCA
from scipy.stats import binned_statistic
from yt.visualization.volume_rendering.api import off_axis_projection
from yt.visualization.fixed_resolution import FixedResolutionBuffer
from matplotlib.lines import Line2D


print('\n')
print('Hello! This Python script will create slices centered at a chosen point of your FLASH star formation simulation data, \
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

frb_res = 2048
w = 0.002

def com(dataset, unit): #calculates com at this timestep in cm or another desired unit
    # Here, data is the dd variable
    data = dataset.all_data()
    cx=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_position_x'])/np.sum(data['all', 'particle_mass']))
    cy=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_position_y'])/np.sum(data['all', 'particle_mass']))
    cz=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_position_z'])/np.sum(data['all', 'particle_mass']))
    if unit == None: 
        return np.array([cx,cy,cz],dtype=float)
    else:
        return np.array([cx,cy,cz])/(1.0*unit).to("cm").value
    
def vcom(dataset): #calculates com at this timestep
    # Here, data is the dd variable
    data = dataset.all_data()
    cx=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_velocity_x'])/np.sum(data['all', 'particle_mass']))
    cy=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_velocity_y'])/np.sum(data['all', 'particle_mass']))
    cz=np.array(np.sum(data['all', 'particle_mass']*data['all', 'particle_velocity_z'])/np.sum(data['all', 'particle_mass']))
    return np.array([cx,cy,cz])


def com_dd(data):
    mass = data['all', 'particle_mass']
    x = data['all', 'particle_position_x']
    y = data['all', 'particle_position_y']
    z = data['all', 'particle_position_z']
    cx = (mass * x).sum() / mass.sum()
    cy = (mass * y).sum() / mass.sum()
    cz = (mass * z).sum() / mass.sum()
    return yt.YTArray([cx, cy, cz])

def vcom_dd(data):
    mass = data['all', 'particle_mass']
    vx = data['all', 'particle_velocity_x']
    vy = data['all', 'particle_velocity_y']
    vz = data['all', 'particle_velocity_z']
    cx = (mass * vx).sum() / mass.sum()
    cy = (mass * vy).sum() / mass.sum()
    cz = (mass * vz).sum() / mass.sum()
    return yt.YTArray([cx, cy, cz])

def offcenter_sink_positions(dataset,c,w): # data = dd
    # Here we calculate the positions of the stars in pixels given the com
    data = dataset.all_data()
    xpos = np.array(data[('io', 'particle_posx')]) #in cm
    ypos = np.array(data[('io', 'particle_posy')]) #1 cm = 3,24078e-19 pc
    zpos = np.array(data[('io', 'particle_posz')])
    #c = com(dataset,None)
    pix = w/1000 * 3.08567758128e18 # cm
    xpos = (xpos-c[0])/pix 
    ypos = (ypos-c[1])/pix
    zpos = (zpos-c[2])/pix 
    pos = np.array([xpos, ypos, zpos])
    mask = (pos >= -500) & (pos <= 500)
    valid_mask = mask.all(axis=0)  
    filtered_pos = pos[:, valid_mask] *2.06265 # AU/pix
    return filtered_pos

def zoom_slice_disk_arr(ds,field,wi,c):
    basis = matrix_Lbasis(ds)
    disk_normal = basis[2]
    north_vector = basis[1]
    center = c
    # Create slc object and make array
    if field == 'magnetic_field_strength':
        slc = yt.OffAxisSlicePlot(ds, disk_normal, ("gas", field), center=center, width=(wi, "pc"),north_vector=north_vector)
        slc.set_buff_size((frb_res, frb_res))
        frb = slc.frb 
        #arr = slc.plots[("gas", field)].image._A
        arr = frb[("flash", field)].value
    else:
        slc = yt.OffAxisSlicePlot(ds, disk_normal, ("flash", field), center=center, width=(wi, "pc"),north_vector=north_vector)
        slc.set_buff_size((frb_res, frb_res))
        frb = slc.frb 
        #arr = slc.plots[("flash", field)].image._A
        arr = frb[("flash", field)].value
    return arr

def frb_2d(dataset,proj, field_name):
    w = 0.01
    width = (w, "pc") 
    res = [1000, 1000]  #create an image with 1000x1000 pixels
    c = com(dataset,None)
    frb = proj.to_frb(width, res, center=c)
    return np.array(frb["flash", field_name])

def disk_normal_PCA(ds):
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
    dd = ds.all_data()
    nosinks = False

    #define cylindrical selection
    
    # Try *everything* that might access sink fields
    sinks_com = com(ds, None)
    sinks_vcom = vcom(ds)
    disk_normal_pca = disk_normal_PCA(ds)
    
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
    if nosinks == False:
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

def matrix_Lbasis(ds):
    L = L_vector(ds)
    random_vector = np.array([0, 1, 0])  
    b1 = random_vector - np.dot(random_vector, L) * L
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(L, b1)
    b2 /= -np.linalg.norm(b2) #'x-axis'
    #print(np.dot(L,b1))
    #print(np.dot(L,b2))

    return np.array([b2,b1,L])

def make_slices(ds,c,compare):
    
    fields = ["temp","numdens","v_rad","v_phi","v_z","h2  ","hp  ","h   "]
    for j in range(len(fields)):
        print(f'Creating a slice of: {fields[j]}')
        im = zoom_slice_disk_arr(ds,fields[j],w,c)
        if compare == False:
            np.save(f'slice_{step}_{fields[j]}.npy',im)
        else:
            np.save(f'slice_{step}_{fields[j]}_z6nosf2.npy',im)
    return


# Loading the data

print('\n')
print('Loading data..')

ds = yt.load(filename)
dd = ds.all_data()

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

sinkcom = com_dd(dd)
v_com = vcom_dd(dd)

def _v_r(field, data):
    x = data['x'] - sinkcom[0]
    y = data['y'] - sinkcom[1]
    phi = np.arctan2(y,x)

    # Get velocities and correct for centre velocity
    velx = data['velocity_x']
    vely = data['velocity_y']
    velx = velx - v_com[0]
    vely = vely - v_com[1]
    v_r = velx*np.cos(phi) + vely*np.sin(phi)
    return v_r.in_units("km/s")

def _v_phi(field, data):
    x = data['x'] - sinkcom[0]
    y = data['y'] - sinkcom[1]
    phi = np.arctan2(y,x)

    # Get velocities and correct for centre velocity
    velx = data['velocity_x']
    vely = data['velocity_y']
    velz = data['velocity_z']
    velx = velx - v_com[0]
    vely = vely - v_com[1]
    v_phi = -velx*np.sin(phi) + vely*np.cos(phi)
    return v_phi.in_units("km/s")

def _v_z(field, data):
    velz = data['velocity_z']
    velz -= v_com[2]
    return velz.in_units("km/s")

ds.add_field(("flash", "numdens"),  function=_numdens, units="cm**-3",sampling_type="cell")
#The basic fields
ds.add_field(("flash", "numdens"),  function=_numdens, units="cm**-3",sampling_type="cell", force_override=True)
ds.add_field(("flash", "plasma_beta"),  function=_plasma_beta, units="dimensionless",sampling_type="cell", force_override=True)
ds.add_field(("flash", "jeans_length"),  function=_jeans_length, units="cm",sampling_type="cell", force_override=True)
ds.add_field(("flash", "jeans_number"),  function=_jeans_number, units="cm",sampling_type="cell", force_override=True)

#Velocities
ds.add_field(("flash", "v_rad"),  function=_v_r, units="km*s**-1",sampling_type="cell", force_override=True)
ds.add_field(("flash", "v_phi"),  function=_v_phi, units="km*s**-1",sampling_type="cell", force_override=True)
ds.add_field(("flash", "v_z"),  function=_v_z, units="km*s**-1",sampling_type="cell", force_override=True)

dd = ds.all_data()

print('\n')
dens = np.array(dd["numdens"])
print('Max number density is', np.round(np.max(dens)/1e13,2),'* 10^13 cm^-3')
print("Let's see where the point of max density and max refinement is..")
print("This point is (wrt COM):")

#Find the cell with max density and max refinement level
ref = np.array(dd["grid_level"])
sorted_indices = np.argsort(dens)[::-1]
target_ref = 13.0
index_max = None
for idx in sorted_indices:
    if ref[idx] == target_ref:
        index_max = idx
        break

if index_max is not None:
    print(f"Found densest cell at refinement level {target_ref} with index {index_max}")
    
    x = float(dd["x"][index_max])
    y = float(dd["y"][index_max])
    z = float(dd["z"][index_max])
    
    c = np.array([x, y, z])
    cdiff = c-np.array(sinkcom)
    print("Coordinates wrt sink COM:")
else:
    print(f"No cell found at refinement level {target_ref}")

print('x = ',np.round(cdiff[0]*6.68458712e-14, 2), 'AU')
print('y = ',np.round(cdiff[1]*6.68458712e-14, 2), 'AU')
print('z = ',np.round(cdiff[2]*6.68458712e-14, 2), 'AU')
print('\n')

z6_nosf2_site = [9.66969951e13, 1.17277591e14, -3.22801285e13] #cm

# Creating the slices

print('Before we go further, would you like to save the (offcenter) sink positions at this step?')
while True:
    pos = input('Type "y" or "n": ')
    if pos == 'y':
        print('Calculating sink positions..')
        matrix = matrix_Lbasis(ds)
        sinkpos = np.dot(matrix,offcenter_sink_positions(ds,c,w))
        print('Saving..')
        np.save('offcenter_sinkpos_'+str(step)+'.npy',sinkpos)
        print('Positions saved. Proceeding..')
        break
    elif pos == 'n':
        print('Okay! Proceeding..')
        break
    else:
        print("Incorrect input, please try again.")

print('\n')

print('Now we move on to making the slices, this may take a while..')

make_slices(ds,c,False)
#make_slices(ds,sinkcom+z6_nosf2_site)

print('Complete! The script will now close.')




