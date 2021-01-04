# References

# http://www-static-2019.jmmc.fr/mirrors/obsvlti/book/Segransan_1.pdf - Used for uv-plane equations
# https://matplotlib.org/3.1.1/api/ - Where I collected different commands for matplotlib

import numpy as np                  # importing packages
import matplotlib.pyplot as plt

VLA = np.loadtxt("Dconfig.txt")   # Loading in VLA configuration D data
source = np.loadtxt("source.txt")       # Source 1 and 2 data

f = 5e9 # 5 GHz frequency of interferometer
c = 3e8 # speed of light in a vacuum
Lambda= c/f # Wavelength of light from c=f*lambda

XYZ = VLA[:,0:3] # Array with VLA data

RAh,RAm,RAs = source[:,0],source[:,1],source[:,2] # Collecting the hours, minutes, seconds of Right Ascension
DECh,DECm,DECs = source[:,3],source[:,4],source[:,5] # Collecting source declinations
JySource = source[:,6]

Y,Z = VLA[:,1],VLA[:,2] # Telescope coordinates
plt.scatter(Y,Z) # Plotting the positions of the antennas
plt.title('VLA D configuration')
plt.ylabel('y (ns)')
plt.xlabel('x (ns)')
plt.show()

DATA = [] # Empty arrays for modulus baselines, Azimuth and Elevation values
for i, x_i in enumerate(VLA): # This loop is for calculating all 351 baselines, elevations and azimuth components
    for j, y_i in enumerate(VLA):
        if i < j: # This condition ensures no duplicate values are calculated, ie 2-1 and 1-2, 10-5 and 5-10 etc.
            DATA.append(x_i-y_i)
            
DATA = np.array(DATA) # Loading in all the antenna coordinates
X = DATA[:,0] # Allocating to X,Y,Z
Y = DATA[:,1]
Z = DATA[:,2]

dec = np.pi/4 # Placeholder declination

U = Lambda*(np.sin(0)*X + np.cos(0)*Y) # U-V values, again with RA set to 0 hrs
V = Lambda*(-np.sin(dec)*np.cos(0)*X + np.sin(dec)*np.sin(0)*Y + np.cos(dec)*Z)

#plt.scatter(V/1e3,U/1e3,s=1) # Plotting the UV coverage which is both the positive and negatives values of U and V
#plt.scatter(-V/1e3,-U/1e3,s=1) # Values for U and V are divided by 1,000 to match the units used in the VLA document provided
#plt.show()
# Can uncomment this to see the UV plane at time = 0

t = np.linspace(-np.pi/24,np.pi/24,120) # One hour of RA in radians from -0.5 hr to +0.5hr
u = [] # Blank u-v arrays for appending
v = []

for i in range(len(X)): # Loop for calculating u-v
    u = np.append(u, (f/c)*(np.sin(t)*X[i] + np.cos(t)*Y[i])) # Equations taken from Segransan + Neal Jackson lectures
    v = np.append(v, (f/c)*(-np.sin(dec)*np.cos(t)*X[i] + np.sin(dec)*np.sin(t)*Y[i] + np.cos(dec)*Z[i]))

plt.scatter(v/1e3,u/1e3,s=5,c='black') # Plotting the u-v plane
plt.scatter(-v/1e3,-u/1e3,s=5,c='black') # Negative have to be plotted for other half of u-v plane
plt.xlabel('x (kilo wavelengths)')
plt.ylabel('y (kilo wavelengths)')
plt.title('VLA UV coverage 1HA at dec=90 degrees')
plt.show()




TrueSky = np.zeros((400,400)) # For sources

TrueSky[200,200] = JySource[0] # This is the first centred image
TrueSky[20,50] = JySource[1] # This is the second source. 3 arc minutes difference in Declination and 10 seconds difference in RA.

plt.imshow(TrueSky,extent=[-200,200,-200,200]) # This is an image of the two sources
plt.title('Observed sources')
plt.xlabel('RA offset (arcsec, J2000)')
plt.ylabel('DEC offset (arcsec, J2000)')

plt.show()

Visibility = np.fft.fft2(TrueSky) # Fourier transform of sources
plt.imshow(np.real(Visibility),extent=[-200,200,-200,200]) # This image shows the real fourier space of the two observed sources
plt.xlabel('RA offset (arcsec, J2000)')
plt.ylabel('DEC offset (arcsec, J2000)')
plt.title('Fourier space of two point sources')
plt.show()


Sampling = np.zeros((400,400)) # Empty array for sampling function 
for i in range(len(u)):
    Sampling[int(u[i]/500),int(v[i]/500)] = 1 # Fills in empty image array with a 1 if uv data exists
    Sampling[int(-u[i]/500),int(-v[i]/500)] = 1 # also divided by 500 so all u,v data fits within sampling array
ShiftSampling = np.fft.fftshift(Sampling) # Shifts the image to the centre of the spectrum
plt.imshow(np.real(ShiftSampling),extent=[-200,200,-200,200]) # Plotting sampling function
plt.title('U-V sampling function') # sampled grid
plt.xlabel('RA offset (arcsec, J2000)')
plt.ylabel('DEC offset (arcsec, J2000)')
plt.show()

# Synthesised beam

Beamft = np.fft.fft2(Sampling) # Fourier transform of UV image to show synthesised(dirty) beam image
Beamft = np.fft.fftshift(Beamft) # This command shifts the image to the centre of the plot rather than being at the corners
Beamft = np.real(Beamft)
plt.imshow(Beamft,vmin=-10,vmax=300,extent=[-200,200,-200,200]) # Real values plotted
plt.title('Synthesised beam')                                 # vmin and vmax values best show the dirty beam
plt.colorbar()                          # Shows values of the synthesised beam
plt.xlabel('RA offset (arcsec, J2000)')
plt.ylabel('DEC offset (arcsec, J2000)')
plt.show()

SampleVis = Sampling * Visibility # Multiplying together to get sampled visibilities
DirtyMap = np.fft.ifft2(SampleVis) # Inverse fourier transform to get back the dirty map
plt.imshow(np.real(DirtyMap),extent=[-200,200,-200,200]) # Plotting and centering dirty map
plt.xlabel('RA offset (arcsec, J2000)')
plt.ylabel('DEC offset (arcsec, J2000)')
plt.title('Dirty map')
plt.show()

Primary = Lambda / 25 # Angular size of primary beam in radians at 5GHz with 25m apertures
PrimaryDeg = Primary * (180/np.pi) * 3600 # Angular size of beam in arcseconds
radius = int(np.round(PrimaryDeg/2)) # Integer and rounded value for radius of primary beam in arcseconds


x = np.linspace(1,2*radius,2*radius) # Creating lists for meshgrid that range over the width of the beam
y = np.linspace(1,2*radius,2*radius)
xx, yy = np.meshgrid(x,y) # Meshgrid for calculating distance to a pixel

PrimaryBeam = np.zeros((2*radius,2*radius)) # Blank image for primary beam of size width of primary beam
for i in range(1,2*radius): # Loop creating the primary beam
    r = ((xx-x[radius])**2 + (yy-y[radius])**2)**0.5 # Length from the centre of the primary beam                  
    PrimaryBeam += np.exp(-(r/[i])**2) # 'Temporary' equation for the primary beam
                          
plt.imshow(PrimaryBeam,extent=[-300,300,-300,300]) # Primary beam plotted, centre image is [0,0]
plt.xlabel('arcseconds') 
plt.ylabel('arcseconds')
plt.title('Primary Beam of VLA')
plt.colorbar()
plt.show()
