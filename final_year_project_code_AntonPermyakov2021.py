# -*- coding: utf-8 -*-
"""
@author: Anton
"""

#%% Manual selection of path

thin_wall_number = 25
# Path to folder with images , *.tif* allows to select only files in tiff format
path = r"Z:\\mapp1\\Shared\\BeAM\\Basler\\03_LC_In718-15_02_21\\Anton_{}_Pwr\\Images\\*.tif*".format(thin_wall_number)
#path = 'C:\\Users\\Anton\\Desktop\\Ni_16_nocontrol_Pwr\\Images\\*.tif*'
path_to_csv = "Z:\\mapp1\\Shared\\BeAM\\Basler\\03_LC_In718-15_02_21\\Anton_{}_Pwr".format(thin_wall_number)


# Laser Power (W) and Laser Velocity (mm/min) from  Anton DoE
#laser_power = 168 # 1
#laser_power = 212 # 2
#laser_power = 255 # 3
#laser_power = 299 # 4
laser_power = 343 # 5

#laser_power = 300 # for thin walls with different hatches

#laser_velocity = 1500 # 1 
#laser_velocity = 1875  # 2
#laser_velocity = 2250 # 3
#laser_velocity = 2625 # 4
laser_velocity = 3000 # 5

#hatch_spacing = 1
#hatch_spacing = 2
#hatch_spacing = 3
#hatch_spacing = 4
#hatch_spacing = 6
#hatch_spacing = 8
#hatch_spacing = 12
#hatch_spacing = 16
#hatch_spacing = 22

# Creating an empty list for time stamps of tiff images
datetime_list1 = []
# Lists for images
#tiff_list = [] # images will be saved as numpy arrays in tiff_list
#image_list = [] # images will be saved as TiffImageFile in image list; this will allow to inspect metadata individually

#%% Reading positions/power data
import time
#from os import remove
from glob import glob
import datetime
import pandas as pd
from PIL.TiffTags import TAGS # The Pillow (PIL) library must be at version 8.1.0 for the code to work
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from skimage.filters import gaussian, threshold_otsu

properties = ['area','perimeter',
             'major_axis_length',
             'max_intensity',
             'mean_intensity',
             #'equivalent_diameter',
             'minor_axis_length']
             #'solidity', 'eccentricity']
             #'coords']
# area (data type: int) - Number of pixels of the region.
# perimeter (data type:float) - Total perimeter of all objects in binary image.
# equivalent_diameter (data type:float) - The diameter of a circle with the same area as the region.
# major_axis_length (data type:float) - The length of the major axis of the ellipse that has the same normalized second central moments as the region.
# max_intensity (data type:float) - Value with the greatest intensity in the region.
# minor_axis_length - The length of the ellipse’s minor axis has the same normalized second central moments as the region.
# mean_intensity (data type:float) - Value with the mean intensity in the region.
# solidity — Ratio of pixels in the region to pixels of the convex hull image.
# eccentricity — Eccentricity of the ellipse that has the same second-moments as the region
# coords(data type: (N, 2) ndarray) : Coordinate list (row, col) of the region.

dataframe = pd.DataFrame(columns=properties)                                                                
                                                                
pixels_to_um = 5.4 # 1 pixel = 5.4 microns (from calibration of thermal camera)
beam_diameter = 750 # 750 microns
boltzmann_constant = 5.670367E-8 # in W*(m^-2) *K^-4

feedrate = pd.read_csv(path_to_csv + '\\Feedrate.csv',names=['Feedrate','Time'])
LaserPower = pd.read_csv(path_to_csv + '\\LaserPower.csv',names=['LaserPower','Time'])
LaserStatus = pd.read_csv(path_to_csv + '\\LaserStatus.csv',names=['LaserStatus','Time'])
x_pos = pd.read_csv(path_to_csv + '\\XPos.csv',names = ['XPos','TimeX'])
x_pos.XPos = x_pos.XPos-x_pos.XPos[0] # Relative x-coordinate = Substracting first x-coordinate from all x-coordinates
y_pos = pd.read_csv(path_to_csv + '\\YPos.csv',names = ['YPos','TimeY'])
y_pos.YPos = y_pos.YPos-y_pos.YPos[0] # Relative y-coordinate = Substracting first y-coordinate from all y-coordinates
z_pos = pd.read_csv(path_to_csv + '\\ZPos.csv',names = ['ZPos','TimeZ'])
z_pos.ZPos = z_pos.ZPos-z_pos.ZPos[0] # Relative z-coordinate = Substracting first z-coordinate from all z-coordinates

#Converting time to suitable datetime format
x_pos['TimeX'] = pd.to_datetime(x_pos['TimeX']) # TimeX is datetime timestamp = Good
x_pos['TimeX'] = x_pos['TimeX'].dt.strftime('%H:%M:%S.%f') # Changes format but also changes datetime to string = Not good
x_pos['TimeX'] = pd.to_datetime(x_pos['TimeX'])

y_pos['TimeY'] = pd.to_datetime(y_pos['TimeY'])
y_pos['TimeY'] = y_pos['TimeY'].dt.strftime('%H:%M:%S.%f')
y_pos['TimeY'] = pd.to_datetime(y_pos['TimeY'])

z_pos['TimeZ'] = pd.to_datetime(z_pos['TimeZ'])
z_pos['TimeZ'] = z_pos['TimeZ'].dt.strftime('%H:%M:%S.%f')
z_pos['TimeZ'] = pd.to_datetime(z_pos['TimeZ'])

# These 3 lines of code make sure that time stamps of feedrate, laser power and laser status are all in the same format as x,y,z
# Date is changed to today's date but hours, minutes, seconds and milliseconds remain the same
feedrate['Time'] = pd.to_datetime(feedrate['Time'])
feedrate['Time'] = feedrate['Time'].dt.strftime('%H:%M:%S.%f')
feedrate['Time'] = pd.to_datetime(feedrate['Time'])

LaserPower['Time'] = pd.to_datetime(LaserPower['Time'])
LaserPower['Time'] = LaserPower['Time'].dt.strftime('%H:%M:%S.%f')
LaserPower['Time'] = pd.to_datetime(LaserPower['Time'])

LaserStatus['Time'] = pd.to_datetime(LaserStatus['Time'])
LaserStatus['Time'] = LaserStatus['Time'].dt.strftime('%H:%M:%S.%f')
LaserStatus['Time'] = pd.to_datetime(LaserStatus['Time'])

# LaserPower*LaserStatus indicates when the laser is on or off
Lp_Ls = pd.DataFrame(LaserPower['LaserPower']*LaserStatus['LaserStatus'],columns=['LP*LS'])

Lp_Ls['Feedrate'] = feedrate['Feedrate']
Lp_Ls['X'] = x_pos['XPos']
Lp_Ls['Y'] = y_pos['YPos']
Lp_Ls['Z'] = z_pos['ZPos']
# Time of the dataframe is set to be the same as in LaserPower dataframe
Lp_Ls['Time'] = LaserPower['Time']
# Selecting rows with non-zero values only
Lp_Ls = Lp_Ls.loc[(Lp_Ls['LP*LS']!=0)]
Lp_Ls.set_index('Time',inplace=True)
Lp_Ls.sort_index(ascending=True,inplace=True)

#%% Image Processing
counter = 0
# Recording time of running the code
begin = time.time()
# Scanning path folder for files with .tif ending
for file in glob(path):
    # To limit number of files passed for processing
    if float(file[-5])%2 == 0:
        # Open image file for reading (binary mode)
        with Image.open(file) as f:
            imarray = np.asarray(f)
            # if sum of pixels of an image is more than 2x10^6, then the image probably contains a melt pool
            #if np.sum(imarray)>200000:
            # if maximum intensity of the image is above 100, then the image contains a melt pool
            # this thrseshold was chosen by plotting a histogram of maximum intensities from each image
            # it is possible to use either sum of pixels or maximum intensity for limiting number of images for processing
            if imarray.max() > 100:      
                # Denoising by Gaussian filter
                blurred = gaussian(imarray, sigma=.8)
                #Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
                binary = blurred > threshold_otsu(blurred)
                # Label melt pool
                labels = measure.label(binary,connectivity=imarray.ndim)
                # Retrieve properties of labelled melt pools and put them into a dataframe
                regions = measure.regionprops_table(labels,imarray,properties=properties)
                # Properties (metl pool area, intensity, length , etc.) of a single TIFF image
                data = pd.DataFrame(regions)
                # If area of any region labeled region is more than 2000, then the following code will be executed
                for i in data['area']:
                    if i > 2000 and i < 15000:
                        # Select melt pools with areas above 2000 pixels only, otherwise it takes spatters as seperate melt pools
                        selected_data = data[data['area'] == i]
                        # Extracting meta data from TIFF images
                        meta_dict = {TAGS[key] : f.tag[key] for key in f.tag.keys()}
                        #datetime_tag = meta_dict['DateTime'][0][0:12]
                        datetime_tag = datetime.datetime.strptime(meta_dict['DateTime'][0],'%H:%M:%S.%f ')
                        # Retrieve metadata of each image with melt pool and select DateTime stamp of each image
                        datetime_list1.append(datetime_tag)
                        # Concatenate melt pool propeties to another dataframe to save the properties for all images
                        dataframe = pd.concat([dataframe,selected_data])
                        # Saving selected images to a list, in case you want to check for any anomalies or artefacts
                        #tiff_list.append(imarray)
                        #image_list.append(f)
                        counter+=1
                        print('Number of images with a melt pool: ',counter)
            # Delete images whose maximum pixel intensity is below 100     
            #else:
            #    try:
            #        os.remove(file)
            #    except:
            #        print("Error while deleting a tiff file : ", file)
                                                     
# Store end time
end = time.time() 
# Total time taken
print(f"Total runtime of the image processing part of the code is {(end - begin)/60} minutes")
#%% Making final dataframe for machine learning
# Created pandas dataframe from datetime_tag list
df_tiff = pd.DataFrame(datetime_list1,columns=['Time'])
# At this stage df_tiff is in the same order as dataframe
df_tiff['Time'] = df_tiff['Time'].dt.strftime('%H:%M:%S.%f')
df_tiff['Time'] = pd.to_datetime(df_tiff['Time']) # day, month and year will be taken from day of running the code but hours,minutes and seconds will remain from original file

# Joining dataframes            
# Once this is done, check with your excel spreadsheet
dataframe.reset_index(drop=True,inplace=True)
dataframe = dataframe.join(df_tiff)   
dataframe.set_index('Time',inplace=True)

# Change integer data into floats
dataframe['area'] = dataframe['area'].astype(float)
dataframe['max_intensity'] = dataframe['max_intensity'].astype(float)

# Convert area and melt pool length from pixels to microns
dataframe['area_sq_microns'] = dataframe['area'] * (pixels_to_um**2)
dataframe['melt_pool_length'] = dataframe['major_axis_length']*(pixels_to_um)    
dataframe['melt_pool_width'] = dataframe['minor_axis_length']*(pixels_to_um) 
dataframe['Temperature'] = (dataframe['max_intensity']/boltzmann_constant)**0.25

# Feature engineering - making new features out of collected image properties
#dataframe['ratio_length'] = dataframe['major_axis_length'] / dataframe['minor_axis_length']
#dataframe['perimeter_ratio_major'] = dataframe['perimeter'] / dataframe['major_axis_length']
#dataframe['perimeter_ratio_minor'] = dataframe['perimeter'] / dataframe['minor_axis_length']
#dataframe['peri_over_dia'] = dataframe['perimeter'] / dataframe['equivalent_diameter']

# Adding power and velocity as columns to dataframe with meltpool areas 
dataframe['Power'] = np.ones(len(dataframe)) * laser_power
dataframe['Velocity'] = np.ones(len(dataframe)) * laser_velocity

#dataframe['Hatch'] = np.ones(len(dataframe)) * hatch_spacing

# Sorting images chronologically 
dataframe.sort_index(ascending=True,inplace=True)

# The camera is recorded on one PC (with correct time), and the positions/power on a PC without internet (with the wrong time).
# Time of coordinates is lagging behind time of tiff images
# Time of TIFF images is the correct time
# Find time difference between 1st image with melt pool and when LaserPower*LaserStatus != 0
difference_in_time = dataframe.index[0] - Lp_Ls.index[0]
Lp_Ls.index = Lp_Ls.index + difference_in_time

# Joining LaserPower*LaserStatus and Feedrate to dataframe with melt pool areas
data_frames = dataframe.join(Lp_Ls,how='outer')

# Interpolation of data
data_frames['Power'].interpolate(inplace=True) # This line is not accurate because laser power changes as LaserPower
data_frames['Velocity'].interpolate(inplace=True)
#data_frames['Hatch'].interpolate(inplace=True)
data_frames['LP*LS'].interpolate(inplace=True)
data_frames['Feedrate'].interpolate(inplace=True)
data_frames['X'].interpolate(inplace=True)
data_frames['Y'].interpolate(inplace=True)
data_frames['Z'].interpolate(inplace=True)

data_frames['LinearHeatInput'] = data_frames['LP*LS']/data_frames['Velocity']
data_frames['EnergyDensity'] = data_frames['LP*LS']/(2*beam_diameter*data_frames['Velocity'])
# Removing rows with absent melt pools ; interpolating melt pool areas would have created non-existent melt pools
data_frames.dropna(inplace=True)

# Removing area and major axis length since they are converted to melt pool area and melt pool length
data_frames.drop(['area','major_axis_length','minor_axis_length'],axis=1,inplace=True)
#%% Graphs
# Plotting melt pool area against time
plt.figure()
plt.title('Thin wall, Power = {} W, Velocity = {} mm/min'.format(laser_power,laser_velocity))
plt.xlabel('Time')
plt.ylabel(r'Melt Pool Area $(microns^{2})$')
#plt.ylim(70000,420000)
plt.scatter(data_frames.index,data_frames['area_sq_microns'])
plt.savefig(r'C:\\Users\\Anton\\Desktop\\Final Report\\Images for report\\MeltPoolAreas_ThinWalls\\ThinWall{}.png'.format(thin_wall_number))

# Plotting melt pool area against time
plt.figure()
plt.title('Thin wall, Power = {} W, Velocity = {} mm/min'.format(laser_power,laser_velocity))
plt.xlabel('Time')
plt.ylabel('Maximum Intensity')
#plt.ylim(70000,420000)
plt.scatter(data_frames.index,data_frames['max_intensity'])
plt.savefig(r'C:\\Users\\Anton\\Desktop\\Final Report\\Images for report\\Maximum Intensities ThinWalls\\ThinWall{}.png'.format(thin_wall_number))

# Plot melt pool area vs z step
plt.figure()
plt.title('Thin wall, Power = {} W, Velocity = {} mm/min'.format(laser_power,laser_velocity))
plt.xlabel('Z step')
plt.ylabel(r'Melt Pool Area $(microns^{2})$')
#plt.ylim(70000,420000)
plt.scatter(data_frames['Z'],data_frames['area_sq_microns'])
plt.savefig(r'C:\\Users\\Anton\\Desktop\\Final Report\\Images for report\\ZStep_ThinWalls\\ThinWall{}.png'.format(thin_wall_number))

#%% Saving dataframes as csv and excel files

#data_frames.to_csv('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thinwalls_different_hatch\\thin_wall_12.csv',index=True,date_format='%H:%M:%S.%f')
#data_frames.to_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thinwalls_different_hatch\\thin_wall_12.xlsx',index=True)

data_frames.to_excel(r'C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thinwalls_extended\\thin_wall_{}_extended.xlsx'.format(thin_wall_number),index=True)
#%% 3D Plotting

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# coordinates
x = data_frames['X']
x = x.reset_index()
x.drop(['Time'],inplace=True,axis=1)
y = data_frames['Y']
y = y.reset_index()
y.drop(['Time'],inplace=True,axis=1)
z = data_frames['Z']
z = z.reset_index()
z.drop(['Time'],inplace=True,axis=1)

# Colorbar 
color_bar = data_frames['melt_pool_length']
color_bar = color_bar.reset_index()
color_bar.drop(['Time'],inplace=True,axis=1)

# Plotting
threedee = plt.figure(figsize=(12,6)).gca(projection='3d')
p = threedee.scatter(x,y,z,c=color_bar,cmap='magma')
threedee.set_xlabel('x')
threedee.set_ylabel('y')
threedee.set_zlabel('z')
plt.colorbar(p)

#%% Opening excel files
data_frames1=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thinwalls_extended\\thin_wall_1_extended.xlsx',index_col='Time')
data_frames2=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thinwalls_extended\\thin_wall_2_extended.xlsx',index_col='Time')
data_frames3=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thinwalls_extended\\thin_wall_3_extended.xlsx',index_col='Time')
data_frames4=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thinwalls_extended\\thin_wall_4_extended.xlsx',index_col='Time')
data_frames5=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thinwalls_extended\\thin_wall_5_extended.xlsx',index_col='Time')
data_frames6=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_6.xlsx',index_col='Time')
data_frames7=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_7.xlsx',index_col='Time')
data_frames8=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_8_2.xlsx',index_col='Time')
data_frames9=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_9.xlsx',index_col='Time')
data_frames10=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_10.xlsx',index_col='Time')
data_frames11=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_11.xlsx',index_col='Time')
data_frames12=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_12.xlsx',index_col='Time')
data_frames13=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_13.xlsx',index_col='Time')
data_frames14=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_14.xlsx',index_col='Time')
data_frames15=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_15.xlsx',index_col='Time')
data_frames16=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_16.xlsx',index_col='Time')
data_frames17=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_17.xlsx',index_col='Time')
data_frames18=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_18.xlsx',index_col='Time')
data_frames19=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_19.xlsx',index_col='Time')
data_frames20=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_20.xlsx',index_col='Time')
data_frames21=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_21.xlsx',index_col='Time')
data_frames22=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_22.xlsx',index_col='Time')
data_frames23=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_23.xlsx',index_col='Time')
data_frames24=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_24.xlsx',index_col='Time')
data_frames25=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_25.xlsx',index_col='Time')
data_frames26=pd.read_excel('C:\\Users\\Anton\\Desktop\\ML\\DataFramesThinWalls\\thin_wall_26.xlsx',index_col='Time')

#%% Assembling dataframes into one big dataframe

df_final = pd.concat([data_frames1,data_frames2,
                      data_frames3,data_frames4,
                      data_frames5,data_frames6,
                      data_frames7,data_frames8,
                      data_frames9,data_frames10,
                      data_frames11,data_frames12,
                      data_frames13,data_frames14,
                      data_frames15,data_frames16,
                      data_frames17,data_frames18,
                      data_frames19,data_frames20,
                      data_frames21,data_frames22,
                      data_frames23,data_frames24,
                      data_frames25,data_frames26])

#%% Exploratory data analysis
import seaborn as sns

sns.pairplot(df_final);

sns.pairplot(df_final, vars = ['max_intensity','mean_intensity','area_sq_microns',
                               'melt_pool_length','Velocity','Power','Feedrate']);
sns.pairplot(df_final, x_vars = ['mean_intensity', 'melt_pool_length'], 
             y_vars = ['area_sq_microns']);

#%% Splitting data into testing and training datasets
# These split of data will be used for all machine learning models

from sklearn.model_selection import train_test_split
# All of the dropped dataframe columns correspond to features that we want to predict
# Models are trained on Velocity, Power ,LP*LS ,Feedrate, x,y,z
X = df_final.drop(['perimeter','area_sq_microns',
                   'melt_pool_length','minor_axis_length',
                   'equivalent_diameter','max_intensity',
                   'mean_intensity','ratio_length',
                   'perimeter_ratio_major','perimeter_ratio_minor',
                   'peri_over_dia'], axis=1)

y = df_final['area_sq_microns']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.15,
                                                    shuffle=True,
                                                    random_state=101)

#%% ANN - Sequential Model - 2 hidden layers - StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

#Scale data, otherwise model will fail.
#Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = Sequential()
# Hidden layers
model.add(Dense(128,input_dim=7,activation='relu'))
model.add(Dense(64, activation='relu'))
#Output layer
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

model.summary()

history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs =100)
y_pred_neural = model.predict(X_test_scaled)
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
acc = history.history['mae'] # mae = mean_absolute_error
val_acc = history.history['val_mae']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.xlabel('Real values of maximum intensity')
plt.ylabel('Predicted values of maximum intensity')
plt.title('Neural Nets Model')
plt.scatter(y_test,y_pred_neural);
plt.show()

#Predict on test data
predictions = model.predict(X_test_scaled[:5])
print("Predicted values are:\n ", predictions)
print("Real values are: ", y_test[:5])
##############################################

#Comparison with other models..
#Neural network - from the current code
mse_neural, mae_neural = model.evaluate(X_test_scaled, y_test)
print('Mean squared error from neural net: ', mse_neural)
print('Mean absolute error from neural net: ', mae_neural)
print('Root mean squared error from neural net:', np.sqrt(mse_neural))
print('Explained variance score (R^2 - value): ',explained_variance_score(y_test, y_pred_neural))

#%% Linear regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

### Linear regression
lr_model = linear_model.LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print('Mean squared error from linear regression: ', mse_lr)
print('Mean absolute error from linear regression: ', mae_lr)
print('Root mean squared error from linear regression:', np.sqrt(mse_lr))
print('Explained variance score (R^2 - value): ',explained_variance_score(y_test, y_pred_lr))

plt.figure()
plt.xlabel('Real values of maximum intensity')
plt.ylabel('Predicted values of maximum intensity')
plt.title('Accuracy of Linear Regression Model')
plt.scatter(y_test,y_pred_lr);

plt.figure()
plt.xlabel('Real values of melt pool area $(microns^{2})$')
plt.ylabel('Predicted values of melt pool area $(microns^{2})$')
plt.title('Accuracy of Linear Regression Model')
plt.scatter(y_test,y_pred_lr);

coeffecients = pd.DataFrame(lr_model.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients
############################################################

#%% Random forest  - Scaled data
#Increase number of tress and see the effect
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 50, random_state=101)
model.fit(X_train_scaled, y_train)

y_pred_RF = model.predict(X_test_scaled)

mse_RF = mean_squared_error(y_test, y_pred_RF)
mae_RF = mean_absolute_error(y_test, y_pred_RF)
print('Mean squared error using Random Forest: ', mse_RF)
print('Mean absolute error using Random Forest: ', mae_RF)
print('Root mean squared error from Random Forest:', np.sqrt(mse_RF))
print('Explained variance score (R^2 - value): ',explained_variance_score(y_test, y_pred_RF))

#Feature ranking
import pandas as pd
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

plt.figure()
plt.xlabel('Real values of maximum intensity')
plt.ylabel('Predicted values of maximum intensity')
plt.title('Random Forest Regressor Model (50 trees)')
plt.scatter(y_test,y_pred_RF);

plt.figure()
plt.xlabel('Real values of melt pool area $(microns^{2})$')
plt.ylabel('Predicted values of melt pool area $(microns^{2})$')
plt.title('Accuracy of Forest Regressor Model (50 trees)')
plt.scatter(y_test,y_pred_RF);

#%% Predicting on brand new data

# [[Feature1, Feature2]]
new_data = [[998,1000]]

# Don't forget to scale!
new_data = scaler.transform(new_data)

new_predictions = model.predict(new_data)

#%% Saving and loading a model

from tensorflow.keras.models import load_model
model.save('ANN_model_thinwalls.h5')

ANN_model_1 = load_model('ANN_model_thinwalls.h5')

ANN_model_1.predict(new_data)

# Machine learning code was adapted from Sreenivas Bhattiprolu's Python for Microscopists YouTube tutorials