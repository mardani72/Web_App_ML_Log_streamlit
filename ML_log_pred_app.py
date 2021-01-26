import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import SGDRegressor

########################################################################

st.markdown('## **Various ML Models to Predict a Well Log**')
st.markdown('### **Predict Neutron Porosity (NPHI) from other well logs using various machine learning algorithms, adjuct hyper-parameters.**') 
st.markdown(" Adjuct hyper-parameters to see the model performances. \
 For additional information  please contact *ryan.mardani@dataenergy.ca* or visit the website: https://dataenergy.ca")
st.sidebar.header('User Input')
st.sidebar.markdown('GR, RHOB, DTC, and SP logs are used as predictors to target NPHI log data.')
st.sidebar.markdown('[Download input data](https://raw.githubusercontent.com/mardani72/Web_App_ML_Log_streamlit/main/log_force.csv)')

#########################################################################



url = "https://raw.githubusercontent.com/mardani72/Web_App_ML_Log_streamlit/main/log_force.csv"
df1 = pd.read_csv(url)
sample_rate = st.sidebar.slider('Sample every ... point (select larger values for faster run):', 1, 30, 15)
df = df1.iloc[::sample_rate,:]
st.sidebar.write('Data Samples:', df.shape[0])
############################################################# data prepration for keep out set
blind = df[df['WELL'] == '15/9-13']
training_data = df[df['WELL'] != '15/9-13']

X = training_data[["GR","RHOB","DTC","SP"]].values
y = training_data['NPHI'].values

X_blind = blind[["GR","RHOB","DTC","SP"]].values
y_blind = blind['NPHI'].values

############################################################## Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_b = scaler.fit_transform(X_blind)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.sidebar.markdown('### Machine Learning Algorithm')
regression_name = st.sidebar.selectbox('', ("Aad Boost", "Neural Network (MLP)", 'KNN', 'SVM',
	'BAYS','Decision trees', 'SGD'))

def add_parameter_ui(reg_name):
	params = dict()
	
	if reg_name == 'Aad Boost':
		n_estimators = st.sidebar.slider("n_estimators", 50, 500,100)
		learning_rate = st.sidebar.slider("learning_rate", 0.1, 10.0, 1.0)
		params['n_estimators'] = n_estimators
		params['learning_rate'] = learning_rate

	elif reg_name =='Neural Network (MLP)':
		max_iter = st.sidebar.slider("max_iter", 10, 300, 100)
		params['max_iter'] = max_iter
		batch_size = st.sidebar.slider("batch_size", 2, 30, 10)
		params['batch_size'] = batch_size

	elif reg_name =='KNN':
		K = st.sidebar.slider("K", 1, 15, 5)
		params['K'] = K

	elif reg_name == 'SVM':
		C = st.sidebar.slider("C", 0.1 , 10.0)
		epsilon = st.sidebar.slider("epsilon", 0.01 , 0.09)
		params['C'] = C
		params['epsilon'] = epsilon

	elif reg_name == 'BAYS':
		n_iter = st.sidebar.slider("n_iter", 100 , 1000,200)
		params['n_iter'] = n_iter

	elif reg_name == 'SGD':
		eta0 = st.sidebar.slider("eta0", 0.001 , 1.0, 0.1)
		params['eta0'] = eta0

	else:
		max_depth = st.sidebar.slider("max_depth", 2 , 30)
		max_features = st.sidebar.slider("max_features", 1 , 4)
		params['max_depth'] = max_depth
		params['max_features'] = max_features
	
	return params

params = add_parameter_ui(regression_name)

def get_regression(reg_name, params):
    reg = None

    if reg_name == 'Aad Boost':
    	reg = AdaBoostRegressor(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])

    elif reg_name == 'Neural Network (MLP)':
    	reg = MLPRegressor(max_iter=params['max_iter'], batch_size=params['batch_size'], random_state=42)
    
    elif reg_name == 'KNN':
        reg = KNeighborsRegressor(n_neighbors=params['K'])
    elif reg_name == 'SVM':
        reg = svm.SVR(C=params['C'], epsilon=params['epsilon']) 

    elif reg_name == 'BAYS':
        reg = linear_model.BayesianRidge(n_iter=params['n_iter']) 
    elif reg_name == 'SGD':
        reg = linear_model.SGDRegressor(eta0=params['eta0']) 
    else:
        reg = DecisionTreeRegressor(max_features=params['max_features'], 
            max_depth=params['max_depth'], random_state=42)
    return reg

reg = get_regression(regression_name, params)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
R2 = r2_score(y_test, y_pred)



############################################################ examine models with blind data
y_pred_b = reg.predict(X_b)
R2_b = r2_score(y_blind, y_pred_b)
blind['Pred_NPHI'] =  y_pred_b
###################################################################################
Depth_b = blind.DEPTH_MD.values
GR_b = blind.GR.values
RHOB_b = blind.RHOB.values
NPHI_b = blind.NPHI.values
DTC_b = blind.DTC.values
SP_b = blind.SP.values

fig = plt.figure()

gs1 = GridSpec(1, 5, left=0.05, right=0.68, hspace=0.4,wspace=0.1)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
ax3 = fig.add_subplot(gs1[2])
ax4 = fig.add_subplot(gs1[3])
ax5 = fig.add_subplot(gs1[4])

gs2 = GridSpec(2, 1, left=0.75, right=0.98, hspace=0.4, wspace=0.1)
ax6 = fig.add_subplot(gs2[0])
ax7 = fig.add_subplot(gs2[1])

ax1.plot(GR_b, Depth_b, color='darkgreen', alpha=.8, lw=0.7, ls= '-')
ax1.set_title('GR', fontsize=8, color='darkgreen', fontweight='bold')
ax1.invert_yaxis()
ax1.set_ylabel('Depth(m)', fontsize=9)
ax1.grid(True, color='0.7', dashes=(5,2,1,2) )
ax1.set_facecolor('#F8F8FF')
ax1.tick_params(axis="x", labelsize=6)
ax1.tick_params(axis="y", labelsize=7)

ax2.plot(RHOB_b, Depth_b, color='darkgreen', alpha=.8, lw=0.7, ls= '-')
ax2.set_title('RHOB', fontsize=8, color='darkgreen',fontweight='bold')
ax2.invert_yaxis()
ax2.grid(True, color='0.7', dashes=(5,2,1,2) )
ax2.set_facecolor('#F8F8FF')
ax2.tick_params(axis="x", labelsize=6)
ax2.set_yticklabels([])

ax3.plot(DTC_b, Depth_b, color='darkgreen', alpha=.8, lw=0.7, ls= '-')
ax3.set_title('DTC', fontsize=8, color='darkgreen', fontweight='bold')
ax3.invert_yaxis()
ax3.grid(True, color='0.7', dashes=(5,2,1,2) )
ax3.set_facecolor('#F8F8FF')
ax3.tick_params(axis="x", labelsize=6)
ax3.set_yticklabels([])

ax4.plot(SP_b, Depth_b, color='darkgreen', alpha=.8, lw=0.7, ls= '-')
ax4.set_title('SP', fontsize=8, color='darkgreen', fontweight='bold')
ax4.invert_yaxis()
ax4.grid(True, color='0.7', dashes=(5,2,1,2) )
ax4.set_facecolor('#F8F8FF')
ax4.tick_params(axis="x", labelsize=6)
ax4.set_yticklabels([])

ax5.plot(NPHI_b, Depth_b, color='mediumblue', alpha=.9, lw=0.6, ls= '-')
ax5.plot(y_pred_b, Depth_b, color='orangered', alpha=.9, lw=0.9, ls= '-')
ax5.set_title('NPHI', fontsize=8, color='mediumblue', fontweight='bold')
ax5.invert_yaxis()
ax5.grid(True, color='0.7', dashes=(5,2,1,2) )
ax5.set_facecolor('#F8F8FF')
ax5.tick_params(axis="x", labelsize=6)
ax5.text(0.05, 1400, 'Pred. NPHI', color='orangered', fontsize=6,fontweight='bold', bbox={'alpha': 0.001, 'pad': 1})
ax5.set_yticklabels([])


sns.distplot(blind.NPHI, ax = ax6)
sns.distplot(blind.Pred_NPHI, ax = ax6)
ax6.set_title('NPHI Porosity Distribution', fontsize=7, color='k', fontweight='bold')
ax6.set_facecolor('#F8F8FF')
ax6.text(0.005, 18, 'Predicted NPHI', color='orangered', fontsize=6, fontweight='bold', bbox={'alpha': 0.001, 'pad': 1})
ax6.text(0.005, 16.5, 'True NPHI', color='cornflowerblue', fontsize=6,fontweight='bold', bbox={'alpha': 0.001, 'pad': 1})
ax6.tick_params(axis="x", labelsize=6)
ax6.tick_params(axis="y", labelsize=6)
ax6.set_xlabel('NPHI', fontsize=6)
ax6.set_ylim([0, 20])


sns.regplot(x = 'NPHI', y ='Pred_NPHI', data = blind, marker='*', color='royalblue', scatter_kws={'s': blind['NPHI']},
 line_kws={"color": "gray", 'lw':'0.7'}, ax= ax7)
ax7.set_title('True NPHI Vs. Prediction', fontsize=7, color='k', fontweight='bold')
ax7.annotate("R2 = {:.3f}".format(r2_score(y_blind, y_pred_b)), (0.06, 0.5), fontsize=6.5)
ax7.grid(True, color='0.9', dashes=(5,2,1,2) )
ax7.set_facecolor('#F8F8FF')
ax7.tick_params(axis="x", labelsize=6)
ax7.tick_params(axis="y", labelsize=6)
ax7.set_ylabel('Pred NPHI', fontsize=6)
ax7.set_xlabel('True NPHI', fontsize=6)

st.pyplot(fig)


st.write(f'Regressor Model = {regression_name}')
st.write(f'R2 Score (model) =', R2)
st.write(f'*R2 Score (Blind) =', R2_b)
st.markdown('*Blind data is a single well data that was involved in the modeling process; used for prediction evaluation purposes.')
st.markdown('To speed up computation, we used a coarse sampling of data to decrease waiting time to see the result.')

st.markdown('### **About dataset:**')
st.markdown('The original dataset belongs to Force 2020 comptetion, you can find more data here: https://www.dataenergy.ca/opendata ')
st.markdown('Looking at the summary of the dataset has shown that NPHI log is one of the most important data that is missing for some intervals.\
 We selected 3 wells (considering calculation cost) and used two of them for training and kept out one for the blind test (15/9-13) which is plotted above.')

st.dataframe(df1.iloc[0:10,:])

st.write('Shape of dataset:', df.shape)
