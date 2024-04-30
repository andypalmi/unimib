import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

def plot_map(df, cur_var, soil_vars):
	"""
	Plots a scatter map using Plotly Express.

	Parameters:
	- df (pandas.DataFrame): The dataframe containing the data.
	- cur_var (str): The variable to be plotted on the map.
	- soil_vars (list): List of soil variables.

	Returns:
	None
	"""
	# round values to 2 decimals
	df[soil_vars] = df[soil_vars].round(2)
	# create map
	fig = px.scatter_mapbox(df, lat='GPS_LAT', lon='GPS_LONG', hover_name=cur_var, hover_data=['GPS_LAT','GPS_LONG']+soil_vars,
							color=cur_var, color_discrete_sequence=["fuchsia"], zoom=3, height=600)
	fig.update_layout(mapbox_style="open-street-map")
	fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
	# plot map
	st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def load_csv(data):
	"""
	Loads a CSV file into a pandas DataFrame.

	Parameters:
	- data (file object): The CSV file to be loaded.

	Returns:
	pandas.DataFrame: The loaded dataframe.
	"""
	if data is not None:
		# read csv
		df = pd.read_csv(data)
		# return
		return df

if __name__ == '__main__':
	# set page width
	st.set_page_config(layout="wide")
	# define soil variables
	soil_vars = ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC']
	# initialize
	uploaded_file = None
	df = None

	# define a layout with two columns
	col1, col2 = st.columns([1, 2])
	with col1:
		# in the first column
		# - create file uploader
		uploaded_file = st.file_uploader('Upload a file', type='csv')
		df = load_csv(uploaded_file)
		# - create soil variable selector
		choice = st.selectbox('Choose a soil variable', soil_vars)
		# - check that has been uploaded a file
		#   - if so, load it in a dataframe

	if df is not None:
		# in the second column
		# - plot the map using the function 'plot_map'
		with col2:
			plot_map(df, choice, soil_vars)
	# in the second column
	# - if a dataframe has been loaded (df is not null)
	#   - plot the map using the function 'plot_map'
	