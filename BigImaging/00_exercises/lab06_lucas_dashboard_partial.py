import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px



def plot_map(df, cur_var, soil_vars):
	# round values to 2 decimals
	df[soil_vars] = df[soil_vars].round(2)
	# create map
	fig = px.scatter_mapbox(df, lat='GPS_LAT', lon='GPS_LONG', hover_name=cur_var, hover_data=['GPS_LAT','GPS_LONG']+soil_vars,
							color=cur_var, color_discrete_sequence=["fuchsia"], zoom=3, height=600)
	fig.update_layout(mapbox_style="open-street-map")
	fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
	# plot map
	st.plotly_chart(fig, use_container_width=True)




if __name__ == '__main__':
	# set page width
	st.set_page_config(layout="wide")
	# define soil variables
	soil_vars = ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC']
	# initialize
	uploaded_file = None
	df = None
	
	# define a layout with two columns

	# in the first column
	# - create file uploader
	# - create soil variable selector
	# - check that has been uploaded a file
	#   - if so, load it in a dataframe

	# in the second column
	# - if a dataframe has been loaded (df is not null)
	#   - plot the map using the function 'plot_map'
	