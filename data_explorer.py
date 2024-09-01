import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns

@st.cache_data
def load_iris():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    data = pd.read_csv(url, header=None, names=column_names)
    return data

df = load_iris()

st.header('Iris Dataset Assignment')

#Displaying the dataset
if st.checkbox('Show raw data'):
    st.dataframe(df)

#sepal length for each species
st.subheader('Sepal length for each species', divider='gray')
avg_sepal_len = df.groupby('species')['sepal_length'].mean()
st.write(avg_sepal_len)

#scatter plot comparing two features
st.subheader('Comparing two features', divider='gray')
feature_1 = st.selectbox('Choose first feature:', df.columns[:-1] )
feature_2 = st.selectbox('Choose second feature:', df.columns[:-1] )

scatter_plot = px.scatter(df, x=feature_1, y=feature_2, color='species', hover_name='species')
st.plotly_chart(scatter_plot)

#filter data based on species
st.subheader('Filter data based on species', divider='gray')
selected = st.multiselect('Choose species:', df['species'].unique())

if selected:
    filtered_species = df[df['species'].isin(selected)]
    st.dataframe(filtered_species)
else:
    st.write('Select some species.')

#pairplot for selected species
st.subheader('Pair plot for selected species', divider='gray')
if st.checkbox('Show pair plot for selected species'):
    if selected:
        fig_pairplot = sns.pairplot(filtered_species, hue='species')
    else:
        fig_pairplot = sns.pairplot(df, hue='species')
    
    st.pyplot(fig_pairplot)

#distribution of selected features
st.subheader('Distribution of selected features', divider='gray')
selected_feature = st.selectbox('Choose a feature:', df.columns[:-1])

feature_hist = px.histogram(df, x=selected_feature, color='species', nbins=30, hover_data=df.columns)
st.plotly_chart(feature_hist)