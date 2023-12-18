import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Adding to supress warning
st.set_option('deprecation.showPyplotGlobalUse', False)


# Set page title and icon
st.set_page_config(page_title="Iris Dataset Explorer", page_icon="🌸")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Modeling", "Make Predictions!", "Extras"])

# Read in data
df = pd.read_csv('data/iris.csv')

# HOME PAGE
if page == "Home":
    st.title("📊 Iris Dataset Explorer")
    st.subheader("Welcome to our Iris dataset explorer app!")
    st.write("This app is designed to make the exploration and analysis of the Iris dataset easy and accessible. Whether you're interested in the distribution of data, relationships between features, or the performance of a machine learning model, this tool provides an interactive and visual platform for your exploration. Enjoy exploring the world of data science and analytics with the Iris dataset!")
    st.image('https://bouqs.com/blog/wp-content/uploads/2021/11/iris-flower-meaning-and-symbolism.jpg')
    st.write("Use the sidebar to navigate between different sections.")



# DATA OVERVIEW PAGE
elif page == "Data Overview":
    st.title("🔢 Data Overview")

    st.subheader("About the Data")
    st.write("This is one of the earliest datasets used in the literature on classification methods and widely used in statistics and machine learning.  The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.")
    st.image('https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png')
    st.link_button("Click here to learn more", "https://en.wikipedia.org/wiki/Iris_flower_data_set", help = "Iris Dataset Wikipedia Page")



    st.subheader("Quick Glace at the Data")

    # Display dataset
    if st.checkbox("DataFrame"):
        st.dataframe(df)
    
    # Column List
    if st.checkbox("Column List"):
        st.code(f"Columns: {df.columns.tolist()}")
        if st.toggle("Further breakdown of columns"):
            st.code(f"Numerical Columns: {df.select_dtypes(include='number').columns.tolist()}\nObject Columns: {df.select_dtypes(include = 'object').columns.tolist()}")

    # Shape
    if st.checkbox("Shape"):
        st.write(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")


# EDA PAGE
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    eda_type = st.multiselect("What type of EDA are you interested in exploring?", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    # Histograms
    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column:", num_cols, index = None)
        if h_selected_col:
            chart_title = f"Distribution of {' '.join(selected_col.split('_')).title()}"
            hue_toggle = st.toggle("Species Hue")
            if hue_toggle:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title = chart_title, color = 'species', barmode = 'overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title = chart_title))
    
    # Box Plots
    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for your box plot:", num_cols, index = None)
        if b_selected_col:
            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
            if st.toggle("Species Hue on Box Plot"):
                st.plotly_chart(px.box(df, x=b_selected_col, y = 'species', title = chart_title, color = 'species'))
            else:
                st.plotly_chart(px.box(df, x=b_selected_col, title = chart_title))

            
    # Scatterplots
    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, index = None)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, index = None)

        if selected_col_x and selected_col_y:
            hue_toggle = st.toggle("Species Hue")
            chart_title = f"{' '.join(selected_col_x.split('_')).title()} vs. {' '.join(selected_col_y.split('_')).title()}"

            if hue_toggle:
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='species', title = chart_title))
            else:
                #st.write(plot_title)
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, title = chart_title))
    
    # Count Plots
    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox("Select a categorical variable:", cat_cols, index = None)
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            st.plotly_chart(px.histogram(df, x=selected_col, title = chart_title, color = 'species'))


# MODELING PAGE
if page == "Modeling":
    st.title(":gear: Modeling")

    # Set up X and y

    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[features]
    y = df['species']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Model Selection
    model_option = st.selectbox("Select a Model", ["KNN", "Logistic Regression", "Random Forest"], index = None)

    if model_option:

        # Train and evaluate the selected model
        if model_option == "KNN":
            k_value = st.slider("Select the number of neighbors (k)", 1, 21, 5, 2)
            model = KNeighborsClassifier(n_neighbors=k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()

        if st.button("Let's see the performance!"):
            model.fit(X_train, y_train)

            # Display Results
            st.subheader(f"{model} Evaluation")
            st.text(f"Training Accuracy: {model.score(X_train, y_train)}")
            st.text(f"Testing Accuracy: {model.score(X_test, y_test)}")


            st.subheader("Confusion Matrix:")
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = 'Blues')
            st.pyplot()
            

# Predictions Page
if page == "Make Predictions!":
    st.title(":rocket: Make Predictions on Iris Dataset")

    # Create sliders for user to input data
    st.subheader("Adjust the sliders to input data:")

    s_l = st.slider("Sepal Length (cm)", 0.0, 10.0, 0.0, 0.01)
    s_w = st.slider("Sepal Width (cm)", 0.0, 10.0, 0.0, 0.01)
    p_l = st.slider("Petal Length (cm)", 0.0, 10.0, 0.0, 0.01)
    p_w = st.slider("Petal Width (cm)", 0.0, 10.0, 0.0, 0.01)

    # Your features must be in order that the model was trained on
    user_input = pd.DataFrame({
            'sepal_length': [s_l],
            'sepal_width': [s_w],
            'petal_length': [p_l],
            'petal_width': [p_w]
            })

    # Check out "pickling" to learn how we can "save" a model
    # and avoid the need to refit again!
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[features]
    y = df['species']

    # Model Selection
    st.write("The predictions are made using RandomForestClassifier as it performed the best out of all of the models.")
    model = RandomForestClassifier()
        
    if st.button("Make a Prediction!"):
        model.fit(X, y)
        prediction = model.predict(user_input)
        st.write(f"{model} predicts this iris flower is {prediction[0]} species!")
        st.balloons()



if page == 'Extras':
    st.title("How to Add Columns!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg")

    with col2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg")

    with col3:
            st.header("An owl")
            st.image("https://static.streamlit.io/examples/owl.jpg")
    



    tab1, tab2, tab3 = st.tabs(["Tab1", "Tab2", "Tab3"])
    
    with tab1:
        st.write("This is the first tab! Welcome!")
    with tab2:
        st.write("This is the second tab! Wooo!")
    with tab3:
        st.write("Last but not least!")
