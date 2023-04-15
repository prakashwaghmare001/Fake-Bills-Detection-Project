# import libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components



# load pickle file
model = pickle.load(open('E:/ExcelR_project/Fake_Bill_Detection_Project/Fake_bills_prediction_app/random_forest_classifier.pkl','rb'))



def model_training():
    # title 
    html_temp = """
    <div style="background:black; padding:1px; margin:1px">
    <h2 style="color:white;text-align:center;">Fake Bills Detection App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.write('---')



    # add image in app
    from PIL import Image
    image = Image.open('E:/ExcelR_project/Fake_Bill_Detection_Project/Fake_bills_prediction_app/Fake_bill.jpeg')
    st.image(image, caption='Genuine or Fake')
    st.write('---')
    st.text('''This institution aims to put in place methods for identifying counterfeit euro 
banknotes. They therefore call on you, a data specialist, to set up a model that 
would be able to automatically identify the real ones from the fake ones. And this 
simply from certain dimensions of the banknote and the elements that compose it.''')
    st.write('---')
    
    

    # Data Visualization Part
    # You can add html report in app (autoviz (auto eda) save as a html report)
    df = pd.read_csv("E:/ExcelR_project/Fake_Bill_Detection_Project/Fake_bills_prediction_app/df_clean.csv")
    df.drop(["cluster"],inplace=True,axis=1)
    st.subheader('Data Visualization')
    st.sidebar.subheader('Data Visualization')
    
    chart_select = st.sidebar.selectbox(label="Select Chart Type",options=["--","Histogram","Scatterplot","Boxplot","Distplot","Heatmap"])
    numerical_column = list(df.select_dtypes(['float','int']).columns)
    if chart_select == "--":
        print()
    # Histogram
    if chart_select == "Histogram":
        st.sidebar.subheader('Histogram Setting')
        try:
            x_values = st.sidebar.selectbox("X axis",options=numerical_column)
            fig, ax = plt.subplots(figsize=(10,5))
            sns.histplot(df,x=x_values,hue=df['is_genuine'],ax=ax)
            st.write(fig)
        except Exception as e:
            print(e)
    # Scatterplot
    if chart_select == "Scatterplot":
        try:
            HtmlFile = open("E:/ExcelR_project/Fake_Bill_Detection_Project/Fake_bills_prediction_app/AutoViz_Plots/is_genuine/pair_scatters.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            # print(source_code)
            components.html(source_code,width=1000,height=400)
        except Exception as e:
            print(e)
    # Boxplot
    if chart_select == "Boxplot":
        st.sidebar.subheader('Boxplot Setting')
        try:
            x_values = st.sidebar.selectbox("X axis",options=numerical_column)
            fig, ax = plt.subplots(figsize=(10,5))
            sns.boxplot(df,x=x_values,ax=ax)
            st.write(fig)
        except Exception as e:
            print(e)
    # Distplot
    if chart_select == "Distplot":
        try:
            HtmlFile = open("E:/ExcelR_project/Fake_Bill_Detection_Project/Fake_bills_prediction_app/AutoViz_Plots/is_genuine/distplots_nums.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            # print(source_code)
            components.html(source_code,width=950,height=400)
        except Exception as e:
            print(e)
    # Heatmap
    if chart_select == "Heatmap":
        try:
            HtmlFile = open("E:/ExcelR_project/Fake_Bill_Detection_Project/Fake_bills_prediction_app/AutoViz_Plots/is_genuine/heatmaps.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            # print(source_code)
            components.html(source_code,width=1700,height=520)
        except Exception as e:
            print(e)


    def predict_genuine_fake(diagonal,height_left,height_right,margin_low,margin_up,length):
        input = np.array([[diagonal,height_left,height_right,margin_low,margin_up,length]]).astype(np.float64)
        prediction = model.predict(input)
        return int(prediction)

    # user input parameters slider
    st.sidebar.subheader('User Input Parameters')
    diagonal = st.sidebar.slider('diagonal', 171.04, 173.01, 171.81)
    height_left	= st.sidebar.slider('height_left', 103.14, 104.88,104.86)
    height_right = st.sidebar.slider('height_right', 102.82, 104.95, 104.95)
    margin_low = st.sidebar.slider('margin_low', 2.98, 6.9, 4.52)
    margin_up = st.sidebar.slider('margin_up', 2.27, 3.91, 2.89)
    length = st.sidebar.slider('length', 109.49, 114.44, 112.83)



    # input Parameters
    st.write('---')
    st.subheader('''Random Forest classifier to predict genuine or fake bills.''')
    st.write('The parameters you slide')
    data = {'diagonal': diagonal,
                'height_left': height_left,
                'height_right': height_right,
                'margin_low': margin_low,
                'margin_up': margin_up,
                'length': length}
    features = pd.DataFrame(data,index=[0])
    st.write(features)



    # adding button in app
    st.write('Press the predict button to predict genuine or fake bill')
    if st.button("Predict"):
        output = predict_genuine_fake(diagonal, height_left, height_right, margin_low, margin_up, length)

        if output == 0:
            output_o = """
            <div style = "background:black; padding:1px; margin:10px">
            <h6 style="color:white;text-align:center; padding:10px">Fake Bill</h6>
            </div>
            """
            st.markdown(output_o,unsafe_allow_html=True)

        elif output == 1:
            output_1 = """
            <div style = "background:black; padding:1px; margin:10px">
            <h6 style="color:white; text-align:center; padding:10px">Genuine Bill</h6>
            </div>
            """
            st.markdown(output_1,unsafe_allow_html=True)


if __name__=='__main__':
    model_training()
