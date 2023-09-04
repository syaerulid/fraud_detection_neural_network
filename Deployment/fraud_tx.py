#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install pandas


# In[ ]:


pip install numpy


# In[ ]:


pip install matplotlib


# In[ ]:


pip install seaborn


# In[ ]:


pip install streamlit


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from PIL import Image


# In[2]:


def load_and_transform():
    # load
    df1 = pd.read_csv("cc_info.csv")
    df2 = pd.read_csv("transactions_cc.csv")
    df_mix = df2.merge(df1, how = 'left', left_on = 'credit_card', right_on = 'credit_card')
    df_mix.rename(columns = {'credit_card' : 'unique_tx'}, inplace = True)
    df_mix_snip = df_mix.head()
    df3 = pd.read_csv("dataframe_ft_eng.csv")
    # transform

    df3.drop(columns = ['Unnamed: 0'], inplace = True)
    df3_head = df3.head()
    replace_dict = {
        'Afternoon': 'Afternoon (12-18)',
        'Morning': 'Morning (6-12)',
        'Evening-Night': 'Evening (18-24)',
        'Late Night' : 'Late Night (0-6)'
    }

    df3['session'] = df3['session'].replace(replace_dict)
    
    df4 = pd.read_csv("df_model.csv")
    df4.drop(columns = ['Unnamed: 0'], inplace = True)
    df4_head = df4.head()
    
    df5 = pd.read_csv("df_outlier.csv")
    df5.drop(columns = ['Unnamed: 0'], inplace = True)
    df5_head = df5.head()
    
    df6 = pd.read_csv("df_stdz.csv")
    df6.drop(columns = ['Unnamed: 0'], inplace = True)
    df6_head = df6.head()
    
    df7 = pd.read_csv("result_data.csv")
    df7.drop(columns = ['Unnamed: 0'], inplace = True)
    df7_head = df7.groupby(['Classification']).size().reset_index(name = 'count')
    
    df8 = pd.read_csv("df_result_final.csv")
    df8.drop(columns = ['Unnamed: 0'], inplace = True)
    df8_head = df8.head()

    return df1, df2, df_mix, df_mix_snip, df3, df3_head, df4, df4_head, df5, df5_head, df6, df6_head, df7, df7_head, df8, df8_head


# In[3]:


result = load_and_transform()


# In[4]:


def header():
    image = Image.open("fraud_detect.png")
    st.image(image, caption="Example of Fraud Activities")


# In[5]:


def main_page(df_mix_snip, df3_head, df4_head, df5_head, df6_head, df7_head, df8):
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Load DataFrame','Viz','Outlier Handling','Standardization','Modeling','Result','Maps'])
    
    with tab1:
        st.header('Unmasking Fraudulent Transactions: Unveiling Anomalies with Autoencoder Neural Networks and Keras')
        header()
        st.write("---")
        st.subheader("What's this :blue[project] about?")
        st.write("""
        "The primary objective of this project is to develop a robust fraud detection system that can effectively distinguish fraudulent transactions from legitimate ones using 
        state-of-the-art Autoencoder neural networks and the Keras framework. 
        
        By leveraging advanced machine learning techniques, 
        this research aims to enhance financial security and protect businesses and consumers from financial fraud.""")
        st.write("---")
        st.write("This is snippet or preview from Original DataFrame that we use")
        st.dataframe(df_mix_snip)
        st.write("This is a snippet or preview of our DataFrame after performing feature engineering on it")
        st.dataframe(df3_head)

    with tab2:
        st.markdown("<h2 style='text-align: center;'>This is some Viz that we produce when working in this project</h2>", unsafe_allow_html=True)
        visualization(df3)
    
    with tab3:
        st.subheader("Outlier Handling")
        st.write("In the next step we will provide some viz about outlier in this dataframe, we need to handle them before going to modeling")
        st.write("This is snippet or preview from DataFrame model that still contain outlier")
        st.dataframe(df4_head)
        st.write("Below is process that we do to transform the outlier, so we can reduce skewness and extreme values that will affect our model")
        outlier_handling()
        st.write("This is snippet or preview of our DataFrame that no longer contain outlier after we perform transformation")
        st.dataframe(df5_head)
    
    with tab4:
        st.subheader("Standardization")
        st.write("we perform standardization before enter modeling, *so all features / column have equal weight* and features with wider ranges, didn't dominate the result")
        st.write("This is Process of standardization")
        standardization()
        st.write("This is snipper of our DF after Standardization, but input in our model is (array type) so we will convert this to array later")
        st.dataframe(df6_head)
    
    with tab5:
        st.subheader("Modeling")
        st.write("Step 1")
        early_stopping()
        st.write("Step 2")
        modeling()
        st.write("Step 3")
        st.write("Reconstruct the data and compare it to distinguish between normal and fraudulent transactions")
        reconstruct_data()
    
    with tab6:
        st.subheader("Result")
        st.write("This is the result of our model")
        st.dataframe(df7_head)
        st.subheader("This is our final dataframe + result")
        final_result(df8_head)
        explanation()
        st.subheader("Search something from this dataframe (like fraud, session, etc)")
        text_input()
        
        
    with tab7:
        st.subheader("this is our map")
        maps()

    
    
    
def sidebar():
    st.sidebar.markdown("## Sidebar")
    st.sidebar.subheader("Why Fraud Detection Matters:")
    st.sidebar.write("- Financial losses due to fraud.")
    st.sidebar.write("- Damage to reputation and customer trust.")
    st.sidebar.write("- Legal and regulatory implications.")

    st.sidebar.subheader("The Role of Machine Learning:")
    st.sidebar.write("- How machine learning can enhance fraud detection.")
    st.sidebar.write("- Advantages of using Autoencoder neural networks.")
    st.sidebar.write("- The relevance of Keras in deep learning.")
    
    st.sidebar.subheader("Dataset:")
    st.sidebar.write("https://www.kaggle.com/datasets/iabhishekofficial/creditcard-fraud-detection?select=cc_info.csv")
    
    st.sidebar.subheader("Reference:")
    st.sidebar.write("https://medium.com/@samuelsena/pengenalan-deep-learning-part-6-deep-autoencoder-40d79e9c7866")
    st.sidebar.write("https://kaggle.com/learn")
    st.sidebar.write("https://www.statology.org/data-binning-in-python/")
    st.sidebar.write("https://www.inscribe.ai/fraud-detection")
    
    st.sidebar.subheader("Contact me:")
    st.sidebar.write("https://www.twitter.com/syaerulid")
    st.sidebar.write("https://www.linkedin.com/in/syaerul-rochman/")


# In[6]:


df3 = result[5]
def visualization(df3):
    # 1. Distribution of transaction hour

    fig_hour, ax = plt.subplots(figsize=(10,8))
    hour = df3['hour_only']
    nums_bins = 48

    ax.hist(hour, nums_bins, color = 'green')
    ax.set_xlabel('Hour (0-24)')
    ax.set_ylabel('tx count')
    ax.set_title('Distribution of Transaction by Hour (0-24)')
    # Display the histogram in streamlit
    st.pyplot(fig_hour)
    expander = st.expander("See the Explanation above this hour histogram chart")
    expander.write("""
        Based on the histogram plot above, 
        distribution of hour transaction is left skew pattern, 
        with *majority transactions* occcuring between hours 15-24, 
        also from hours 5-10 there is almost no transaction happened
    """)

    

    # 2. Distribution of tx count by Day
    tx_day_distri = df3.groupby(['day_of_the_week']).size().reset_index(name='tx_count')
    tx_day_sort = tx_day_distri.sort_values('tx_count')

    tx_day, ax = plt.subplots(figsize=(10, 8))

    x = tx_day_sort['day_of_the_week']
    y = tx_day_sort['tx_count']

    color_palette = sns.color_palette('mako')
    day_session = ax.bar(x, y, color=color_palette)

    ax.set_title('Distribution of tx_count by Day')

    for bar, value in zip(day_session, y):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha='center', va='bottom')

    # Display the second histogram in Streamlit
    st.pyplot(tx_day)
    expander = st.expander("See the Explanation about this chart")
    expander.write("""
    based on the barchart above, it is evident that the *majority transaction* occured from Thursday to Saturday, 
    while Sunday - Tuesday
    recorded *lowest transaction* activity
    """)


    # 3. Distribution of tx count by their session
    session_dis = df3.groupby(['session']).size().reset_index(name = 'tx_count')
    session_dist = session_dis.sort_values('tx_count', ascending = False)

    fig_session, ax  = plt.subplots(figsize=(10,8))
    x = session_dist['session']                              
    y = session_dist['tx_count']
    color_palette = sns.color_palette('rocket_r')
    bars_session = ax.bar(x,y, color = color_palette)                              

    ax.set_title('Distribution of tx count based on their Session')

    for bar, value in zip(bars_session, y):
      ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha='center', va='bottom')

    # display the barchart on streamlit, about tx count and their session
    st.pyplot(fig_session)
    expander = st.expander("See the Explanation about this Chart")
    expander.write("""
    The *majority of transactions* happen during the Evening-Night session, 
    while the *lowest number of transactions* occurs during the Morning session.
    """)

    # 4. Distribution of tx by their month
    month_tx = df3.groupby(['month_only']).size().reset_index(name = 'tx_count')
    fig_month, ax = plt.subplots(figsize=(10,8))
    
    my_labels = month_tx['month_only']
    y_ax = np.array(month_tx['tx_count'])
    ax.pie(y_ax, labels = my_labels, autopct='%.2f')
    ax.set_title('Distribution Transaction by Month')
    
    # display the pir chart on streamlit
    st.pyplot(fig_month)
    
    expander = st.expander("See the Explanation about this Chart")
    expander.write("""
    based on the pie chart above, we can see clearly that *they have almost even distribution* transaction / month, 
    but July having notably fewer transactions. 
    This could be attributed to data collection possibly commencing towards the end of July.
    """)
    
    # 5. Top 50 City by tx_count
    city_count = df3.groupby(['city']).agg({'hour_only' : 'count'}).reset_index()
    city_count = city_count.sort_values('hour_only', ascending = False)
    city_count.rename(columns = {'hour_only' : 'total_tx'}, inplace = True)
    city_count_50 = city_count.head(50)
    
    fig_city, ax = plt.subplots(figsize=(10,8))
    x = city_count_50['city']
    y = city_count_50['total_tx']

    current_palette = sns.color_palette("magma")
    ax.set_title('Top 50 City with Most Transactions')
    ax.barh(x,y, color = current_palette)
    
    # display horizontal bar chart
    st.pyplot(fig_city)
    
    expander = st.expander("See the Explanation about this Chart")
    expander.write("""
    The bar chart above provides details on the top 50 cities with the highest transaction volume, with Dallas, El Paso, New York, 
    Houston and Washington leading in transaction frequency
    """)
    
    #6. Distribution of State and Transaction
    state_count = df3.groupby(['state']).agg({'hour_only' : 'count'}).reset_index()
    state_count = state_count.sort_values('hour_only', ascending = False)
    state_count.rename(columns = {'hour_only' : 'total_tx'}, inplace = True)
    
    fig_state, ax = plt.subplots(figsize=(10,8))
    x = state_count['state']
    y = state_count['total_tx']
    current_palette = sns.color_palette("Set2")
    ax.barh(x,y, color = current_palette)
    ax.set_title("Distribution Transaction / State")
    
    # display horizontal barchart
    st.pyplot(fig_state)
    
    # 7. Distribution of Transaction Amount
    fig_tx, ax = plt.subplots(figsize=(10,8))
    hour = df3['transaction_dollar_amount']
    nums_bins = 96

    ax.hist(hour, nums_bins, color = 'green')
    ax.set_title("Distribution of Transaction by amount")
    
    # display horizontal barchart
    st.pyplot(fig_tx)
    
    expander = st.expander("See the Explanation about this Chart")
    expander.write("""
    based on the histogram above, we can conclude that transaction amount have right skew distribution, 
    with majority amount is from 0.0x - 150, also there is outlier amount around 800-1000
    """)
    
    # 8. Distribution of Credit Card Limit
    fig_credit, ax = plt.subplots(figsize=(10,8))
    credit = df3['credit_card_limit']
    nums_bins = 48
    ax.hist(credit, nums_bins, color = 'orange')
    ax.set_title("Distribution of Credit Card Limit")
    
    st.pyplot(fig_credit)
    
    expander = st.expander("See the Explanation about this Chart")
    expander.write("""Based on the histogram above, credit card limit have left skew distribution""")


# In[7]:


def outlier_handling():
    code = """
    df_outlier = pd.DataFrame()
    
    df_outlier['unique_tx'] = df_model['unique_tx']
    df_outlier['zip_code'] = np.log(df_model['zipcode'])
    df_outlier['credit_card_limit'] = np.log(df_model['credit_card_limit'])
    df_outlier['tx_amount'] = np.log(df_model['transaction_dollar_amount'])
    df_outlier['transformed_longitude'] = np.log(np.abs(df_model['Long']))
    df_outlier['transformed_longitude'] *= np.sign(df_model['Long'])
    df_outlier['transformed_latitude'] = np.log1p(df_model['Lat'].round(10))
    df_outlier['hour'] = df_model['hour_only']
    df_outlier['num_day'] = df_model['num_day']
    df_outlier['month'] = df_model['month_only']
    df_outlier['year'] = df_model['year_only']
    df_outlier['session_Late Night'] = df_model['session_Late Night']
    df_outlier['session_Morning'] = df_model['session_Morning']
    df_outlier['session_Afternoon'] = df_model['session_Afternoon']
    df_outlier['session_Evening-Night'] = df_model['session_Evening-Night']
    """
    st.code(code, language='python')


# In[8]:


def standardization():
    code = """
    from sklearn.preprocessing import StandardScaler

    columns_to_standardize = [
    'unique_tx',
    'zip_code',
    'credit_card_limit',
    'tx_amount',
    'transformed_longitude',
    'transformed_latitude',
    'hour',
    'num_day',
    'month',
    'year',
    ]

    df_to_standardize = df_outlier[columns_to_standardize]

    # Instance
    scaler = StandardScaler()

    # standardized data
    stdz_data = scaler.fit_transform(df_to_standardize)

    # Replace the original columns in df_outlier with the standardized values
    df_outlier[columns_to_standardize] = stdz_data
    """
    st.code(code, language='python')


# In[9]:


def early_stopping():
    code = """
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
    )
    """
    
    st.code(code, language='python')
    expander = st.expander("Why we use early stopping?")
    expander.write("""
    Basically, Early Stopping is a tools that help us to stop the epochs 
    if there is no improvement of validation loss using (min_delta) to count 
    after some number of patience (interval epochs wait before stop). 
    Using Early Stopping we can stop training model before they begin to overfit""")


# In[10]:


def modeling():
    code = """
    # define input layer
    input_layer = Input(shape=(14,))

    # define the encoder layers
    encoded = Dense(128, activation = 'relu')(input_layer)
    encoded = Dense(64, activation = 'relu')(encoded)
    encoded = Dense(32, activation = 'relu')(encoded)

    # define the decoder layers
    decoded = Dense(64, activation = 'relu')(encoded)
    decoded = Dense(128, activation = 'relu')(decoded)
    decoded = Dense(14, activation = 'linear')(decoded) # linear activation

    # create the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs = decoded)

    # Compile the model
    autoencoder.compile(optimizer = 'adam', loss ='mean_absolute_error')

    # Train the autoencoder
    history = autoencoder.fit(input_stdz, input_stdz, epochs=50, batch_size=128, callbacks= [early_stopping], shuffle=True, validation_split=0.2)
    
    # Get the best validation loss from the EarlyStopping callback
    best_val_loss = early_stopping.best

    # Get the training loss at the end of training
    final_train_loss = history.history['loss'][-1]

    print("Best Validation Loss:", best_val_loss)
    print("Final Training Loss:", final_train_loss)
    
    Best Validation Loss: 0.008525182493031025
    Final Training Loss: 0.010011902078986168
    """
    
    st.code(code, language='python')
    expander = st.expander("Explanation")
    expander.write("""After performing the autoencoder modeling, we achieved the best validation loss of approximately 0.0085xx. 
    Thanks to the early stopping criteria we set, 
    we didn't need to go through all 50 epochs; instead, the training stopped at the 31st epoch.""")
    
    image = Image.open("plot_val.png")
    expander = st.expander("Click Here, to look the plot")
    expander.image(image)


# In[11]:


def reconstruct_data():
    code = """
    # calculate reconstruction errors for training data
    train_reconstructions = autoencoder.predict(input_stdz)
    train_errors = np.mean(np.abs(input_stdz - train_reconstructions), axis = 1)

    # copy data
    new_stdz_data = input_stdz.copy()

    # calculate mean and std of reconstruction error
    mean_error = np.mean(train_errors)
    std_error = np.std(train_errors)

    # set a thresold (you can adjutst this based on your data)
    threshold = best_val_loss + 2 * std_error

    # Calculate reconstruction errors for new data points
    new_data_reconstructions = autoencoder.predict(new_stdz_data)
    new_data_errors = np.mean(np.abs(new_stdz_data - new_data_reconstructions), axis=1)

    # Classify data points as fraud or normal based on the threshold
    classified_labels = []
    for error in new_data_errors:
        if error > threshold:
            classified_labels.append("Fraud")
        else:
            classified_labels.append("Normal")

    print(classified_labels)
    """
    
    st.code(code, language='python')
    expander = st.expander("So, what happened here?")
    expander.write("""The autoencoder is an unsupervised machine learning model where the input data is compared to its target, 
    which is essentially a reconstruction of the input data. 
    
    In our case, we standardize the input data and then create a copy of this standardized data""")
    
    expander = st.expander("More explanation")
    expander.write("""
    After that we set a thresold with this formula:
    
    threshold = best_validation_loss + 2 * std_error
    
    If *error > thresold*, we will classify them as fraud transaction, else, 
    they are valid transaction
    """)
    
    


# In[12]:


df8_head = result[15]
def final_result(df8_head):
    st.dataframe(df8_head)


# In[13]:


def text_input():
    user_input = st.text_input("Search specific keyword from this dataframe")
    
    filtered_df = df8[df8.applymap(lambda x: user_input.lower() in str(x).lower()).any(axis=1)]
    
    # display the filtered df
    st.write("Result:")
    st.dataframe(filtered_df)


# In[14]:


def explanation():
    expander = st.expander("Explanation about the result")
    expander.write("""
    In the end, after reconstructing the data using the autoencoder model, we assign these classifications back to our original dataset. 
    This allows us to identify which transactions are classified as fraud and which ones are considered normal.
    
    By doing this, we create a dependent variable or column for classification, which can be used for supervised learning classification tasks in the future.
    """)


# In[15]:


def maps():
    image = Image.open("fraud_map.png")
    expander = st.expander("Fraud Map")
    expander.image(image)
    
    # 
    st.markdown("[Interactive Version of Fraud Map](https://public.tableau.com/app/profile/syaerul.rochman/viz/fraudmap_16938269689100/fraud_map)")
    
    
    image_2 = Image.open("normal_map.png")
    expander = st.expander("Normal Map")
    expander.image(image_2)
    
    st.markdown("[Interactive Version of Normal Map](https://public.tableau.com/app/profile/syaerul.rochman/viz/normal_tx/city)")


# In[ ]:


if __name__ == "__main__":
    df1, df2, df_mix, df_mix_snip, df3, df3_head, df4, df4_head, df5, df5_head, df6, df6_head, df7, df7_head, df8, df8_head = load_and_transform()
    
    # main page
    main_page(df_mix_snip, df3_head, df4_head, df5_head, df6_head, df7_head, df8)
    
    
    # sidebar
    sidebar()
    


# In[ ]:





# In[ ]:





# In[ ]:




