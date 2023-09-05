# Fraud Detection with Autoencoder Neural Networks

The primary objective of this project is to develop a robust fraud detection system capable of effectively distinguishing fraudulent transactions from legitimate ones. This is achieved by utilizing state-of-the-art Autoencoder neural networks and the Keras framework.

## Project Goals

- Utilize advanced machine learning techniques to enhance financial security.
- Protect businesses and consumers from financial fraud.

## Model Details

- The model employs an Autoencoder architecture with 14 input features.
- The input data is reconstructed, and the reconstruction error is compared to a predefined threshold.
- If the error surpasses the threshold, the transaction is flagged as fraudulent.

By implementing this approach, we aim to significantly improve the accuracy of fraud detection, thereby safeguarding financial transactions and minimizing the impact of fraudulent activities.

For more information, you can visit the [Fraud Detection Streamlit App](https://fraud-detect.streamlit.app/).

## Running the Streamlit App on your own directory
I already includes dependencies in `requirements.txt`, <br>
so if you want to run this streamlit app on your own directory, follow the instruction below:

1. Clone the Git repository to your local machine:

   ```bash
   git clone https://github.com/syaerulid/fraud_detection_neural_network.git

2. Change to file directory
   ```bash
   cd fraud_detection_neural_network/Deployment
4. Run the streamlit app
   ```bash
   streamlit run fraud_tx.py

For Google Colab notebook version :<br>
you can check this link<br>
[Colab Version of Fraud Detection Model](https://colab.research.google.com/drive/1vOyygNTyes69__Stv5iWLE-rtQc-_9aI#scrollTo=sAECyvhWCwXS)

