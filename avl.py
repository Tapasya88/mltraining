import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


data = pd.read_csv("C:/Users/Administrator/Documents/training/avl.csv")


X = data[['Temperature  (deg F) ', 'Relative Humidity  (%) ', 'Total Snow Depth  (") ', 'Intermittent/Shot Snow  (") ']]
y = data['label']


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = keras.Sequential([
    keras.layers.LSTM(units=64, activation='relu', input_shape=(1, X_train.shape[2])),
    keras.layers.Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

input_data = pd.read_excel("C:/Users/Administrator/Downloads/10 dec test.xlsx")
input_data = input_data[['Temperature  (deg F) ', 'Relative Humidity  (%) ', 'Total Snow Depth  (") ', 'Intermittent/Shot Snow  (") ']]
input_data_scaled = scaler.transform(input_data)
input_data_reshaped = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))
print("+++++++++++++++++Reshaped++++++++++++++++")
print(input_data_reshaped)

new_input_predictions = model.predict(input_data_reshaped)


threshold = 0.5
for i, prediction in enumerate(new_input_predictions):
    if prediction >= threshold:
        print(f"Avalanche {i + 1} is predicted to happen.")
    else:
        print(f"Avalanche {i + 1} is predicted not to happen.")


thresholds_to_try = [0.3, 0.5, 0.7]
for threshold in thresholds_to_try:
    y_pred_binary = (model.predict(X_test) > threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Threshold: {threshold}, Accuracy: {accuracy:.4f}")
    cm = confusion_matrix(y_test, y_pred_binary)
    print(f"Confusion Matrix (Threshold {threshold}):\n{cm}")


# In[2]:


model.save("C:/Users/91810/Downloads/avalanche py")


# In[3]:


model = keras.Sequential([
    keras.layers.LSTM(units=64, activation='relu', input_shape=(1, X_train.shape[2])),
    keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("C:/Users/Administrator/Documents/training/avalanche py")

##
# In[4]:
from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

app = Flask(__name__)

# Load the pre-trained model
model = keras.models.load_model("C:/Users/Administrator/Documents/training/avalanche py")  
scaler = StandardScaler()


# Endpoint for prediction
#Start page
@app.route('/')
def index():
    return render_template('avalanche.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json("dataVal")

        # Assuming the input data is a list of dictionaries
        #input_data = pd.DataFrame(data)
        #print(input_data)
        
        #print(data['dataVal'])
        # Make predictions
        predictions = model.predict([[data['dataVal']]])

        # Apply threshold and create response
        threshold = 0.5
        results = [{'avalanche': i + 1, 'prediction': 'happen' if pred >= threshold else 'not happen'} for i, pred in enumerate(predictions.flatten())]

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)})
    

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5100)


# In[ ]:





# In[ ]:



