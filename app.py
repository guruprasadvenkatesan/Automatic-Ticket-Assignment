# 1. Library imports
import uvicorn
from fastapi import FastAPI
from TicketAssignment import TicketAssignment
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 2. Create the app object
app = FastAPI()
pickle_in = open("tokenizer.pickle","rb")
tokenizer=pickle.load(pickle_in)
loaded_model = tf.keras.models.load_model('Automatic_Ticket_Assignment.h5')


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_ticket(data:TicketAssignment):
    data = data.dict()
    description=data['description']
    txt=preprocess_data(description)
    seq= tokenizer.texts_to_sequences([txt])
    padded = pad_sequences(seq,maxlen=max_length)
   pred = model.predict(padded)
   pred = np.argmax(pred, axis=1)
    return {
        'prediction': pred 
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload