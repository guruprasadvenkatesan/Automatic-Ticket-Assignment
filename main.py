import uvicorn
import NLP
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Message(BaseModel):
    text: str

app = FastAPI()
loaded_model = tf.keras.models.load_model('Automatic_Ticket_Assignment.h5')
pickle_in = open("tokenizer.pickle","rb")
tokenizer=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post("/ticket/")
async def ticket_assignment(message: Message):
    text =NLP.preprocess_data(message.text)
    seq= tokenizer.texts_to_sequences([txt])
    padded = pad_sequences(seq,maxlen=max_length)
    pred = loaded_model.predict(padded)
    pred =np.argmax(pred, axis=1)
    return pred 

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# use uvicorn main:app --reload to run the server
