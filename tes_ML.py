import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#kita akan load model nya lagi untuk di tes

model_tes = joblib.load("piplane_model.pkl")

model = model_tes["model"]
scaler = model_tes["scaler"]
encode_model = model_tes["encoder"]

print("model sudah di load")

#cotoh data yang akan di tes

datates = pd.DataFrame({
    "brand": ["ASUS"],
    "processor_brand": ["Intel"],
    "processor_name": ["Core i5"],
    "processor_gnrtn": ["11th"],
    "ram_gb": ["8 GB"],
    "ram_type": ["DDR4"],
    "ssd": ["512 GB"],
    "hdd": ["No HDD"],
    "os": ["Windows 11"],
    "os_bit": ["64-bit"],
    "graphic_card_gb": ["4 GB"],
    "weight": ["1.8 kg"],
    "warranty": ["1 Year"],
    "Touchscreen": ["No"],
    "msoffice": ["Yes"],
    "rating": ["4.3"],
    
})



#kita endcode dulu 
encode = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
encode = encode_model.transform(datates[[
    'brand',
    'processor_brand',
    'processor_name',
    'processor_gnrtn',
    'ram_gb',
    'ram_type',
    'ssd',
    'hdd',
    'os',
    'os_bit',
    'graphic_card_gb',
    'weight',
    'warranty',
    'Touchscreen',
    'msoffice',
    'rating']]) 




#sscalling
skelear_model = scaler.transform(encode)
   
#prediksi 

prediction = model.predict(skelear_model)



print("hasil prediksi harga laptop adalah : ", prediction)

