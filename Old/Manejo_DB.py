#write a code to read the data from the json and display it in the console

import json
import pandas as pd
with open('../Resources/SeisBenchV1_v1_1.json') as file:
    data = json.load(file)

#converty to pandas dataframe

df = pd.DataFrame(data)
print(df.head())
