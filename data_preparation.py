import pandas as pd
import requests
from io import StringIO

def load_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = StringIO(response.text)
        df = pd.read_csv(data, delimiter='\t', header=None)
        df.columns = ['RecipeID', 'Relation', 'Ingredient']
        filtered_data = df[df['Relation'] == 'recipeFoodstuff']
        return filtered_data
    else:
        raise Exception("Failed to fetch data from the URL")
