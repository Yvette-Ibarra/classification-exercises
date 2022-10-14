import os
import pandas as pd

import env


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    filename = "titanic.csv"

    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  

    


def get_iris_data():
    filename = "iris.csv"

    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        df = pd.read_sql('SELECT * FROM measurements JOIN species USING(species_id);', get_connection('iris_db'))
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df 



 def gget_telco_data():   
    filename = "telco.csv"

    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        df = pd.read_sql(' SELECT * FROM customers LEFT JOIN contract_types USING(contract_type_id) LEFT JOIN internet_service_types USING (internet_service_type_id) LEFT JOIN payment_types USING (payment_type_id);', get_connection('iris_db'))
        # Write that dataframe to disk for later. Called "caching" the data for later.
        
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df 