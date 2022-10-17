import os
import pandas as pd

import env


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_titanic_data(): 
    '''
    This function reads the telco data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT * FROM passengers
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    
    return df  

def get_titanic_data():
    filename = "titanic.csv"

    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename,index_col=0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        df = new_titanic_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  

def new_iris_data(): 
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT * FROM measurements JOIN species USING(species_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    
    return df   


def get_iris_data():
    filename = "iris.csv"

    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename ,index_col=0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        df = get_iris_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index = False)

        # Return the dataframe to the calling code
        return df 


def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df
def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
        
    return df
