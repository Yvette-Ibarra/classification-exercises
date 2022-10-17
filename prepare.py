import numpy as np
import pandas as pd

# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
# help with missing value by replacing blank with: median, mode, average, calculate using other column
from sklearn.impute import SimpleImputer


#-----------------------IRIS -------------------------------------------------------------------
def prep_iris(df):
    ''' This function takes in iris dataframe will drop columns ['species_id,'measurement_id']
    rename 'species name' to 'species'.  
    Creates a dummy data frame to encode the categorical values of species and concatanate 
    back into the original dataframe
    '''
    # use method to drop columns
    iris_df.drop(columns = ['species_id','measurement_id'], inplace = True)
    
    # use method .rename to rename columns
    iris_df.rename(columns={"species_name": "species"}, inplace = True)
    
    # create dummy data frame that encodes the 3 unique species name
    dummy_iris_df = pd.get_dummies(iris_df['species'], dummy_na = False)
    
    # concatenate the dummy_df with original data frame
    new_iris_df = pd.concat([iris_df, dummy_iris_df], axis = 1)
    
    return new_iris_df


#-----------------------TITANIC -------------------------------------------------------------------

def prep_titanic(df):
    '''
    This function takes in dataframe and
    drops columns embarked', 'pclass', 'passenger_id', 'deck' and
    encodes 'sex', 'class', 'embark_town' with drop_first false and
    concatenates encoded df with original df
    '''
    
    # dropped columns 'embarked', 'pclass', 'passenger_id', 'deck
    titanic_df.drop(columns = ['embarked', 'pclass', 'passenger_id', 'deck'], inplace = True)
    
    # encode titanic dataframe for sex', 'class', 'embark_town
    dummy_df = pd.get_dummies(titanic_df[['sex', 'class', 'embark_town']], dummy_na=False, drop_first=[False])
    
    # concatenate dummy data frame to original dataframe
    df = pd.concat([titanic_df,dummy_df], axis=1)

    return df

#-------------------TELCO CHURN ------------------------------------------------------------------

def prep_telco(df):
    '''
    This function takes in dataframe and 
    drops columns:'payment_type_id', 'internet_service_type_id', 'contract_type_id' 
    encode categorical columns, drop_first set to False: 'senior_citizen'gender','partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection', 'tech_support','streaming_tv','streaming_movies','paperless_billing', 'total_charges', 'churn','contract_type','internet_service_type','payment_type'
    Concatenate dummy_df to original data frame
    '''
    
    # drop unnecessary : payment_type_id', 'internet_service_type_id', 'contract_type_id' 
    telco_df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id' ])
    
    # encode categorical drop_first set to False 'senior_citizen'gender','partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection', 'tech_support','streaming_tv','streaming_movies','paperless_billing', 'total_charges', 'churn','contract_type','internet_service_type','payment_type'
    dummy_df = pd.get_dummies(telco_df[['senior_citizen', 
                                    'gender','partner',
                                    'dependents',
                                    'phone_service',
                                    'multiple_lines',
                                    'online_security',
                                    'online_backup',
                                    'device_protection', 
                                    'tech_support',
                                    'streaming_tv',
                                    'streaming_movies',
                                    'paperless_billing', 
                                    'total_charges', 
                                    'churn',
                                    'contract_type',
                                    'internet_service_type',
                                    'payment_type']], dummy_na=False, drop_first=[False])
    
    # Concatenate dummy_df to original data frame
    df = pd.concat([telco_df, dummy_df], axis=1)
    
    return df

#---------------------- Function for train_validate_test---------------------
def train_validate_test(df, target):
    
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])
    
    return train, validate, test