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
    df.drop(columns = ['species_id','measurement_id'], inplace = True)
    
    # use method .rename to rename columns
    df.rename(columns={"species_name": "species"}, inplace = True)
    
    # create dummy data frame that encodes the 3 unique species name
    dummy_iris_df = pd.get_dummies(df['species'], dummy_na = False)
    
    # concatenate the dummy_df with original data frame
    new_iris_df = pd.concat([df, dummy_iris_df], axis = 1)
    
    return new_iris_df

def clean_iris(df):

    '''Prepares acquired Iris data for exploration'''
    
    # drop column using .drop(columns=column_name)
    df = df.drop(columns='species_id')
    
    # remame column using .rename(columns={current_column_name : replacement_column_name})
    df = df.rename(columns={'species_name':'species'})
    
    # create dummies dataframe using .get_dummies(column_name,not dropping any of the dummy columns)
    dummy_df = pd.get_dummies(df['species'], drop_first=False)
    
    # join original df with dummies df using .concat([original_df,dummy_df], join along the index)
    df = pd.concat([df, dummy_df], axis=1)
    
    return df

def split_iris_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on species.
    return train, validate, test DataFrames.
    '''
    
    # splits df into train_validate and test using train_test_split() stratifying on species to get an even mix of each species
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    
    # splits train_validate into train and validate using train_test_split() stratifying on species to get an even mix of each species
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species)
    return train, validate, test

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

def clean_titanic_data(df):
    '''
    This function will clean the data prior to splitting.
    '''
    # Drops any duplicate values
    df = df.drop_duplicates()

    # Drops columns that are already represented by other columns
    cols_to_drop = ['deck', 'embarked', 'class', 'passenger_id']
    df = df.drop(columns=cols_to_drop)

    # Fills the small number of null values for embark_town with the mode
    df['embark_town'] = df.embark_town.fillna(value='Southampton')

    # Uses one-hot encoding to create dummies of string columns for future modeling 
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)

    return df

def split_titanic_data(df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    # Creates the test set
    train, test = train_test_split(df, test_size = .2, random_state=123, stratify=df.survived)
    
    # Creates the final train and validate set
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.survived)
    
    return train, validate, test

def impute_titanic_mode(train, validate, test):
    '''
    Takes in train, validate, and test, and uses train to identify the best value to replace nulls in embark_town
    Imputes that value into all three sets and returns all three sets
    '''
    imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    return train, validate, test

def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test
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

def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test
#---------------------- Function for train_validate_test---------------------
def train_validate_test(df, target):
    ''' This function takes in a dataframe and target variable to sratify and  slpits the data into 
    train , validate, test'''
    
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])
    
    return train, validate, test



