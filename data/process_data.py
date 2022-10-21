import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    Args:
    messages_filepath: path to the message data set
    
    messages_filepath: path to the categories data set
    
    
    return:
    df: merge data set
    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='outer', on=['id'])
        
    return df


def clean_data(df):
    '''
    Args:
    df: merged data set from load_data()
    
    cleaning data to take the following steps:
    -splitting categories features into separate category columns
    -converting category values from strings to numberics
    -removing duplicates
    
    return:
    df: clean dataframe
    
    '''
    
    # create a dataframe of the 36 individual category columns
    categories= df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc [0]
    
    # extract a list of new column names for categories.
    #slicing the last two characters of a string
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors = 'coerce')
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1, sort=False)
    
    # drop duplicates
    df.drop_duplicates(keep=False, inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    Args:
    df: clean dataframe
    
    database_filename: name of created database
    
    return: 
    none
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()