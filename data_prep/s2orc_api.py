"""
Script that retrieves survey/review papers from the S2ORC API per available field. 
We then cluster results into one of five main fields: Engineering, Life Sciences, Physical Sciences and Mathematics, Social and Behavioural Sciences, Arts and Humanities.
Results are then filtered per field to include those with the an existing abstract and an accessible full-text PDF. 
Gemini is then used to get the main scientific entity per paper, which are saved to be reviewed manually.
"""

import requests
import os
import json
import time
import pandas as pd

from google import genai
from tqdm import tqdm 

# Existing s2orc fields to extract surveys from
s2orc_fields = ['Materials Science', 'Engineering', 'Medicine', 'Biology', 'Agricultural and Food Sciences', 'Computer Science', 'Chemistry', 'Physics', 'Geology', 'Mathematics', 'Environmental Science', 'Psychology',
                'Geography', 'Sociology', 'Business', 'Political Science', 'Economics', 'Law', 'Linguistics', 'Art', 'History', 'Philosophy', 'Education']

# Map existing s2orc fields to one of the five main fields of research according to the ORKG taxonomy
allowed_fields = {
    'Engineering': {'Materials Science','Engineering'},
    'Life Sciences': {'Medicine','Biology','Agricultural and Food Sciences'},
    'Physical Sciences and Mathematics': {'Computer Science','Chemistry','Physics','Geology','Mathematics','Environmental Science'},
    'Social and Behavioural Sciences': {'Psychology','Geography','Sociology','Business','Political Science','Economics','Law','Linguistics'},
    'Arts and Humanities': {'Art','History','Philosophy','Education'}
}

def check_license(license_dict):
  status = license_dict['status']
  if status == 'HYBRID' or status == 'GOLD' or status == 'GREEN':
    return True
  return False

def collect_data(minCitationCount=100, s2orc_key="YOUR API KEY"):
    """
    Function that goes over s2orc fields and fetches survey/review papers filtered to a specific min citation count number. 
    The function saves the results of each field as json files + a final dataframe with the metadata. 
    """

    url = 'https://api.semanticscholar.org/graph/v1/paper/search/bulk'

    for field in s2orc_fields:

        query_params = {
            'query': 'survey | review',
            'limit': 1,
            'fieldsOfStudy': field,
            'openAccessPdf': "True",
            'minCitationCount': minCitationCount,
            'sort': 'publicationDate:desc',
            'fields': 'externalIds,title,year,abstract,url,fieldsOfStudy,s2FieldsOfStudy,openAccessPdf,citationCount,publicationDate'
            }
        
        headers = {'x-api-key': s2orc_key}

        response = requests.get(url, params=query_params)

        total_responses = response.json()['total']
        print(f"Got {total_responses} for field: {field}")

        ## Start collecting survey papers from results ##
        ids, dois, titles, abstracts, urls, openAccessPdfs, fieldsOfStudy, s2FieldsOfStudy, citationCounts, publicationDate = [], [], [], [], [], [], [], [], [], []
        
        idx = 0
        errors = []
        
        # create folder with the name of the field: replace spaces with undescores if the exist
        field_ = field.replace(" ", "_")
        
        os.makedirs(f"{field_}", exist_ok=True)

        while len(ids)  <= total_responses:
            try:
                ids.extend([publication['paperId'] for publication in response.json()['data']])
                dois.extend([publication['externalIds']['DOI'] for publication in response.json()['data']])
                titles.extend([publication['title'] for publication in response.json()['data']])
                abstracts.extend([publication['abstract'] for publication in response.json()['data']])
                urls.extend([publication['url'] for publication in response.json()['data']])
                openAccessPdfs.extend([publication['openAccessPdf'] for publication in response.json()['data']])
                fieldsOfStudy.extend([publication['fieldsOfStudy'] for publication in response.json()['data']])
                s2FieldsOfStudy.extend([publication['s2FieldsOfStudy'] for publication in response.json()['data']])
                citationCounts.extend([publication['citationCount'] for publication in response.json()['data']])
                publicationDate.extend([publication['publicationDate'] for publication in response.json()['data']])
                print(f"Got data for idx {idx}")
            
            except Exception as error:
                errors.append((idx, error))
                print(f"Error for {idx}: {error}")

        # Save response.json() to a json file
        with open(f"{field_}/{field_}_s2orc_{idx}.json", 'w') as f:
            json.dump(response.json(), f)
            print(f"Saved response of {idx}")
            
            token = response.json()['token']
            query_params['token'] = token
            
            idx += 1
            response = requests.get(url, params=query_params)
            time.sleep(5)

        # create dataframe of results for s2orc field
        s2orc_publications = pd.DataFrame({
            'id': ids,
            'doi': dois,
            'title': titles,
            'abstract': abstracts,
            'url': urls,
            'openAccessPdf': openAccessPdfs,
            'fieldsOfStudy': fieldsOfStudy,
            's2FieldsOfStudy': s2FieldsOfStudy,
            'citationCount': citationCounts,
            'publicationDate': publicationDate
            })
        
        s2orc_publications.to_pickle(f's2orc_publications_{field_}.pkl')
        print(f'Saved DataFrame of {field} as pkl.')

def filter_data_per_field(field: str, dataframes):
    """
    Function that gets the dataframes per field (e.g. for Engineering, gets both Engineering and Materials Science), filters them to exclude rows from other fields, concatenates and deduplicates and returns a unified,
    filtered dataframe per field.
    Args: 
    - field: string depicting the main field being filtered (one of the five)
    - dataframes: list of dataframes to be included for a specific field
    """
    subfields = allowed_fields[field]
    filtered_dataframes = []

    # filter each dataframe to keep only the rows that have allowed fieldOfStudy; this is done to try to reduce multidisciplinary results 
    for dataframe in dataframes:

        filtered_dataframe = dataframe[dataframe["fieldsOfStudy"].apply(lambda x: bool(x) and set(x).issubset(allowed_fields[field]))]
        filtered_dataframes.append(filtered_dataframe)

    concatenated_field_data = pd.concat(filtered_dataframes).drop_duplicates(subset=['id'])
    # sort by latest first
    concatenated_field_data = concatenated_field_data.sort_values(by='publicationDate', ascending=False)
    # remove those with no abstracts
    concatenated_field_data = concatenated_field_data.dropna(subset=['abstract'])
    # remove those with unaccessible fulltext
    concatenated_field_data = concatenated_field_data[concatenated_field_data['openAccessPdf'].apply(check_license)]

    return concatenated_field_data

def gemini_get_main_entities(dataframe, GOOGLE_API_KEY="YOUR API KEY"):
    """
    Function that uses Gemini to prompt each retrieved survey papers to get the main scientific entity the survey is about. 
    Args:
    - dataframe: the dataframe containing the s2orc results with their metadata.
    - GOOGLE_API_KEY: your key to be able to use Gemini.
    """
    client = genai.Client(api_key=GOOGLE_API_KEY)
    if 'main_entity' not in dataframe.columns:
         dataframe['main_entity'] = [None] * len(dataframe)

    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
       if row['main_entity'] == None:
          title = row['title']
          abstract = row['abstract']
          prompt = f"Given a scientific abstract of a survey paper about a scientific topic, your task is to extract the main scientific entity the abstract is about. Your reponse should consist of the writing the main entity alone and nothing more. If the given paper is not a survey paper about a given topic that can be used as a credible knowledge source about the scientific entity, you response must be None. Title of survey paper: {title}. Abstract: {abstract}. Main entity:"
          response = client.models.generate_content(
             model="gemini-2.5-flash",
             contents=prompt
             )
          dataframe.at[idx, 'main_entity'] = response.text
          time.sleep(5)
    
    return dataframe

def main():

    collect_data()

    # read results and filter datasets; prompt gemini to get main scientific entities; save filtered dataframe per field to be reviewed manually
    for field, subfields in allowed_fields.item():
        
        field_dataframes = []
        for subfield in subfields:
            subfield_ = subfield.replace(" ", "_")
            subfield_dataframe = pd.read_pickle(f's2orc_publications_{subfield_}.pkl')
            field_dataframes.append(subfield_dataframe)
        
        field_ = field.replace(" ", "_")
        filtered_field_data = filter_data_per_field(field=field, dataframes=field_dataframes)
        filtered_field_data = gemini_get_main_entities(filtered_field_data)
        filtered_field_data.to_pickle(f'filtered_data_{field_}.pkl')

if __name__ == '__main__':
 	main()

        

