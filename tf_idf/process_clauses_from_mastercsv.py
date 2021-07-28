"""
Script-Level Docstring. Complete when uploading this script file to Github.

@author: Yi Yang
"""

import re
import string

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from textblob import TextBlob

# Dictionary: key - category of the legal contract; value - unique number identifier
CONTRACT_CATEGORIES = {'Affiliate Agreement': 0,
                           'Agency Agreement': 1,
                           'Co-Branding Agreement': 2,
                           'Collaboration Agreement': 3,
                           'Cooperation Agreement': 3,
                           'Consulting Agreement': 4,
                           'Development Agreement': 5,
                           'Distribution Agreement': 6,
                           'Distributor Agreement': 6,
                           'Endorsement Agreement': 7,
                           'Franchise Agreement': 8,
                           'Hosting Agreement': 9,
                           'Intellectual Property Agreement': 10,
                           'Joint Venture Agreement': 11,
                           'License Agreement': 12,
                           'Maintenance Agreement': 13,
                           'Manufacturing Agreement': 14,
                           'Marketing Agreement': 15,
                           'Non-Compete Agreement': 16,
                           'Non Competition Agreement': 16,
                           'No-Solicit Agreement': 16,
                           'Non-Disparagement Agreement': 16,
                           'Outsourcing Agreement': 17,
                           'Promotion Agreement': 18,
                           'Reseller Agreement': 19,
                           'Service Agreement': 20,
                           'Sponsorship Agreement': 21,
                           'Strategic Alliance Agreement': 22,
                           'Supply Agreement': 23,
                           'Transportation Agreement': 24
                           }

# File path to access master_clauses.csv file
masterclauses_filepath = r'C:\Users\yiyan\Desktop\CUAD_v1\master_clauses.csv'

fileid_to_filenames = pd.read_csv(masterclauses_filepath, usecols=['Filename'])
# Dictionary: key - unique number identifier; value - legal contract file name (string)
DICT_FILEID_TO_FILENAMES = fileid_to_filenames.to_dict(orient='dict')['Filename']
# print(DICT_FILEID_TO_FILENAMES)

fileid_to_rawcategory = pd.read_csv(masterclauses_filepath, usecols=['Document Name-Answer'])

# Dictionary: key - unique number identifier; value - legal contract file category (unprocessed string)
DICT_FILEID_TO_RAWCATEGORY = fileid_to_rawcategory.to_dict(orient='dict')['Document Name-Answer']

# print(DICT_FILEID_TO_RAWCATEGORY)


def convert_dictval_to_id(dictionary: dict):
    """
    Converts the value of each key-value pair in given dictionary to a unique id number. This id number corresponds
    to the value in the key-value pair of legal_contract_category dictionary.

    Parameters:
        dictionary (dict): Dictionary with key-value pairs of legal file_id and legal category_names (unprocessed string)

    Returns:
        Iterable[dict]: Dictionary with key as document_id and value as category_id
    """

    dict_fileid_to_categoryid = {}
    stemmer = SnowballStemmer("english")
    # count = 0

    for key in dictionary:
        raw_category_name = dictionary[key]
        processed_category = re.sub("[^a-zA-Z\s]+", "", raw_category_name).lower()
        processed_category = " ".join(processed_category.split())
        list_of_processed_category_words = processed_category.split()
        # print("count: " + str(count))
        # print(list_of_processed_category_words)
        # print(processed_category)
        set_of_processed_category_words = set()

        # Stems each word
        for i in range(len(list_of_processed_category_words)):
            set_of_processed_category_words.add(stemmer.stem(list_of_processed_category_words[i]))

        for contract_category in CONTRACT_CATEGORIES:
            get_category_label = contract_category.split()[0].lower()
            # The preprocessing algorithm is the same as processed_category to maintain equality
            category_label_regexed = re.sub("[^a-zA-Z\s]+", "", get_category_label)
            stemmed_processed_key = stemmer.stem(category_label_regexed)

            # Gets all the possible categories a legal contract can be in
            for word in set_of_processed_category_words:
                if stemmed_processed_key == word or stemmed_processed_key in word:
                    numbered_category = CONTRACT_CATEGORIES[contract_category]
                    # Checks if our category turned into the correct unique number identifier
                    # and matches with the actual category the legal contract belongs to
                    # file_category_numbered_dict[key] = [contract_category, numbered_category, file_category_dict[key]]

                    if key not in dict_fileid_to_categoryid:
                        dict_fileid_to_categoryid[key] = [numbered_category]
                    else:
                        # Makes sure no duplicate numbers are added
                        if numbered_category not in dict_fileid_to_categoryid[key]:
                            dict_fileid_to_categoryid[key] += [numbered_category]

                    # count should be >=510 since there's 510 files and each of them must be in at least one category.
                    # count += 1
                    break

    # Debugging to check which files fell through and didn't get processed
    # print("File IDs that didn't get processed: ")
    # for i in range(510):
    #     if i not in dict_fileid_to_categoryid:
    #         print(i, dictionary[i])

    # Debugging, print out the new dictionary
    # for key in dict_fileid_to_categoryid:
    #     print(str(key) + ": " + str(dict_fileid_to_categoryid[key]))

    # Debugging
    # print("count : " + str(count))

    return dict_fileid_to_categoryid


def extract_headerrow_from_csvfile(filepath: str):
    """
    Extracts entire column dealing with all 41 of the legal contract's categories in the cuad category csv file.
    Stores the result in to a dictionary.

    Parameter:
        filepath (string): file path to cuad category csv file.
    Returns:
        Iterable[tuple]:
            Dictionary of all 41 categories in a legal contract
            Set of values from the aforementioned dictionary
    """

    header_cols = pd.read_csv(filepath, usecols=['Category (incl. context and answer)'])
    # Dictionary of all the possible categories a legal contract can be in
    header_cols_dict = header_cols.to_dict(orient='dict')['Category (incl. context and answer)']

    # Dictionary with key (clause_id: 0-41) and value (clause label: "Document Name, Parties, Agreement Date, etc')
    clauseid_clauses_dict = {}
    for key in header_cols_dict:
        value = header_cols_dict[key].split(":")[-1].strip()
        # print(value)
        clauseid_clauses_dict[key] = value

    # Debugging
    # for key in clauseid_clausename_dict:
    #     print(key, clauseid_clausename_dict[key])

    clauses = clauseid_clauses_dict.values()

    return clauseid_clauses_dict, clauses


def inverted_dictionary(dictionary: dict):
    """
    Helper Method. Inverts the dictionary by swapping changing key-value pairs into value-key pairs.

    Returns:
        Iterable[dict]: inverted dictionary
    """
    ret_dict = {}

    for key, value in dictionary.items():
        ret_dict[value] = key

    return ret_dict


def extract_clausesentences_cols(filepath: str, clauses_cols: list):
    """
    Extracts all the rows from the 41 clause columns. Stores each column's rows into a dictionary. Each dictionary
    is appended into a list.

    Returns:
        Iterable[list]: list of dictionaries
    """
    list_of_dict_clausesentences_to_fileid = []
    for col in clauses_cols:
        list_of_dict_clausesentences_to_fileid.append(pd.read_csv(filepath, usecols=[col]))

    return list_of_dict_clausesentences_to_fileid


def extract_and_create_new_csv(preprocess: bool):
    """
    Creates a new CSV file with four columns: Sentence Text, Sentence ID, File ID, and Category ID.

        Sentence Text rows are made by extracting all the clause columns (41) from master csv. Each sentence text is kept
        in its original form as found in their respective legal contract. Sentences are also preprocessed to extract
        features.

        Sentence ID rows are made by vectoring the clause a sentence belongs to into a number.

        File ID rows are made by vectoring the file name a clause comes from

        Category ID rows are made by vectoring the legal contract domain that the clause comes from


    Returns three lists:
        List #1 contains all rows from Sentence Text column
        List #2 contains all rows ids from Clause ID column
        List #3 contains all rows from Category ID column
    """

    dict_fileid_to_categoryid = convert_dictval_to_id(dictionary=DICT_FILEID_TO_RAWCATEGORY)
    tuple_result = extract_headerrow_from_csvfile(filepath=r'C:\Users\yiyan\Desktop\CUAD_v1\tf_idf\catergory_descriptions.csv')
    clauses = list(tuple_result[1])
    list_of_dict_clausesentences_to_fileid = extract_clausesentences_cols(
        filepath=masterclauses_filepath, clauses_cols=clauses)
    dict_clauses_to_clauseid = inverted_dictionary(tuple_result[0])

    output_dict = {"Sentence Text": [], "Sentence ID": [], "File ID": [], "Category ID": []}

    # Lists to be returned for tf-idf rf classification
    all_sentences = []
    all_sentenceid = []
    all_categories = []

    index = 0
    for dict in list_of_dict_clausesentences_to_fileid:
        simplified_dict = dict.to_dict()[clauses[index]]
        for key, value in simplified_dict.items():
            if key in dict_fileid_to_categoryid:
                # Remove rows that do not have any sentences
                if preprocess == True:
                    processed_str = preprocessing(value)
                else:
                    processed_str = value

                if len(processed_str) > 0:
                    if len(dict_fileid_to_categoryid[key]) == 1:
                        output_dict["Sentence Text"].append(processed_str)
                        output_dict["Sentence ID"].append(dict_clauses_to_clauseid[clauses[index]])
                        output_dict["File ID"].append(key)
                        output_dict["Category ID"].append(dict_fileid_to_categoryid[key][0])

                        all_sentences += [processed_str]
                        all_sentenceid += [dict_clauses_to_clauseid[clauses[index]]]
                        all_categories += [dict_fileid_to_categoryid[key][0]]

                        # Create multiple rows if a sentence has multiple categories
                    else:
                        for category in dict_fileid_to_categoryid[key]:
                            output_dict["Sentence Text"].append(processed_str)
                            output_dict["Sentence ID"].append(dict_clauses_to_clauseid[clauses[index]])
                            output_dict["File ID"].append(key)
                            output_dict["Category ID"].append(category)

                            all_sentences += [processed_str]
                            all_sentenceid += [dict_clauses_to_clauseid[clauses[index]]]
                            all_categories += [dict_fileid_to_categoryid[key][0]]

        index += 1

    # Debugging
    # print("Sent Text", len(output_dict["Sentence Text"]))
    # print("Sent ID", len(output_dict["Sentence ID"]))
    # print("File ID", len(output_dict["File ID"]))
    # print("Category ID", len(output_dict["Category ID"]))

    # Writing output to a csv file
    df = pd.DataFrame(output_dict)
    df.to_excel(r'C:\Users\yiyan\Desktop\CUAD_v1\tf_idf\tf_idf_data.xlsx')

    return [all_sentences, all_sentenceid, all_categories]


def preprocessing(raw_sentence):
    """
    Takes in an unprocessed string representing a sentence and processing it classic ml classifiers
    """
    # Set of stop words in English
    STOP_WORDS = set(stopwords.words('english'))
    # table = str.maketrans('', '', string.punctuation)
    stemmer = SnowballStemmer("english")

    # lowercase, remove trailing spaces, remove commas, remove open close bracket
    processing_str = raw_sentence.lower().strip()[1:-1]
    for r in ((",", ""), ("'", ""), ("[* * *]", ""), ("<omitted>", ""), ("\\n", "")):
        processing_str = processing_str.replace(*r)

    if len(processing_str) == 0:
        return ""

    blob_object = TextBlob(processing_str)
    tokens = blob_object.words

    processed_tokens = []
    for token in tokens:
        # Remove punctuations
        # token = token.translate(table)
        # Remove stop words
        if token not in STOP_WORDS:
            # Stem word
            token = stemmer.stem(token)
            processed_tokens.append(token)

    processed_sentence = ""
    for token in processed_tokens:
        processed_sentence += token + " "

    return processed_sentence





