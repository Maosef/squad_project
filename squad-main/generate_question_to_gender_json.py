#! /usr/bin/env python3

import gender_guesser.detector as gender
import json
import os, sys

"""
USAGE: Run 'python3 generate_question_to_gender_json.py'
within directory that has './data/dev_eval.json'

Given squad-main/data/dev_eval.json, this script generates
question_to_gender.json inside the folder
question_to_gender.

we only want squad-main/data/dev_eval.json
script MUST be run in squad-main directory

squad-main/data/dev-v2.0.json is the raw data. it's already preprocessed
to dev_eval

To determine the number of answers of m,f, or u:
Run 'python3 generate_question_to_gender_json.py' 
>>> fp = open('question_to_gender/question_to_gender.json', 'r')
>>> question_to_gender = json.load(fp)
>>> fp.close()
>>> len(question_to_gender)
6078
>>> from collections import Counter
>>> genders_list = list(question_to_gender.values())
>>> genders_list[0]
'f'
>>> genders_counter = Counter(genders_list)
>>> genders_counter
Counter({'u': 6029, 'm': 31, 'f': 18})
"""

DEV_JSON_FILE_NAME = './data/dev_eval.json'
DIRECTORY_NAME = 'question_to_gender'
QUESTION_TO_GENDER_FILENAME = 'question_to_gender.json'

detector_output_to_gender_character = {
    'male' : 'm',
    'female' : 'f',
    'mostly_male' : 'm',
    'mostly_female' : 'f',
    'unknown' : 'u',
    'andy' : 'u'
}

"""
Function that detects the gender of string name.
Condense non-definitive categories

Return values of detector.get_gender (a str) and
what we convert them to
    'male' : 'm'
    'female' : 'f'
    'mostly_male' : 'm'
    'mostly_female' : 'f'
    'unknown' : 'u'
    'andy' (androgynous) : 'u'

Note: The difference between andy and unknown is that the 
former is found to have the same probability to be male than 
to be female, while the later means that the name wasn’t found in the database.

The detector may mark non-person entities with a gender.
Assume these rarely occur; we won't try to remedy this.
>>> d = gender.Detector(case_sensitive=False)
>>> d_sensitive = gender.Detector() # is case sensitive by default
>>> d.get_gender('France')
'female'
>>> d.get_gender('france')
'female'
>>> d_sensitive.get_gender('france')
'unknown'
>>> d_sensitive.get_gender('France')
'female'

We'll opt for case insensitivity to capture non-capitalized names
>>> d_sensitive.get_gender('Sally')
'female'
>>> d_sensitive.get_gender('sally')
'unknown'

Input:
    detector - gender_guesser.detector.Detector() object
    name - string (in python3, str obj is Unicode) to determine gender

Output:
    "m" - male, "f" - female, "u" - unknown

"""

def detect_gender(detector, name):
    detector_output = detector.get_gender(name)
    return detector_output_to_gender_character[detector_output]


def main():
    # if data file doesn't exist, exit
    if not os.path.isfile(DEV_JSON_FILE_NAME):
        print(f'Error: {DEV_JSON_FILE_NAME} does not exist')
        sys.exit(1)
    d = gender.Detector(case_sensitive=False)
    # create folder question_to_gender if doesn't exist
    if not os.path.isdir(DIRECTORY_NAME):
        os.mkdir(DIRECTORY_NAME)
    # read in dev
    print(f'Reading in {DEV_JSON_FILE_NAME}...')
    with open(DEV_JSON_FILE_NAME, 'r') as fp:
        dev = json.load(fp)
    # for each quesiton, get first answer from list of possible
    # answers. determine gender and add to question_to_gender dict
    question_to_gender = dict()
    for k,v in dev.items():
        uuid = v['uuid'] # str
        # list of answers may be empty
        # if so, assign unknown gender
        if len(v['answers']) != 0:
            first_answer = v['answers'][0] # str
            # maps uuid to gender m,f,u (str)
            question_to_gender[uuid] = detect_gender(d, first_answer)
        else:
            question_to_gender[uuid] = 'u'
            
    # dump to json file in directory
    dump_path = os.path.join(DIRECTORY_NAME, QUESTION_TO_GENDER_FILENAME)
    print(f'Dumping {dump_path}...')
    with open(dump_path, 'w') as fp:
        json.dump(question_to_gender, fp)
    print('Done')

if __name__ == '__main__':
    main()
