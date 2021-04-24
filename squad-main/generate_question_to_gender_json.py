#! /usr/bin/env python3

import gender_guesser.detector as gender

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
    d = gender.Detector(case_sensitive=False)
    name = "Bob"
    print(d.get_gender("Bob"))

if __name__ == '__main__':
    print("hello")
    main()
