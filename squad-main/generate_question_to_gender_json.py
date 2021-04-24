#! /usr/bin/env python3

import gender_guesser.detector as gender

def main():
    d = gender.Detector()
    name = "Bob"
    print(d.get_gender("Bob"))

if __name__ == '__main__':
    print("hello")
    main()
