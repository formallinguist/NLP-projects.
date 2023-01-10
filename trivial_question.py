#!/usr/bin/python3
# -*- coding: utf-8 -*-

# WILL NOT WORK WITH THE RUN BUTTON ON VISUAL STUDIO CODE!
# USE THE TERMINAL:
# python3 trivial_questions.py -i your_input_file -o your_output_file

import pandas as pd
from pandas.core.frame import DataFrame

import random
import argparse

def Indian_surgeons_dob(fin_n, fout_n):
    """
    Function that given an input and an output CSV files, creates 
    trivial style questions for each row of the input file. The question
    it creates is about the Indian surgeon's birth date, and it produces two distractors
    for each correct answer.
    """

    df = pd.read_csv(fin_n)
    out_rows = []

    for _,row in df.iterrows():

        question = "Which Indian surgeon is born on "+row["dob"].strip()+"?"
        correct = row["personLabel"]
        random_numbers = random.sample(range(-50,51),2)

        distractor1 = df.iloc[random_numbers[0]]["personLabel"]
        distrator2 = df.iloc[random_numbers[1]]["personLabel"]

        out_rows.append([question,correct,distractor1,distrator2])

    df_out = DataFrame(out_rows)
    fieldnames = ["Question","Correct","Incorrect1","Incorrect2"]
    df_out.to_csv(fout_n,header=fieldnames,index=False)
    
    text = open("dob.csv","r")
    text = ' '.join([i for i in text])
    text = text.replace(","," ")
    
    filewrite=open("filename.txt",'w')
    filewrite.write(text)
    filewrite.close()

parser = argparse.ArgumentParser(description="Creates trivial exercises for Indian surgeons date of birth randomly based on a CSV file.")
parser.add_argument('-i',"--input",help="input file",required=True)
parser.add_argument('-o',"--output",help="output file",required=True)

args = parser.parse_args()

Indian_surgeons_dob(args.input, args.output)
