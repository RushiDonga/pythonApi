from fastapi import FastAPI, Request
import base64
import pdfplumber
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import os
import json

import requests
import io
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

app = FastAPI()

@app.post("/")
async def getInformation(info : Request):
    req_info = await info.json()
    source = req_info['base64']

    print(source)

    # if request.method == 'POST':
    #     if request.is_json:
    #         data = request.get_json()
    #         source = data.get('base64')
    #         print("Data Sent: {}".format(source))
    #     else:
    #         print('Invalid JSON Format')
    # else:
    #     print('Invalid Request Method')
 
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "image-to-ocr.json"
    client = vision_v1.ImageAnnotatorClient()

    def extract_text_from_pdf(pdf_path):
        extracted_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text.extend(page.extract_text().split('\n'))
        return extracted_text

    def extract_text_from_image(image_content):
        client = vision_v1.ImageAnnotatorClient()
        image = types.Image(content=image_content)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        extracted_text = [text.description for text in texts]
        return extracted_text[1:]

    def extract_text_from_base64(base64_string):
        decoded_data = base64.b64decode(base64_string)

        # Check if it is a PDF
        if decoded_data[:4] == b'%PDF':
            pdf_path = 'temp.pdf'
            with open(pdf_path, 'wb') as pdf_file:
                pdf_file.write(decoded_data)
            extracted_text = extract_text_from_pdf(pdf_path)
            return extracted_text
        else:
            # Assume it is an image (JPEG, PNG)
            extracted_text = extract_text_from_image(decoded_data)
            return extracted_text

    extracted_text = extract_text_from_base64(source)
    print(extracted_text)

    
    # Function to preprocess the extracted text
    def preprocess_text(text):
        # Convert text to lowercase
        text = text.lower()

        # Tokenize the text into individual words
        words = nltk.word_tokenize(text)

        # Initialize the lemmatizer
        lemmatizer = WordNetLemmatizer()

        # Remove stopwords and punctuations, and apply lemmatization
        processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english') and word not in string.punctuation]

        # Join the processed words back into a single string
        processed_text = ' '.join(processed_words)

        return processed_text

    # Preprocess the extracted text
    processed_text = [preprocess_text(text) for text in extracted_text]
    
    # Fetch data from processed_text line by line and remove empty lines
    processed_lines = [line for line in processed_text if line.strip()]

    ############################################
    ######## patient name ######################
    ##########################################

    # Create a dataframe from the processed text
    df = pd.DataFrame(processed_lines, columns=['Processed Text'])

    # Find the row that contains "patient name"
    row_index = df[df['Processed Text'].str.contains('patient name', na=False)].index

    # Initialize the variable to store the fetched data
    patient_name = ''

    # Check if "patient name" is present in the dataframe
    if len(row_index) > 0:
        row = df.loc[row_index[0]]
        text = row['Processed Text']

        # Split the text by whitespace
        words = text.split()

        # Find the index of "patient name" in the list of words
        name_index = words.index('patient') + 2  # Get the index of the word following "patient name"

        # Check if the index is within the range of the list
        if name_index < len(words):
            # Fetch the two words after "patient name"
            patient_name = ' '.join(words[name_index: name_index + 2])

    # Print the fetched patient name
    #print("Patient Name:", patient_name)

    #############################################
    ######### patient id ######################
    ###########################################

    # # Iterate over the processed lines
    # for line in processed_lines:
    #     # # Check if "patient id" is present in the line
    #     # if "patient id" in line:
    #     #     # Split the line into words
    #     #     words = line.split()
    #     #     # Find the index of "patient id" in the words list
    #     #     index = words.index("patient") + 6
    #     #     # Fetch the two words following "patient id"
    #     #     patient_id = " ".join(words[index:index+2])
    #     #     # Print the patient ID
    #     #     # print("Patient ID:", patient_id)
    #     #     break  # If you only want to fetch the first occurrence, you can break the loop here
    #     if "patient id" in line:
    #         try:
    #             # Split the line into words
    #             words = line.split()
    #             # Find the index of "patient id" in the words list
    #             index = words.index("patient") + 6
    #             # Fetch the two words following "patient id"
    #             patient_id = " ".join(words[index:index+2])
    #             # Print the patient ID
    #             # print("Patient ID:", patient_id)
    #             break  # If you only want to fetch the first occurrence, you can break the loop here
    #         except IndexError:
    #             continue
    patient_id = ''  # Initialize the variable with a default value

    # Iterate over the processed lines
    for line in processed_lines:
        if "patient id" in line:
            try:
                words = line.split()
                index = words.index("patient") + 6
                patient_id = " ".join(words[index:index+2])
                break
            except IndexError:
                continue

    ######################################
    ############ age #####################
    ######################################

    import re
    age = ''

    # Define the pattern for matching the age
    age_pattern = r"age/gender\s+(\d+)\s+year"

    # Search for the age pattern in the processed lines
    age_matches = [re.search(age_pattern, line) for line in processed_lines]

    # Extract the age values from the matches
    ages = [match.group(1) for match in age_matches if match]

    # #Print the extracted ages
    # for age in ages:
    #     print(age)
    # Update the age variable with the first extracted age (if available)
    if ages:
        age = ages[0]

    ####################################
    ########## gender ##################
    ####################################

    # Iterate through the processed lines and check for gender information
    gender = None
    for line in processed_lines:
        if 'male' in line:
            gender = 'Male'
            break
        elif 'female' in line:
            gender = 'Female'
            break

    # Print the extracted gender
    #print('Gender:', gender)

    
    ################3######################
    ######## visit date ##################
    ###################################
    
    from datetime import datetime

    # Function to fetch and convert the visit date
    def fetch_visit_date(text):
        # Regex pattern to match the visit date in the format "dd month yyyy"
        pattern = r"\b(\d{2}\s+\w+\s+\d{4})\b"

        # Find matches of the pattern in the text
        matches = re.findall(pattern, text)

        if matches:
            # Extract the first match
            visit_date = matches[0]

            # Convert the visit date to the desired pattern "dd/mm/yy"
            converted_date = re.sub(r"\b(\d{2})\s+(\w+)\s+(\d{4})\b", r"\1/\2/\3", visit_date)

            # Convert the converted date string to a datetime object
            date_object = datetime.strptime(converted_date, "%d/%B/%Y")

            # Convert the datetime object to the desired format "dd/mm/yy"
            output_date = date_object.strftime("%d/%m/%y")

            return output_date

        return None

    # Fetch the visit date from the processed text
    visit_date = None
    for text in processed_text:
        visit_date = fetch_visit_date(text)
        if visit_date:
            break

    # Create a dictionary to store the result
    #visit_date= {"visit date": visit_date} if visit_date else {"visit date": "Not found"}

    # Print the result
    #print(visit_date)

    #################################
    ########### dr name ############
    ###############################

    import re

    # Preprocess the extracted text
    processed_text = [preprocess_text(text) for text in extracted_text]

    # Fetch the doctor name using regular expressions
    doctor_name = None
    pattern =  r"dr\.\s+(\w+)" 
    for line in processed_text:
        matches = re.search(pattern, line)
        if matches:
            doctor_name = matches.group(1)
            break

    # Print the doctor name if found
    # if doctor_name:
    #     print({"Dr Name": 'Dr.' + ' ' + doctor_name})
    # else:
    #     print("Doctor name not found.")

    ####################################
    ########### phone number ############
    ####################################

    import re

    # Search for phone number pattern after the word "phone"
    phone_number_pattern = r"phone (\d+)"
    phone_number = None

    for line in processed_text:
        match = re.search(phone_number_pattern, line)
        if match:
            phone_number = match.group(1)
            break

    # if phone_number:
    #     print({"Phone Number": phone_number})
    # else:
    #     print("Phone number not found.")

    # Fetch data from processed_text line by line and remove empty lines
    processed_lines = [line for line in processed_text if line.strip()]

    # Initialize the variable to store the index of the line containing "rx"
    rx_index = None

    try:
        # Find the index of the line containing "rx"
        rx_index = processed_lines.index('rx') + 1  # Add 1 to skip the line with "rx" itself
    except ValueError:
        pass  # "rx" not found, continue execution

    # Filter the lines after "rx" based on specific keywords
    filtered_lines = []
    keywords = ["day", "days", "month", "months", "dose", "doses", "mg" , "gm" , "w/v"]

    for line in processed_lines[rx_index:]:
        if any(keyword in line for keyword in keywords):
            filtered_lines.append(line)

    # Create a dataframe from the filtered lines
    df = pd.DataFrame(filtered_lines, columns=['Filtered Text'])
    print(df)   

     # ######################################
    # ########## medicine_name ############
    # #####################################     

    import json

    # ... previous code ...

    # Extract the duration from the remaining strings
    duration_results = []
    specific_words = ["capsule", "tablet", "tablets", "injection", "drop", "drops"]
    
    # Define the pattern for duration
    duration_pattern = r"\d+\s+(?:day|days|month|months|dose|doses)"

    for line in df['Filtered Text']:
        for word in specific_words:
            if word in line:
                medicine_name = line.split(word)[0].strip()
                remaining_string_parts = line.split(word)[1:]  # Get remaining parts of the line
                remaining_string = ' '.join(remaining_string_parts).strip()  # Join and strip the remaining parts
                duration_match = re.search(duration_pattern, remaining_string)

                if duration_match:
                    duration = duration_match.group()
                    duration_results.append({'Medicine Name': medicine_name, 'Duration': duration})
                else:
                    duration_results.append({'Medicine Name': medicine_name, 'Duration': 'Not found'})
                break


    # Create a dictionary to hold the patient name
    result = {
        "Patient Name": patient_name,
        "Patient ID": patient_id,
        "Gender": gender,
        "Age" : age,
        "Phone Number": phone_number,
        "Dr Name": 'Dr.' + ' ' + doctor_name,
        "Visit Date": visit_date if visit_date else " ",
        "Medicine Duration": duration_results
    }

    # Convert the dictionary to JSON string
    result_json = json.dumps(result)

    # Print the fetched patient name in JSON format
    #print(result_json)

    #print(processed_lines)
    return result_json #'\n'.join(patient_name)      

