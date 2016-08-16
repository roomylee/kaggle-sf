# coding: utf-8

# KAS (Kaggle Auto Submission)
# ===

# ### Author
# - Debugging Sparrow / dbgsprw@gmail.com
# - jangjunha / jangjunha113@gmail.com

# ### Requirements
# - requests
# - beautifulsoup4

from bs4 import BeautifulSoup
from zipfile import ZipFile
import requests


session = requests.Session()    
competition_name = ''


def init(user_email, password, _competition_name):
    global competition_name
    
    URL = "https://www.kaggle.com/account/login"
    login_data = {
        'UserName': user_email,
        'Password': password,
        'JavaScriptEnabled' : True
    }
    competition_name = _competition_name
    r = session.post(URL, data=login_data)
    
    test_URL = 'https://www.kaggle.com/c/%s/submissions/attach' % competition_name
    
    r = session.get(test_URL)
    if r.url == test_URL :
        print('Login Succeed')
        return True
    print('Login Failed')
    return False


#@filename = file_path
#@compress = True or False for compressing to .zip
def submission(csv_filename, compress):
    global competition_name
    
    filename = csv_filename
    if compress == True :
        with ZipFile(csv_filename, 'w') as myzip:
            filename = csv_filename + ".zip"
            myzip.write(filename)
    
    r_pre = session.get('https://www.kaggle.com/c/%s/submissions/attach' % competition_name)
    soup = BeautifulSoup(r_pre.content, 'html.parser')
    token = soup.find('input', {'name': '__RequestVerificationToken'})['value']
    competition_id = soup.find('input', {'name': 'CompetitionId'})['value']
    
    payload = {
        'CompetitionId': competition_id,
        '__RequestVerificationToken': token,
        'IsScriptVersionSubmission': 'False',
        'SubmissionDescription': 'This-is-description!'
    }
    files = {
        'SubmissionUpload': open(filename, 'rb')
    }
    
    r = session.post('https://www.kaggle.com/competitions/submissions/accept', data=payload, files=files)
    if r.status_code == 200:
        print("Submission Succeed")
        return True
    print("Submission Failed")
    return False
