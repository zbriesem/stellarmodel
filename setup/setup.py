import os

def home_dir():
    homedir = '/Users/Briesemeister/stellarmodel'
    if not os.path.exists(homedir):
        raise Exception('something is very wrong: %s does not exist'%homedir)
    return homedir