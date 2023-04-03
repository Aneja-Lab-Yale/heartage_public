# Aneja Lab Misc Functions
# Core Functions
# Aneja Lab | Yale School of Medicine
# Sanjay Aneja, MD
# Created (4/20/20)
# Updated (4/20/20)

import os
import time, sys
import importlib

# Listdir No Hidden
# Created (4/20/20)
# Debugged (4/20/20)
# Created By (Sanjay Aneja, MD)
def listdir_nohidden(
        path
):
    """
    Description:
        This function generates a list of files within a folder (path) but ignores hidden files
        Requires import OS
    Param:
        Path (path) - parent folder
    Return:
        Generator which can create list of files in director
    """
    for f in os.listdir(path):
       if not f.startswith('.'):
            yield f

# Listdir Dicom
# Created (4/20/20)
# Debugged (4/20/20)
# Created By (Sanjay Aneja, MD)
def listdir_dicom(
        path

):
    """
    Description:
        This function generates a list of dicom files within a folder
        Requires Import OS
    Param:
        path (path)- folder with dicom images
    Return:
        Generator of list of dicom files within folder
    """
    for f in os.listdir(path):
        if f.endswith('.dcm'):
            yield f


# Absolute Paths
# Created (4/21/20)
# Debugged (4/21/20)
# Created By (Sanjay Aneja, MD)
def abs_path(

):
    """
    Description:
        This function changes current working directory to root folder requiring only absolute paths to be used.
        This is useful if importing data from different directoy not in repo.
        requires: Import OS
    Param:
        None
    Return:
        None
    """
    os.chdir('/')
    print('Note: Absolute Paths For Files')


# Import Check
# Created (5/4/20)
# Debugged (5/4/20)
# Created By (Sanjay Aneja, MD, Guneet Janda)
def install_check(
        requirements_file

):
    """
    Description:
        This function checks if you have required packages installed for program
        Requires import importlib
    Param:
        required_file = path to requirements.txt
    Return:
        All necessary packages installed vs Error
    """
    required_packages = {}
    with open(requirements_file, "r") as file:
        for line in file:
            key, value = line.strip().split("~=")
            required_packages[key] = value
    problem_packages = list()
    for package in required_packages:
        try:
            p = importlib.import_module(package)
        except ImportError:
            problem_packages.append(package)

    if len(problem_packages) is 0:
        print('All necessary packages are installed and ready for import.')
    else:
        print('ERROR The following required packages are not installed: ' + "\n"
              + ', '.join(problem_packages) + "\n" 'Please see requirements file')
    return

