import os
import urllib
import urllib.request
import tarfile

def download():
    print("starting download...")
    
    raw_dir = '../../data/raw' #directory for raw download info

    #Stanford Dog Breed Dataset -- see citations in README
    url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'

    #Download tar file temporarily 
    
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        file_tmp = response.read()

    print("file downloaded")
        
    print("extracting .tar to ", raw_dir)
    tar = tarfile.open(file_tmp, "r:gz")
    for member in tar.getmembers():
        f = tar.extractall(raw_dir)
        content = f.read()
        f.save(

        
    print("deleting temporary .tar file")
    os.remove(file_tmp)

    

    
    return True
