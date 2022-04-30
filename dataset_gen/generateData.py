import numpy as np
import cv2
from time import time
from xml.dom import minidom
from tqdm import tqdm
import random

# parse an xml file by name
file = minidom.parse('char.xml')

# use getElementsByTagName() to get tag
models = file.getElementsByTagName('image')

mykeys = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","Ñ","É","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","é",":","(",")","!","&","-",",","'","?",".","£",'"']

myvalues =np.arange(len(mykeys))

mydict = dict(zip(mykeys,myvalues))

# print(f"Mydict: {mydict}")

def combine_horizontally(image_names, labels, id, padding=5):
    images = []
    label = []
    max_height = 0  # find the max height of all the images
    total_width = 0  # the total width of the images (horizontal stacking)
    for name in image_names:
        # open all images and find their sizes
        img = cv2.imread(name)
        images.append(img)
        image_height = img.shape[0]
        image_width = img.shape[1]
        if image_height > max_height:
            max_height = image_height
        # add all the images widths
        total_width += image_width
    height = int(max_height/2)
    width = int(total_width/len(image_names))
    dim = (width, height)
    # create a new array with a size large enough to contain all the images
    # also add padding size for all the images except the last one
    final_image = np.zeros((int(max_height/2)+(padding*2), (len(image_names)-1)*(padding*2)+ total_width, 3), dtype=np.uint8)
    current_x = padding  # keep track of where your current image was last placed in the x coordinate\
    for i, image in enumerate(images):
        # add an image to the final array and increment the x coordinate
        image = cv2.resize(image, dim)
        height = image.shape[0]
        width = image.shape[1]
        centx = (width/2+current_x)/np.shape(final_image)[1]
        centy = (height/2+padding)/np.shape(final_image)[0]
        normWidth = width/np.shape(final_image)[1]
        normHeight = height/np.shape(final_image)[0]
        # print(f"Width: {width}, Height: {height}, Max width: {np.shape(final_image)[1]}, Max height: {np.shape(final_image)[0]}")
        # print(f"unnormcentx:{(width/2+current_x)}, unnormcenty: {(height/2+padding)}")
        # print(f"label: {labels[i]}, centx:{centx}, centy: {centy}, norm width: {normWidth}, norm height: {normHeight}")
        label.append([labels[i], centx, centy, normWidth, normHeight])
        final_image[padding:height+padding,current_x :width+current_x, :] = image
        #add the padding between the images
        current_x += width+padding
    
    with open('train/'+id+'.txt', 'w') as f:
        for l in label:
            f.write(f"{mydict[l[0]]} {l[1]} {l[2]} {l[3]} {l[4]}\n")
    cv2.imwrite('train/'+id+'.png', final_image)



if __name__ == "__main__":
    
    datasize = 10**4
    choices = []
    for id in tqdm(range(datasize)):
        random.seed(time())
        imlen = random.randint(5,10)
        # print(f"models len: {len(models)}")
        for i in range(imlen):
            choice = random.randint(0,len(models)-1)
            if mydict[models[choice].attributes['tag'].value] > 35:
                i -= 1
            else: choices.append(choice)
        # print(f"imlen: {imlen}, choices: {choices}")
        combine_horizontally([models[i].attributes['file'].value for i in choices], [models[j].attributes['tag'].value for j in choices], str(id))
    print("Saved dataset.")


