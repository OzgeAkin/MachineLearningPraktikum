from struct import unpack
from numpy import zeros, uint8, ravel

def imagefeatures_and_labels (datatype):
    
    input_data = ''
    input_labels = ''
    if(datatype == 'train'):
        input_data = open('train-images.idx3-ubyte', 'rb')
        input_labels =  open('train-labels.idx1-ubyte', 'rb')
    if(datatype == 'test'):
        input_data = open('t10k-images.idx3-ubyte', 'rb')
        input_labels =  open('t10k-labels.idx1-ubyte', 'rb')


    s1, s2, s3, s4 = input_data.read(4), input_data.read(4), input_data.read(4), input_data.read(4)
    magicnum = unpack('>I',s1)[0] #2051
    totalimage = unpack('>I',s2)[0] #60000
    rows = unpack('>I',s3)[0] #28
    cols = unpack('>I',s4)[0] #28
    
    s1, s2 = input_labels.read(4), input_labels.read(4)
    totallabel = unpack('>I',s2)[0]
    
    #put the data into numpy array
    images = zeros((totalimage, rows, cols), dtype = uint8)
    image_features = zeros((totalimage, rows*cols), dtype = uint8)
    labels = zeros((totallabel, 1), dtype = uint8)
    if( totalimage == totallabel):
        for i in range(totalimage):
            for row in range(rows):
                for col in range(cols):
                    pixel = unpack('>B', input_data.read(1))[0] #1 byte
                    images[i][row][col] = pixel
            image_features[i] = ravel(images[i])
            labels[i] =  unpack('>B', input_labels.read(1))[0]
    input_data.close()
    input_labels.close()
    return (image_features, ravel(labels))
    