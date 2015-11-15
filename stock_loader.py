"""
stock_loader
~~~~~~~~~~~~
A library to load the stock price data.
The data from ruby jpstock script.
"""

#### Libraries
# Standard library
import cPickle
import gzip
import zipfile
# Third-party libraries
import numpy as np

def load_data():
    """
    training_data : number of elements is the number of sample
                    each element has 2 child elements
                    1st one is the stock prices for each day
                    2nd one is the stock prices of next year
    validation_data, test_data : same structure as training_data
    """

    training_data   = get_dataset(1,30001)
    validation_data = get_dataset(30001,32501)
    test_data       = get_dataset(32501,35001)
    return (training_data, validation_data, test_data)

def get_dataset(fromInt, toInt):
    inputs  = []
    results = []
    zf = zipfile.ZipFile('sample.zip', 'r')
    for i in range(int(fromInt),int(toInt)):
        file_no = "{0:08d}".format(i)
        input_data = load_from_file('sample_output/input_raw_'+file_no+'.csv', zf)
        input_ave = sum(input_data) / float(len(input_data))
        inputs.append(input_data)
        result_data = load_from_file('sample_output/input_ave_'+file_no+'.csv', zf)
        result_data = [float(x)/(2.0*input_ave) if float(x)/(2.0*input_ave)<0.95 else 0.95 for x in result_data]
        results.append(result_data)

    inputs = [np.reshape(x, (366, 1)) for x in inputs]
    results = [np.reshape(x, (12, 1)) for x in results]
    print "----"
    print len(inputs)
    print len(inputs[0])
    print len(results)
    print len(results[0])
    return zip(inputs, results)

def load_from_file(fileName, zf):
    outArray = []
    with zf.open(fileName) as sample:
        for line in sample:
            outArray.append(float(line))
    return outArray

