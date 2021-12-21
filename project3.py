from matplotlib import pyplot as plt
import numpy as np
import random

class HouseLibrary:

    def __init__(self): # library objects
        self.features = [] # initialize list
        self.prices = [] # initialize list
        self.size = 0 # initialize size

    def loadData(self, filename: str):
        file = open(filename, 'r') # opens file
        file.readline() # skips first line because it is a title
        lines = file.readlines() # turns all lines in the file into list

        for line in lines: # scans through list
            attributes = line.split(',') # gets rid of the comma
            self.prices.append(float(attributes[-1]))  # append price
            self.features.append(attributes[1:-1])  # append features to list
            self.size = self.size + 1 # adds to the size of the list

        self.features = np.matrix(self.features)  # converts to a matrix
        self.features = self.features.astype(float)  # convert all the features to a float
        self.prices = np.matrix(self.prices).T  # convert to a matrix

    def size(self):
        return self.size

    def mean(self): # finds the mean
        sum = 0 # initialize
        counter = 0
        for i in self.prices: # scans through the list
            sum = sum + i # adds to sum
            counter = counter + 1
        mean = sum/counter # divides sum by number of prices
        return mean

    def min(self): # finds the minimum value
        return min(self.prices)

    def max(self): # finds the maximum value
        return max(self.prices)

    def StandardDev(self): # finds the standard deviation
        sum = 0 # initialize
        counter = 0
        for i in self.prices: # scans through the prices
            sum = sum + i # adds to the sum
            counter = counter + 1
        mean = sum/counter # finds the mean

        sum2 = 0
        for i in self.prices:
            sum2 = sum2 + (mean - i)**2 # subtracts the mean from the prices
        mean2 = sum2 / counter # finds the new mean
        sd = np.sqrt(mean2) # gets the square root
        return sd

    def histogram(self):
        # Creating histogram
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hist(self.prices, bins=[0, 2, 4, 6, 8]) # makes a histogram of the prices

        # Show plot
        plt.xlabel("Prices") # makes labels
        plt.ylabel("# of Prices")
        plt.title('Price Histogram')
        plt.show()

    def scatter(self, filename):
        file = open(filename, 'r') # opens file
        file.readline() # skips first line because it is a title
        lines = file.readlines() # turns all lines in the file into list

        GrLivArea = []
        BedroomAbvGr = []
        TotalBsmtSF = []
        FullBath = []

        for line in lines: # scans through list
            attributes = line.split(',')
            GrLivArea.append(attributes[12])
            BedroomAbvGr.append(attributes[16])
            TotalBsmtSF.append(attributes[8])
            FullBath.append(attributes[14])

        fig, ax = plt.subplots(nrows=2, ncols=3)

        a = np.array(GrLivArea)
        b = np.array(BedroomAbvGr)
        c = np.array(TotalBsmtSF)
        d = np.array(FullBath)

        #ax[0][0].scatter(a, a) # the commented out ones are not needed because the features on the x and y axis are
        ax[0][0].scatter(b, a)  # either the same or the x and y axis are flipped which do not show new correlation
        ax[0][1].scatter(c, a)
        ax[0][2].scatter(d, a)
        #ax[1][0].scatter(a, b)
        #ax[1][1].scatter(b, b)
        ax[1][0].scatter(c, b)
        ax[1][1].scatter(d, b)
        #ax[2][0].scatter(a, c)
        #ax[2][1].scatter(b, c)
        #ax[2][2].scatter(c, c)
        ax[1][2].scatter(d, c)
        #ax[3][0].scatter(a, d)
        #ax[3][1].scatter(b, d)
        #ax[3][2].scatter(c, d)
        #ax[3][3].scatter(d, d)

        plt.tight_layout()
        plt.show()

    def pred(self, w):
        mult = self.features * w # multiplies the matrix of features by the according weight
        pred = sum(mult) # takes the sum
        return pred

    def loss(self, pred):
        Y = np.sum(np.square(pred - self.prices)) # squares the difference and gets the sum
        loss = Y * (1/self.size) # divides by the size
        return loss

    def gradient(self, pred):
        Y = pred - self.prices # gets the difference
        XT = self.features.transpose() # gets the transpose of all the features
        gradient = (2/self.size) * (XT * Y) # multiplies the features by the difference then multiplies by (2/size)
        return gradient

    def update(self, w, a, gradient):
        return w - (a * gradient) # updates the weights

if __name__ == '__main__':

    houseLib = HouseLibrary()
    print("\n-------Question 2-------")

    print("\nNumber of Records")
    File = "train.csv"
    houseLib.loadData(File)
    print(houseLib.size)

    print("\nMean Value of Prices")
    print(houseLib.mean())

    print("\nMinimum Value of Prices")
    print(houseLib.min())

    print("\nMaximum Value of Prices")
    print(houseLib.max())

    print("\nStandard Deviation")
    print(houseLib.StandardDev())

    print("\n-------Question 3-------")
    #print(houseLib.histogram()) # uncomment for histogram

    print("\n-------Question 4-------")
    #print(houseLib.scatter(File)) # uncomment for scatter
    print("In the scatter plot we found some had correlation and other did not. To accelerate the training")
    print("process we need to find the gradient of the mean squared error in order to adjust our weights. ")

    #print("\n-------Question 5-------")
    w = np.matrix([np.random.uniform(0, 1) for i in range(25)]).transpose() # create a matrix of random weights
    original_w = w # original weights

    print("\n-------Question 10-------")
    # CODE FOR a = 0.2

    # a1 = 0.2  # learning rate
    # loss_l1 = []  # list of loss
    # for i in range(500):
    #     pred1 = houseLib.pred(w)  # prediction price
    #     loss1 = houseLib.loss(pred1)  # calls the loss function
    #     gradient1 = houseLib.gradient(pred1)  # calls the gradient gradient function
    #     w = houseLib.update(w, a1, gradient1)  # calls the update weights function
    #
    #     loss_l1.append(loss1)  # append loss to list
    #     if loss1 < 1:
    #         break

    print("My algorithm does not find the minimal MSE, actually when i run the algorithm there is an overload in my ")
    print("loss function which does not allow it to find the minimal MSE.")

    print("\n-------Question 11-------")

    w = original_w
    a2 = 10 ** -11  # learning rate
    loss_l2 = []  # list of loss
    for i in range(500):
        pred2 = houseLib.pred(w)  # calls the prediction function
        loss2 = houseLib.loss(pred2)  # calls the loss function
        gradient2 = houseLib.gradient(pred2)  # calls the gradient function
        w = houseLib.update(w, a2, gradient2)  # calls the update weights function

        loss_l2.append(loss2)  # append loss to list
        if loss2 < 0.05:
            break

    plt.plot(loss_l2)  # plots list of loss

    w = original_w
    a3 = 10 ** -12  # learning rate
    loss_l3 = []  # list of loss
    for i in range(500):
        pred3 = houseLib.pred(w)  # calls the prediction function
        loss3 = houseLib.loss(pred3)  # calls the loss function
        gradient3 = houseLib.gradient(pred3)  # calls the gradient function
        w = houseLib.update(w, a3, gradient3)  # calls the update weights function

        loss_l3.append(loss3)  # append loss to list
        if loss3 < 0.05:
            break

    plt.plot(loss_l3)  # plot list of loss
    plt.legend(['a = 10^-11', 'a = 10^-12']) # creates legend
    plt.show()

    print("\n-------Question 12-------")
    print("When a = 10^-11 it converges faster because it is taking larger step sizes than 10^-12 without overshooting")

    print("\n-------Question 13-------")
    print("The MSE for a = 10^-11 was around 0.2 and the MSE for a = 10-12 was around 0.5. My algorithm worked a lot better ")
    print("with the training set rather than the testing. As we can see it converges to the MSE a lot quicker for the ")
    print("training set.")