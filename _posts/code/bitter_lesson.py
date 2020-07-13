import numpy as np
import matplotlib.pyplot as plt

alpha = 0.05
xs = np.arange(0,100,1)
print(xs)

def neg_exponential(x,alpha):
    return np.exp(-x * alpha)

def deriv(x,alpha):
    return -alpha * np.exp(-x * alpha)

def dderiv(x,alpha):
    return (alpha**2) * np.exp(-x * alpha)

def graph1():
    ys = [neg_exponential(x,alpha) for x in xs]
    ds = [deriv(x,alpha) for x in xs]
    dds = [dderiv(x,alpha) for x in xs]
    fig = plt.figure()
    plt.title("Performance Contour")
    plt.xlabel("Structure",fontsize = 16)
    plt.ylabel("Compute",fontsize = 16)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.plot(xs,ys,label="Contour of equal performance")
    #plt.plot(xs,dds,label="deriv")
    #plt.plot(xs,ds,label="second deriv")
    fig.savefig("bitter_lesson_1.jpg")
    plt.show()

def graph2():
    fig = plt.figure()
    plt.title("Performance vs compute with structure constant")
    plt.xlabel("Compute")
    plt.ylabel("Performance")
    plt.xticks([])
    plt.yticks([])
    ys = [-neg_exponential(x,0.1) for x in xs]
    plt.plot(xs,ys)
    plt.savefig("bitter_lesson_2.jpg")
    plt.show()

if __name__ == '__main__':
    graph1()
    graph2()