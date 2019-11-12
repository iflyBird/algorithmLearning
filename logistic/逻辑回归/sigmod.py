import numpy as np
import math
import matplotlib.pyplot as plt
x=np.linspace(-60,60,200);
y=[1/(1+math.e**(-x)) for x in x];
plt.plot(x,y);
plt.show(); 
