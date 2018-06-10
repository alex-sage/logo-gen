import numpy as np


def f(y,t):
    if t < 0.5:
        t = 1-t
    if y < -1:
        return 0
    if -1 <= y <= 1-2*t:
        return 0.5*(y+1)*(y+1)
    if 1-2*t < y < 2*t-1:
        return 2*(1-t)*(y+t)
    if 2*t-1 <= y <= 1:
        return 2*(1-t)*(3*t-1)+(-0.5*y*y + y+0.5*(2*t-1)*(2*t-1)-(2*t-1))
    if y>1:
        return 2*(1-t)*(2*t)


def P_Y(y,t):
    return f(y,t)/(2.0*(1-t)*(2*t))


def Pi_Z(p):
    return 2*(p-0.5)


def Ytilde(y,t):
    return Pi_Z(P_Y(y,t))

# t = 0.5
# yy = np.arange(-1,1,0.05)
# Pyy = map(lambda y: P_Y(y,t), yy)
# yy_tilde = map(lambda y: Ytilde(y,t), yy)
#
# plt.plot(yy,yy_tilde)

#let z1 and z2 be two fixed samples
def lin_sample(z2, z1):
    for t in np.arange(0,1,0.1):
        yt = t*z1 + (1-t)*z2
        yt_tilde = map(lambda y: Ytilde(y,t), yt)