import numpy as np

k1 = 1
L = 1
w = np.sqrt(k1)

sinwL = np.sin(w*L)
coswL = np.cos(w*L)
shwL  = np.sinh(w*L)
chwL  = np.cosh(w*L)

mqf = np.array([[coswL, sinwL/w], [-w*sinwL, coswL]])
mqd = np.array([[chwL, shwL/w], [w*shwL, chwL]])
O = np.zeros((2,2))
MQ_thick = np.block([[mqf, O], [O, mqd]])
MQ_thin = np.eye(4); MQ_thin[1,0] = -w*sinwL; MQ_thin[3,2] = w*shwL
dl1 = np.array([[1, L], [0, 1]])
DL = np.block([[dl1, O], [O, dl1]])

TM1 = np.matmul(MQ_thick, DL)
TM2 = np.matmul(MQ_thin,  DL)


