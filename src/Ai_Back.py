import numpy as np
import scipy.signal as ss
import bisect
from itertools import zip_longest


def LeakyReLU(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def D_LeakyReLU(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def sigmoid(x):
    # Works for single numbers, Python lists, or NumPy arrays
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# Basic defnitions


def Cost_D(expect, result):
    return 2 * (expect - result)

# def Cost_D(expect, result):
#     Value = np.where((result > 0) & (expect <= result) | ((result <= 0) & (expect >= result)), 2 * (expect - result), 6 * (expect - result))
#     return Value

def ReLU(x):
    return np.maximum(0, x)

def D_ReLU(x):
    return (x > 0).astype(float)

def Split(List, Split):
    Accum = np.cumsum(Split)
    Accum = Accum[:-1]
    List_Split = np.split(List, Accum)
    return List_Split

def Interput(branch_subject, inject, stride):
    out = []
    inject = np.ones((int(np.size(branch_subject)/stride), np.size(inject))) * inject
    branch_subject = np.split(branch_subject, np.size(branch_subject)/stride)
    for values in zip([branch_subject,inject]):
        for v in values:
            out.append(v)
    return np.hstack(out)

def Interleave_Strided(branch_subject, inject, stride):
    out = []
    branch_subject = np.split(branch_subject, np.size(branch_subject)/stride)
    for values in zip([branch_subject,inject]):
        for v in values:
            out.append(v)
    return np.hstack(out)


def Interleave(branches):
    # branches = [a_array, b_array, c_array, ...]
    out = []
    for values in zip(*branches):
        for v in values:
            out.append(v)
    return np.array(out)

def clip_grad_arrays(grad, max_norm=1):
    arr = np.array(grad, dtype=float)
    norms = np.linalg.norm(arr, axis=1)
    scale = (max_norm / norms)[:, np.newaxis]
    Bool = (max_norm < norms)
    arr[Bool, :] = arr[Bool, :] *scale[Bool]
    return arr

def clip_grad_arrays_matrixs(grad, max_norm=1):
    arr = grad
    arrayed = np.reshape(arr,[2,-1])
    norms = np.linalg.norm(arrayed, axis=1)
    scale = (max_norm / norms)[:, np.newaxis, np.newaxis]
    Bool = (max_norm < norms)
    arr[Bool, :, :] = arr[Bool, :, :] *scale[Bool]
    return arr

#AI
def Start_Z_Batch(From_To, A, Weight, Bias):
    return A @ Weight.T + Bias

def Next_Z_Batch(From_To, Z, Weight, Bias):
    A = LeakyReLU(Z)
    return A @ Weight.T + Bias



def Full_Next_Z(Prior_Split, Next_Split,Z, Weight, Bias):
    A = LeakyReLU(Z)
    k=np.size(Prior_Split)
    A_Split = Split(A, Prior_Split)
    Weight_Split = Split(Weight, Next_Split * Prior_Split)
    Weight_Matrix = [Weight_Split[n].reshape(Next_Split[n],Prior_Split[n]) for n in range(k)]
    A_Arrays = [Weight_Matrix[n] @ A_Split[n] for n in range(k)]
    return [np.concatenate(A_Arrays) + Bias]

def Back_Batch(Batch_Size, From_To, X_Currunt_Set, X_Next_Set, Weight_Next, PDLX_Next_Set):
    #Bias
    Part_Bias_Grad= D_LeakyReLU(X_Next_Set) * PDLX_Next_Set
    #Weight
    Part_Weight_Grad_Matrix= LeakyReLU(X_Currunt_Set)[:,np.newaxis,:] * Part_Bias_Grad[:, :, np.newaxis]
    #Act_Sensitivity
    Part_PDLX_Prior = Part_Bias_Grad @ Weight_Next
    return [Part_PDLX_Prior, np.mean(Part_Weight_Grad_Matrix, axis = 0), np.mean(Part_Bias_Grad, axis = 0)]

def First_Layer_Batch(Batch_Size, From_To, X_Currunt_Set, X_Next_Set, Weight_Next, PDLX_Next_Set):
    #Bias
    Part_Bias_Grad= D_LeakyReLU(X_Next_Set) * PDLX_Next_Set
    #Weight
    Part_Weight_Grad_Matrix= X_Currunt_Set[:,np.newaxis,:] * Part_Bias_Grad[:, :, np.newaxis]
    #Act_Sensitivity
    Part_PDLX_Prior = Part_Bias_Grad @ Weight_Next
    return [Part_PDLX_Prior, np.mean(Part_Weight_Grad_Matrix, axis = 0), np.mean(Part_Bias_Grad, axis = 0)]


# def First_Layer_Batch(Batch_Size, From_To, X_Currunt_Set, X_Next_Set, Weight_Next, PDLX_Next_Set):
#     #Bias
#     Part_Bias_Grad= D_LeakyReLU(X_Next_Set) * PDLX_Next_Set
#     #Weight
#     Part_Weight_Grad_Matrix= X_Currunt_Set[:,np.newaxis,:] * Part_Bias_Grad[:, :, np.newaxis]
#     Part_Weight_Grad = Part_Weight_Grad_Matrix.reshape(Batch_Size, -1)
#     #Act_Sensitivity
#     Part_PDLX_Prior = Part_Bias_Grad @ Weight_Next
#     return [Part_PDLX_Prior, np.mean(Part_Weight_Grad, axis = 0), np.mean(Part_Bias_Grad, axis = 0)]

def Back_Initial_Batch(Batch_Size, From_To, Result_Set, X_Last_Set, Expected_Set, Weight_Last):
    Part_PDLX_Next = Cost_D(Expected_Set, Result_Set)
    #Bias
    Part_Bias_Grad= Part_PDLX_Next
    #Weight
    Part_Weight_Grad_Matrix= LeakyReLU(X_Last_Set)[:,np.newaxis,:] * Part_Bias_Grad[:, :, np.newaxis]
    #Act_Sensitivity
    Part_PDLX_Prior = Part_Bias_Grad @ Weight_Last
    return [Part_PDLX_Prior, np.mean(Part_Weight_Grad_Matrix, axis = 0), np.mean(Part_Bias_Grad, axis = 0)]


def Back(Prior_Split, Next_Split, X_Currunt, X_Next, Weight_Next, PDLX_Next):
    X_Next_Split = Split(X_Next, Next_Split)
    PDLX_Next_Split = Split(PDLX_Next, Next_Split)
    Weight_Next_Split = Split(Weight_Next, Next_Split * Prior_Split)
    X_Currunt_Split = Split(X_Currunt, Prior_Split)
    From_To = [np.array([Prior_Split[n],Next_Split[n]]) for n in range(np.size(Prior_Split))]
    k=[Part_Back(From_To[n], X_Currunt_Split[n],X_Next_Split[n], Weight_Next_Split[n], PDLX_Next_Split[n]) for n in range(np.size(Prior_Split))]
    return [np.concatenate(pair) for pair in zip(*k)]

def Part_Back(From_To, Part_X_Currunt, Part_X_Next, Part_Weight_Next, Part_PDLX_Next):
    #Bias
    Part_Bias_Grad= D_LeakyReLU(Part_X_Next) * Part_PDLX_Next
    #Weight
    ones = np.ones((From_To[1],From_To[0]))
    Part_Weight_Grad_Matrix= ones * LeakyReLU(Part_X_Currunt) * Part_Bias_Grad[:, np.newaxis]
    Part_Weight_Grad = Part_Weight_Grad_Matrix.reshape(-1)
    #Act_Sensitivity
    Part_PDLX_Prior = Part_Weight_Next.reshape(From_To[1],From_To[0]).T @ Part_Bias_Grad
    return [Part_PDLX_Prior, Part_Weight_Grad, Part_Bias_Grad]


#Rolling Start is needed.
def Rolling(Prior_Split, Next_Split,Z, Weight, Bias):
    A = LeakyReLU(Z)
    k = np.size(Prior_Split)
    A_Split = Split(A, Prior_Split)
    Weight_Split = Split(Weight, [Prior_Split[n]-Next_Split[n]+1 for n in range(np.size(Prior_Split))])
    A_Arrays = [np.convolve(A_Split[n], Weight_Split[n][::-1], mode="valid") for n in range(k)]
    return np.concatenate(A_Arrays) + Bias

def Part_Rolling_Back(From_To, Part_X_Currunt, Part_X_Next, Part_Weight_Next, Part_PDLX_Next):

    #Bias
    Part_Bias_Grad_Array = Part_PDLX_Next * D_LeakyReLU(Part_X_Next)
    Part_Bias_Grad = np.sum(Part_Bias_Grad_Array)

    #Weight
    Flip_Part_Bias_Grad_Array = np.flip(Part_Bias_Grad_Array)
    Part_Weight_Grad = np.convolve(LeakyReLU(Part_X_Currunt), Flip_Part_Bias_Grad_Array, mode="valid")

    #Act_Sensitivity
    Part_PDLX_Prior = np.convolve(Part_Bias_Grad_Array, Part_Weight_Next, mode='full')
    return [Part_PDLX_Prior, Part_Weight_Grad, Part_Bias_Grad]

def Rolling_Back(Prior_Split, Next_Split, X_Currunt, X_Next, Weight_Next, PDLX_Next):
    PDLX_Next_Split = Split(PDLX_Next, Next_Split)
    Weight_Next_Split = Split(Weight_Next, [Prior_Split[n]-Next_Split[n]+1 for n in range(np.size(Prior_Split))])
    X_Currunt_Split = Split(X_Currunt, Prior_Split)
    X_Next_Split =  Split(X_Next, Next_Split)
    From_To = [np.array([Prior_Split[n],Next_Split[n]]) for n in range(np.size(Prior_Split))]
    k=[Part_Rolling_Back(From_To[n], X_Currunt_Split[n], X_Next_Split[n], Weight_Next_Split[n], PDLX_Next_Split[n]) for n in range(np.size(Prior_Split))]
    return [np.concatenate(pair) for pair in zip(*k)]


#Data extraction

def Average_Out(Data, Window):
    reshaped = Data.reshape(int(np.size(Data)/Window), Window)
    return reshaped.mean(axis=1)


#Delete the outer value.
def Smoothing(Data,Window):
    Weight = np.ones(Window)/Window
    return np.convolve(Data, Weight, mode="valid")

def Smoothing_2(Data,Window):
    return ss.savgol_filter(Data,Window,2)



def Get_Rel(Data,Split):
    Peak = np.array([])
    Data = np.array_split(Data, Split)
    Data_Ac = 0
    for n in range(Split):
        Prominence_Value = Data[n].mean()*0.05
        Rel_Points = ss.find_peaks(Data[n], prominence=Prominence_Value, distance=5)
        Peak = np.append(Peak, Rel_Points[0]+Data_Ac)
        Data_Ac += np.size(Data[n])
    Trough = np.array([])
    Data_Ac = 0
    for n in range(Split):
        Prominence_Value = Data[n].mean()*0.05
        Rel_Points = ss.find_peaks([-k for k in Data[n]], prominence=Prominence_Value, distance=5)
        Trough = np.append(Trough, Rel_Points[0]+Data_Ac)
        Data_Ac += np.size(Data[n])
    return [np.sort(np.append(Peak, Trough))]


def Get_Rel_2(Data):
    Rel = np.sort(np.concatenate((ss.argrelmin(Data, order = 5), ss.argrelmax(Data, order = 5)), axis=1))
    result = Rel[(3 < Rel) & (Rel < np.size(Data)-3)]
    return result



#Together!
#Fix
Acc_Open_Dates = [0, 252, 503, 754, 1007, 1259, 1511, 1763, 2013, 2265, 2517, 2769, 3021, 3272, 3523, 3775, 4028, 4280, 4531, 4781, 5033, 5283]
Open_Dates = [252, 251, 251, 253, 252, 252, 252, 250, 252, 252, 252, 252, 251, 251, 252, 253, 252, 251, 250, 252, 250]
Open_Dates_Month = [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 19, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 19, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 19]
Acc_Open_Dates_Month = [0, 21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252, 273, 294, 315, 336, 357, 378, 399, 420, 441, 462, 483, 503, 524, 545, 566, 587, 608, 629, 650, 671, 692, 713, 734, 754, 775, 796, 817, 838, 859, 880, 901, 922, 943, 964, 985, 1007, 1028, 1049, 1070, 1091, 1112, 1133, 1154, 1175, 1196, 1217, 1238, 1259, 1280, 1301, 1322, 1343, 1364, 1385, 1406, 1427, 1448, 1469, 1490, 1511, 1532, 1553, 1574, 1595, 1616, 1637, 1658, 1679, 1700, 1721, 1742, 1763, 1784, 1805, 1826, 1847, 1868, 1889, 1910, 1931, 1952, 1973, 1994, 2013, 2034, 2055, 2076, 2097, 2118, 2139, 2160, 2181, 2202, 2223, 2244, 2265, 2286, 2307, 2328, 2349, 2370, 2391, 2412, 2433, 2454, 2475, 2496, 2517, 2538, 2559, 2580, 2601, 2622, 2643, 2664, 2685, 2706, 2727, 2748, 2769, 2790, 2811, 2832, 2853, 2874, 2895, 2916, 2937, 2958, 2979, 3000, 3021, 3042, 3063, 3084, 3105, 3126, 3147, 3168, 3189, 3210, 3231, 3252, 3272, 3293, 3314, 3335, 3356, 3377, 3398, 3419, 3440, 3461, 3482, 3503, 3523, 3544, 3565, 3586, 3607, 3628, 3649, 3670, 3691, 3712, 3733, 3754, 3775, 3796, 3817, 3838, 3859, 3880, 3901, 3922, 3943, 3964, 3985, 4006, 4028, 4049, 4070, 4091, 4112, 4133, 4154, 4175, 4196, 4217, 4238, 4259, 4280, 4301, 4322, 4343, 4364, 4385, 4406, 4427, 4448, 4469, 4490, 4511, 4531, 4552, 4573, 4594, 4615, 4636, 4657, 4678, 4699, 4720, 4741, 4762, 4781, 4802, 4823, 4844, 4865, 4886, 4907, 4928, 4949, 4970, 4991, 5012, 5033, 5054, 5075, 5096, 5117, 5138, 5159, 5180, 5201, 5222, 5243, 5264, 5283]

def Matching_Data(Index):
    k = bisect.bisect(Acc_Open_Dates_Month, Index)
    return k-1

#something: 251/12-->[21,21,21,21,...20]
#Did 252/12 = 21 in order to make it maximize. Assumes that the last month takes
"""
z = []
for i in Open_Dates:
    for j in range(11):
        z.append(21)
    z.append(i - 231)
"""


def Month_To_Day(Month):
    return Acc_Open_Dates_Month[Month]




#How will I match if the data doesn't exactly a 252 day a year with avg of 21 years?
def Matching_Data_1(Data, End_Time, Duration_Day):
    Range_Month = np.array([round((End_Time +1)*12/252)-1, round((End_Time - Duration_Day +1)*12/252) -1])
    Month_Between = np.arange(Range_Month[0], Range_Month[1])
    Data_Within_Range = Data[Month_Between]
    return Data_Within_Range
