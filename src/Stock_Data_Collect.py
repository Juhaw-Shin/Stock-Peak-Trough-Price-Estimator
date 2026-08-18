import yfinance as yf
import numpy as np
import Ai_Back as B
import Stock_Related as SR
import scipy.signal as ss

#Recent Data collection.

ticker = "MMM ABT ADBE AES AFL A APD ALL MO AEE AEP AXP AIG AMGN ADI AON APA AAPL AMAT ADM T ADSK ADP AZO AVY BALL BAC BAX BDX BBY BIIB BNY BA BSX BMY BF-B CPB COF CAH CCL CAT COR CNP SCHW CVX CI CINF CTAS CSCO C CLX CMS KO CL CMCSA CAG COP ED GLW COST CSX CMI CVS DHR DRI DE DVN DOV DTE DUK ETN EBAY ECL EIX EA ELV EMR ETR EOG EFX EQR EXC XOM FDX FITB FE F BEN GE GEN GD GIS GPC GILD GL GS HAL HIG HAS HSY HD HON HPQ HBAN IBM ITW INTC IFF IP INTU JNJ JPM KEY KMB KLAC KR LH LLY LIN LMT L LOW MTB MAR MRSH MAS MKC MCD MCK MDT MRK MET MU MSFT TAP MCO MS MSI NTAP NEM NEE NKE NI NSC NTRS NOC NUE NVDA OXY OMC ORCL PCAR PH PAYX PEP PFE PNW PNC PPG PPL PFG PG PGR PLD PRU PEG PHM QCOM DGX RTX RF RVTY ROK SPGI SLB SHW SPG SNA SO LUV SWK SBUX STT SYK SYY TPR TGT TXN TXT TMO TJX TRV TFC USB UNP UPS UNH VLO VZ VTRS VMC GWW WMT DIS WM WAT WFC WY WMB XEL YUM ZBH"

Ticker_List = ['MMM', 'ABT', 'ADBE', 'AES', 'AFL', 'A', 'APD', 'ALL', 'MO', 'AEE', 'AEP', 'AXP', 'AIG', 'AMGN', 'ADI', 'AON', 'APA', 'AAPL', 'AMAT', 'ADM', 'T', 'ADSK', 'ADP', 'AZO', 'AVY', 'BALL', 'BAC', 'BAX', 'BDX', 'BBY', 'BIIB', 'BNY', 'BA', 'BSX', 'BMY', 'BF-B', 'CPB', 'COF', 'CAH', 'CCL', 'CAT', 'COR', 'CNP', 'SCHW', 'CVX', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CLX', 'CMS', 'KO', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'GLW', 'COST', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DE', 'DVN', 'DOV', 'DTE', 'DUK', 'ETN', 'EBAY', 'ECL', 'EIX', 'EA', 'ELV', 'EMR', 'ETR', 'EOG', 'EFX', 'EQR', 'EXC', 'XOM', 'FDX', 'FITB', 'FE', 'F', 'BEN', 'GE', 'GEN', 'GD', 'GIS', 'GPC', 'GILD', 'GL', 'GS', 'HAL', 'HIG', 'HAS', 'HSY', 'HD', 'HON', 'HPQ', 'HBAN', 'IBM', 'ITW', 'INTC', 'IFF', 'IP', 'INTU', 'JNJ', 'JPM', 'KEY', 'KMB', 'KLAC', 'KR', 'LH', 'LLY', 'LIN', 'LMT', 'L', 'LOW', 'MTB', 'MAR', 'MRSH', 'MAS', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MU', 'MSFT', 'TAP', 'MCO', 'MS', 'MSI', 'NTAP', 'NEM', 'NEE', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NUE', 'NVDA', 'OXY', 'OMC', 'ORCL', 'PCAR', 'PH', 'PAYX', 'PEP', 'PFE', 'PNW', 'PNC', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PHM', 'QCOM', 'DGX', 'RTX', 'RF', 'RVTY', 'ROK', 'SPGI', 'SLB', 'SHW', 'SPG', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'SYK', 'SYY', 'TPR', 'TGT', 'TXN', 'TXT', 'TMO', 'TJX', 'TRV', 'TFC', 'USB', 'UNP', 'UPS', 'UNH', 'VLO', 'VZ', 'VTRS', 'VMC', 'GWW', 'WMT', 'DIS', 'WM', 'WAT', 'WFC', 'WY', 'WMB', 'XEL', 'YUM', 'ZBH']

Return = {}


Data = np.load("NPZ/New/Price_New.npz")

Return = {}

for x in Ticker_List:
    Return[x]=np.union1d(B.Get_Rel(Data[x+"/Price"],42), B.Get_Rel(Data[x+"/Price"],48))


np.savez("NPZ/New/Ext+New.npz", **Return)







"""
Data_1 = np.load("NPZ/Ticker/Price/Price_2.npz")
Data_2 = np.load("NPZ/Ticker/Price/Price_Smoothed_2.npz")
Data_3 = np.load("NPZ/Ticker/Volume/Volume_2.npz")
Data_4 = np.load("NPZ/Ticker/Volume/Volume_Smoothed_2.npz")

Return = {}

for x in Ticker_List:
    Return_2 = 100*np.diff(Data_1[x])/Data_1[x][:-1]
    Return[x] = Return_2

np.savez("NPZ/Ticker/Price/Return_2.npz", **Return)

Return = {}
for x in Ticker_List:
    Return_2_Smoothed = 100*np.diff(Data_2[x])/Data_2[x][:-1]
    Return[x] = Return_2_Smoothed

np.savez("NPZ/Ticker/Price/Return_Smoothed_2.npz", **Return)

Return = {}
for x in Ticker_List:
    Volume_Rate_2 = 100*np.diff(Data_3[x])/Data_3[x][:-1]
    Return[x] = Volume_Rate_2

np.savez("NPZ/Ticker/Volume/Volume_Rate_2.npz", **Return)

Return = {}
for x in Ticker_List:
    Volume_Rate_2_Smoothed = 100*np.diff(Data_4[x])/Data_4[x][:-1]
    Return[x] = Volume_Rate_2_Smoothed

np.savez("NPZ/Ticker/Volume/Volume_Rate_Smoothed_2.npz", **Return)
"""


"""
for Ticker in Ticker_List:
    Ext_T_Except_0 = np.delete(SR.Ext[Ticker],0)
    Ext_T_Except_0 = np.append(Ext_T_Except_0, 5283)
    for Month in range(6,252):
        Within_Indices = np.where((Ext_T_Except_0 >= B.Month_To_Day(Month)) &(Ext_T_Except_0 < B.Month_To_Day(Month+1)))[0]
        for Ext_index in Within_Indices: #for month =6, it has value between 6 and 7. Look at Accu_Open_Date_Month:[0,21,42, 63...]. Also, it includes the value of starting(such as 0) but not ending (like 21) because the index is starting from 0 in ext.
            Ext_Place = Ext_T_Except_0[Ext_index]
            for Duration in range(4, int(Ext_T_Except_0[Ext_index+1]-Ext_T_Except_0[Ext_index])): #At least 5 days. Will do Duration+1
                                                                                                  #Fall also not near 5 by distance.
                Return[str(Month) + "-" + str(Ext_Place)+"-"+ str(Duration)] = SR.Recent_Data(Ticker, Ext_index+1, Duration+1) #Duration + 1 because the duration is not equal to index of Recent_Data range now. include both first extrima, second.
    np.savez("NPZ/Ticker/Month-Ext/"+Ticker+".npz", **Return)
"""


"""
for x in Ticker_List:
    Price = Data1[x+"/Volume"]
    Return[x] = Price[:len(Price)//2 * 2].reshape(-1, 2).mean(axis=1)
np.savez("NPZ/Ticker/Volume_2.npz", **Return)

for x in Ticker_List:
    Price = Data2[x]
    Return[x] = Price[:len(Price)//2 * 2].reshape(-1, 2).mean(axis=1)

np.savez("NPZ/Ticker/Volume_Smoothed_2.npz", **Return)

for x in Ticker_List:
    Price = Data3[x+"/Price"]
    Return[x] = Price[:len(Price)//2 * 2].reshape(-1, 2).mean(axis=1)

np.savez("NPZ/Ticker/Price_2.npz", **Return)

for x in Ticker_List:
    Price = Data4[x]
    Return[x] = Price[:len(Price)//2 * 2].reshape(-1, 2).mean(axis=1)

np.savez("NPZ/Ticker/Price_Smoothed_2.npz", **Return)
"""


#Volume_Smooth
"""

Data = np.load("NPZ/Ticker/Volume.npz")

print(pair_means)

for x in Ticker_List:
    Price = Data[x+"/Volume"]
    Return[x] = ss.savgol_filter(Price, window_length=21, polyorder=3)


np.savez("NPZ/Ticker/Volume_Smoothed.npz", **Return)
"""


#Errored

"""
#arr[(arr >= 2) & (arr <= 3)]
#Month-Ext = 121-150
#month = 6 <== 여섯번째 달 안에 ext
#in Mo_Last data, have month-1 to find the right value. It is like: Last_month ==> Month Ext is within
for Ticker in ['MMM']:
    Ext_T_Except_0 = np.delete(SR.Ext[Ticker],0)
    for Month in range(6,233):
        Ext_Within = Ext_T_Except_0[(Ext_T_Except_0 >= B.Month_To_Day(Month)) & (Ext_T_Except_0 < B.Month_To_Day(Month+1))]
        for Ext_index, Ext_Place in enumerate(Ext_Within): #for month =6, it has value between 6 and 7. Look at Accu_Open_Date_Month:[0,21,42, 63...]. Also, it includes the value of starting(such as 0) but not ending (like 21) because the index is starting from 0 in ext.
            for Duration in range(4, int(Ext_T_Except_0[Ext_index+1]-Ext_T_Except_0[Ext_index])): #At least 5 days. Will do Duration+1
                                                                           #Fall also not near 5 by distance.
                Return[str(Month) + "-" + str(Ext_Place)+"-"+ str(Duration)] = SR.Recent_Data(Ticker, Ext_index+1, Duration+1) #Duration + 1 because the duration is not equal to index of Recent_Data range now. include both first extrima, second.
    np.savez("NPZ/Ticker/Month-Ext/"+Ticker+".npz", **Return)
"""

#Ext Filter
"""
Ticker_List = ['MMM', 'ABT', 'ADBE', 'AES', 'AFL', 'A', 'APD', 'ALL', 'MO', 'AEE', 'AEP', 'AXP', 'AIG', 'AMGN', 'ADI', 'AON', 'APA', 'AAPL', 'AMAT', 'ADM', 'T', 'ADSK', 'ADP', 'AZO', 'AVY', 'BALL', 'BAC', 'BAX', 'BDX', 'BBY', 'BIIB', 'BNY', 'BA', 'BSX', 'BMY', 'BF-B', 'CPB', 'COF', 'CAH', 'CCL', 'CAT', 'COR', 'CNP', 'SCHW', 'CVX', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CLX', 'CMS', 'KO', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'GLW', 'COST', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DE', 'DVN', 'DOV', 'DTE', 'DUK', 'ETN', 'EBAY', 'ECL', 'EIX', 'EA', 'ELV', 'EMR', 'ETR', 'EOG', 'EFX', 'EQR', 'EXC', 'XOM', 'FDX', 'FITB', 'FE', 'FISV', 'F', 'BEN', 'GE', 'GEN', 'GD', 'GIS', 'GPC', 'GILD', 'GL', 'GS', 'HAL', 'HIG', 'HAS', 'HSY', 'HD', 'HON', 'HPQ', 'HBAN', 'IBM', 'ITW', 'INTC', 'IFF', 'IP', 'INTU', 'JNJ', 'JPM', 'KEY', 'KMB', 'KLAC', 'KR', 'LH', 'LLY', 'LIN', 'LMT', 'L', 'LOW', 'MTB', 'MAR', 'MRSH', 'MAS', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MU', 'MSFT', 'TAP', 'MCO', 'MS', 'MSI', 'NTAP', 'NEM', 'NEE', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NUE', 'NVDA', 'OXY', 'OMC', 'ORCL', 'PCAR', 'PSKY', 'PH', 'PAYX', 'PEP', 'PFE', 'PNW', 'PNC', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PHM', 'QCOM', 'DGX', 'RTX', 'RF', 'RVTY', 'ROK', 'SPGI', 'SLB', 'SHW', 'SPG', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'SYK', 'SYY', 'TPR', 'TGT', 'TXN', 'TXT', 'TMO', 'TJX', 'TRV', 'TFC', 'USB', 'UNP', 'UPS', 'UNH', 'VLO', 'VZ', 'VTRS', 'VMC', 'GWW', 'WMT', 'DIS', 'WM', 'WAT', 'WFC', 'WY', 'WMB', 'XEL', 'YUM', 'ZBH']

def High_pass_Filter(ext_indicies):
    return ext_indicies[ext_indicies>126]

def Eliminate(values, ext_indicies, indicies_2):
    return [np.argmin(values[k])+np.where(ext_indicies == k[0])[0][0] for k in indicies_2] #d of indicies of ext_indicies, like 0 and 1

#compare by value[indices] which is bigger/smaller -- > output is indiceis
#distance = not including some initial number, the array with it to the ending(inclduing ending) is lower or equal to the distance. So we need 5.
#iuf we set distance as 5, we can't eliminat teh numbers that distance is 5.
def Distance_Filter(values, ext_indicies, distance):
    data_diff = np.diff(ext_indicies) #d of ext_indieices
    n = np.where(data_diff < distance)[0] #d of indicies of ext_indicies
    Sub_Seq = [ext_indicies[i:i+1].astype(int) for i in n] #d of ext_indieices
    Should_Be = Eliminate(values, ext_indicies, Sub_Seq)
    return np.delete(ext_indicies, Should_Be)

def Filter(values, ext_indicies, distance):
    ext_indicies_127 = High_pass_Filter(ext_indicies)
    ext_indicies_Deleted = Distance_Filter(values, ext_indicies_127, distance)
    return ext_indicies_Deleted

Return = {}

for t in Ticker_List:
    Ext_T = Ext[t]
    Price_T = Price[t+"/Price"]
    Return[t] = Filter(Price_T, Ext_T, 5).astype(int)

np.savez("NPZ/Ticker/Ext_Fixed.npz", **Return)
"""

#Historical data collection
"""

#Macro_Coef Organization by Last month

Data = np.load("Macro_Coef_2.npz")

Return = {}

Result_list = []

#"Inflation_Rate", "Real_Rate", "Unemployment_Rate", "Credit_GDP", "Money_Supply", "Real_GDP", "VIX", "Time(From 0)"
#0,1,2,3,4,5<-- 5 is the last month, so get these datas.
#arrary at month, n is combined array with each values with above order for each month.
for i in range(5,252):
    for j in range(6):
        Result = np.array([Data[k][i-5+j] for k in ["Inflation_Rate", "Real_Rate", "Unemployment_Rate", "Credit_GDP", "Money_Supply", "Real_GDP", "VIX"]])
        Result = np.append(Result, i-5+j)
        Result_list.append(Result)
    Result_Arr = np.concatenate(Result_list)
    Result_list = []
    Return[str(i)] = Result_Arr

np.savez("Mo_Last.npz", **Return)
"""

#Extrima
"""
Data = np.load("Ticker/Price.npz")

Return = {}

for x in Ticker_List:
    Return[x]=np.union1d(B.Get_Rel(Data[x+"/Price"],42), B.Get_Rel(Data[x+"/Price"],48))


np.savez("Ticker/Ext.npz", **Return)

"""
#Price
"""
Return = {}

Data = yf.download(Ticker, start="2005-01-01", end="2026-01-01", interval="1d")
Data = (Data["Close"]+Data["Open"]+Data["High"]+Data["Low"])/4

for x in Ticker_List:
    Price = np.array(Data[x])
    Return[x+"/Price"] = Price

np.savez("Ticker/Price.npz", **Return)
"""

#Volume
"""

Ticker = "MMM ABT ADBE AES AFL A APD ALL MO AEE AEP AXP AIG AMGN ADI AON APA AAPL AMAT ADM T ADSK ADP AZO AVY BALL BAC BAX BDX BBY BIIB BNY BA BSX BMY BF-B CPB COF CAH CCL CAT COR CNP SCHW CVX CI CINF CTAS CSCO C CLX CMS KO CL CMCSA CAG COP ED GLW COST CSX CMI CVS DHR DRI DE DVN DOV DTE DUK ETN EBAY ECL EIX EA ELV EMR ETR EOG EFX EQR EXC XOM FDX FITB FE FISV F BEN GE GEN GD GIS GPC GILD GL GS HAL HIG HAS HSY HD HON HPQ HBAN IBM ITW INTC IFF IP INTU JNJ JPM KEY KMB KLAC KR LH LLY LIN LMT L LOW MTB MAR MRSH MAS MKC MCD MCK MDT MRK MET MU MSFT TAP MCO MS MSI NTAP NEM NEE NKE NI NSC NTRS NOC NUE NVDA OXY OMC ORCL PCAR PSKY PH PAYX PEP PFE PNW PNC PPG PPL PFG PG PGR PLD PRU PEG PHM QCOM DGX RTX RF RVTY ROK SPGI SLB SHW SPG SNA SO LUV SWK SBUX STT SYK SYY TPR TGT TXN TXT TMO TJX TRV TFC USB UNP UPS UNH VLO VZ VTRS VMC GWW WMT DIS WM WAT WFC WY WMB XEL YUM ZBH"

Ticker_List = ['MMM', 'ABT', 'ADBE', 'AES', 'AFL', 'A', 'APD', 'ALL', 'MO', 'AEE', 'AEP', 'AXP', 'AIG', 'AMGN', 'ADI', 'AON', 'APA', 'AAPL', 'AMAT', 'ADM', 'T', 'ADSK', 'ADP', 'AZO', 'AVY', 'BALL', 'BAC', 'BAX', 'BDX', 'BBY', 'BIIB', 'BNY', 'BA', 'BSX', 'BMY', 'BF-B', 'CPB', 'COF', 'CAH', 'CCL', 'CAT', 'COR', 'CNP', 'SCHW', 'CVX', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CLX', 'CMS', 'KO', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'GLW', 'COST', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DE', 'DVN', 'DOV', 'DTE', 'DUK', 'ETN', 'EBAY', 'ECL', 'EIX', 'EA', 'ELV', 'EMR', 'ETR', 'EOG', 'EFX', 'EQR', 'EXC', 'XOM', 'FDX', 'FITB', 'FE', 'FISV', 'F', 'BEN', 'GE', 'GEN', 'GD', 'GIS', 'GPC', 'GILD', 'GL', 'GS', 'HAL', 'HIG', 'HAS', 'HSY', 'HD', 'HON', 'HPQ', 'HBAN', 'IBM', 'ITW', 'INTC', 'IFF', 'IP', 'INTU', 'JNJ', 'JPM', 'KEY', 'KMB', 'KLAC', 'KR', 'LH', 'LLY', 'LIN', 'LMT', 'L', 'LOW', 'MTB', 'MAR', 'MRSH', 'MAS', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MU', 'MSFT', 'TAP', 'MCO', 'MS', 'MSI', 'NTAP', 'NEM', 'NEE', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NUE', 'NVDA', 'OXY', 'OMC', 'ORCL', 'PCAR', 'PSKY', 'PH', 'PAYX', 'PEP', 'PFE', 'PNW', 'PNC', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PHM', 'QCOM', 'DGX', 'RTX', 'RF', 'RVTY', 'ROK', 'SPGI', 'SLB', 'SHW', 'SPG', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'SYK', 'SYY', 'TPR', 'TGT', 'TXN', 'TXT', 'TMO', 'TJX', 'TRV', 'TFC', 'USB', 'UNP', 'UPS', 'UNH', 'VLO', 'VZ', 'VTRS', 'VMC', 'GWW', 'WMT', 'DIS', 'WM', 'WAT', 'WFC', 'WY', 'WMB', 'XEL', 'YUM', 'ZBH']


Return = {}

Data = yf.download(Ticker, start="2005-01-01", end="2026-01-01", interval="1d")
Data = Data["Volume"]

for x in Ticker_List:
    Volume = np.array(Data[x])
    Return[x+"/Volume"] = Volume

np.savez("Ticker/Volume.npz", **Return)
"""
