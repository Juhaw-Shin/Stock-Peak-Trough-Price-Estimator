import yfinance as yf
import numpy as np
import scipy.signal as ss
import Modification as B

SP500 = yf.Ticker("^GSPC")
SP500 = SP500.history(start="2005-01-01", end="2026-01-01", interval="1d")
SP500_Values_Total = np.array((SP500["Close"]+SP500["Open"]+SP500["High"]+SP500["Low"])/4)




def rsi(prices, t, window=14):
    prices = np.asarray(prices, dtype=float)

    # differences up to t
    delta = np.diff(prices[t-13:t+1])

    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)

    # seed averages (first window ending at t=window)
    avg_gain = np.mean(gain[:window])
    avg_loss = np.mean(loss[:window])

    # Wilder smoothing forward to t
    for i in range(window, len(gain)):
        avg_gain = (avg_gain * (window - 1) + gain[i]) / window
        avg_loss = (avg_loss * (window - 1) + loss[i]) / window

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

#Ext[] = index of trough in ticker
#Duration = 1 to whatever.
#Data including Trough, to some random point(Duration after the Trough).
#Ticker_Value's size = duration as a input.
#Price[] Sets of prices by Ticker(as a array)
#Volume[] sets pf volumes by Ticker
#start with Extrima: [Ext, ]
#index=Duration
"""
Volume = np.load("NPZ/Ticker/Volume/Volume.npz")
Ext = np.load("NPZ/Ticker/Ext.npz")
Return = np.load("NPZ/Ticker/Price/Return_2.npz")
Volume_Rate = np.load("NPZ/Ticker/Volume/Volume_Rate_2.npz")
Return_S = np.load("NPZ/Ticker/Price/Return_Smoothed_2.npz")
Volume_Rate_S = np.load("NPZ/Ticker/Volume/Volume_Rate_Smoothed_2.npz")
Ext_Org = np.load("NPZ/Ticker/Ext_Org.npz")
Mo_Last = np.load("NPZ/Mo_Last.npz")
Price = np.load("NPZ/Ticker/Price/Price.npz")
Weight = np.load(("NPZ/Weight_0.npz"))
Bias = np.load(("NPZ/Bias_0.npz"))
"""

Standard_Data = np.load("NPZ/Mean_Std.npz")


def Standard(x):
    mean = Standard_Data["mean"]
    std = Standard_Data["std"]
    return (x - mean) / std

def load_npz_full(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


Volume = load_npz_full("NPZ/Ticker/Volume/Volume.npz")
Ext = load_npz_full("NPZ/Ticker/Ext.npz")
Return = load_npz_full("NPZ/Ticker/Price/Return_2.npz")
Volume_Rate = load_npz_full("NPZ/Ticker/Volume/Volume_Rate_2.npz")
Return_S = load_npz_full("NPZ/Ticker/Price/Return_Smoothed_2.npz")
Volume_Rate_S = load_npz_full("NPZ/Ticker/Volume/Volume_Rate_Smoothed_2.npz")
Ext_Org = load_npz_full("NPZ/Ticker/Ext_Org.npz")
Mo_Last = load_npz_full("NPZ/Mo_Last.npz")
Price = load_npz_full("NPZ/Ticker/Price/Price.npz")
Ticker_List = ['MMM', 'ABT', 'ADBE', 'AES', 'AFL', 'A', 'APD', 'ALL', 'MO', 'AEE', 'AEP', 'AXP', 'AIG', 'AMGN', 'ADI', 'AON', 'APA', 'AAPL', 'AMAT', 'ADM', 'T', 'ADSK', 'ADP', 'AZO', 'AVY', 'BALL', 'BAC', 'BAX', 'BDX', 'BBY', 'BIIB', 'BNY', 'BA', 'BSX', 'BMY', 'BF-B', 'CPB', 'COF', 'CAH', 'CCL', 'CAT', 'COR', 'CNP', 'SCHW', 'CVX', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CLX', 'CMS', 'KO', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'GLW', 'COST', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DE', 'DVN', 'DOV', 'DTE', 'DUK', 'ETN', 'EBAY', 'ECL', 'EIX', 'EA', 'ELV', 'EMR', 'ETR', 'EOG', 'EFX', 'EQR', 'EXC', 'XOM', 'FDX', 'FITB', 'FE', 'F', 'BEN', 'GE', 'GEN', 'GD', 'GIS', 'GPC', 'GILD', 'GL', 'GS', 'HAL', 'HIG', 'HAS', 'HSY', 'HD', 'HON', 'HPQ', 'HBAN', 'IBM', 'ITW', 'INTC', 'IFF', 'IP', 'INTU', 'JNJ', 'JPM', 'KEY', 'KMB', 'KLAC', 'KR', 'LH', 'LLY', 'LIN', 'LMT', 'L', 'LOW', 'MTB', 'MAR', 'MRSH', 'MAS', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MU', 'MSFT', 'TAP', 'MCO', 'MS', 'MSI', 'NTAP', 'NEM', 'NEE', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NUE', 'NVDA', 'OXY', 'OMC', 'ORCL', 'PCAR', 'PH', 'PAYX', 'PEP', 'PFE', 'PNW', 'PNC', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PHM', 'QCOM', 'DGX', 'RTX', 'RF', 'RVTY', 'ROK', 'SPGI', 'SLB', 'SHW', 'SPG', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'SYK', 'SYY', 'TPR', 'TGT', 'TXN', 'TXT', 'TMO', 'TJX', 'TRV', 'TFC', 'USB', 'UNP', 'UPS', 'UNH', 'VLO', 'VZ', 'VTRS', 'VMC', 'GWW', 'WMT', 'DIS', 'WM', 'WAT', 'WFC', 'WY', 'WMB', 'XEL', 'YUM', 'ZBH']
rng = np.random.default_rng()


Ext = {
    k: np.append(v, 5282)
    for k, v in Ext.items()
}
def Random_Smooth(batch_size):
    Data = np.empty((batch_size,191), dtype=np.float64)
    Data_Return = np.empty(batch_size, dtype=np.float64)
    #Recent Random
    Ticker = rng.choice(Ticker_List, batch_size)
    for i, Ticker_Temp in enumerate(Ticker):
        Ext_Temp = Ext[Ticker_Temp]
        Ext_Size = np.size(Ext_Temp)-1
        Ext_Place = rng.integers(1,Ext_Size-1) # did 1 to avoid having 0. If have 0, then [Extrima-1] just crashes.
                                                #Did Ext_Size-1 because we need next extrima data.
        Dur_Temp = rng.integers(4,np.diff(Ext_Temp)[Ext_Place]+1)#4 here is 5 actually, in indicies the recent data contains. So in real data, four day after extrima which includes extrima: [Ext, ...
        Data_Set = Recent_Data(Ticker_Temp, Ext_Place,Dur_Temp+1)

    #Price/Volume Random
        Currunt_divided = (Ext_Temp[Ext_Place]+Dur_Temp) // 2 #Because 2 day averaged.
        Return_S_Temp = Return_S[Ticker_Temp]
        Volume_Rate_S_Temp = Volume_Rate_S[Ticker_Temp]
        Return_6_Temp = Return_S_Temp[Currunt_divided-64:Currunt_divided-1] # We don't know the value of today yet. can be a problem if we are in second half of the 2 day set.
                                                                            #+ Because it is rate/return, it also considiered. So if day  today is 7, we can use data up to 6 of price/volume,(By upper reason) so for rate/return, we can only use 5.
        Volume_Rate_6_Temp = Volume_Rate_S_Temp[Currunt_divided-64:Currunt_divided-1]
    #History
        Extrima_Exact = Ext_Temp[Ext_Place]
        Month = B.Matching_Data(Extrima_Exact)
        Historical_Data = Mo_Last[str(Month-1)]
    #Next extrima
        This_Extrima_Price = Price[Ticker_Temp+"/Price"][Ext_Temp[Ext_Place]]
        Next_Extrima_Price = Price[Ticker_Temp+"/Price"][Ext_Temp[Ext_Place+1]]
        Next_Return = 100*(Next_Extrima_Price - This_Extrima_Price)/This_Extrima_Price
      #Combine
        Combined_Data = np.concatenate((Data_Set,Return_6_Temp,Volume_Rate_6_Temp,Historical_Data))

        Data[i] = Combined_Data
        Data_Return[i] = Next_Return
    return [Standard(Data), Data_Return[:, np.newaxis]]

def Data_Collection(Ticker):
    Data = []
    Target = np.array([])
    Price_T = Price[Ticker+"/Price"]
    Ext_T = Ext[Ticker].astype(int)
    Volume_T = Volume[Ticker+"/Volume"]
    Ext_Org_T = Ext_Org[Ticker]
    Ext_Temp = Ext[Ticker]
    Ext_Size = np.size(Ext_Temp)-1
    for Extrima in range(1,Ext_Size-1): # did 1 to avoid having 0. If have 0, then [Extrima-1] just crashes.
                                            #Did Ext_Size-1 because we need next extrima data.
        for Dur_Temp in range(4,np.diff(Ext_Temp)[Extrima]+1): #4 here is 5 actually, in indicies the recent data contains. So in real data, four day after extrima which includes extrima: [Ext, ...
            Duration = Dur_Temp
            End = Ext_T[Extrima] + Duration

            # In Index

            Ticker_Values = Price_T[Ext_T[Extrima]:End+1]
            Ticker_Values_Fall = Price_T[Ext_T[Extrima-1]:Ext_T[Extrima]+1]
            Volume_Values = Volume_T[Ext_T[Extrima]:End+1]
            Volume_Values_Fall = Volume_T[Ext_T[Extrima-1]:Ext_T[Extrima]+1]
            Volume_Values_Normalized = Volume_Values/np.mean(Volume_T[End-125:End+1])
            Volume_Values_Fall_Normalized = Volume_Values_Fall/np.mean(Volume_T[End-125:End+1])
            Ticker_Values_D = np.diff(Ticker_Values)/Price_T[Ext_T[Extrima]]
            Ticker_Values_Fall_D = np.diff(Ticker_Values_Fall)/Price_T[Ext_T[Extrima-1]]


            SP500_Returns = np.diff(SP500_Values_Total[End-125:End+1]) / SP500_Values_Total[End-125:End+1][:-1]
            Ticker_Returns = np.diff(Price_T[End-125:End+1]) / Price_T[End-125:End+1][:-1]

            #Beta
            Var = np.var(SP500_Returns, ddof=1)
            Cov = np.cov(Ticker_Returns, SP500_Returns)[0, 1]
            Beta=Cov/Var

            Duration_Fall = Ext_T[Extrima] - Ext_T[Extrima-1]

            #Specific Returns
            #Requirment: Days more or equal to 5.
            Return_1, Return_2, Return_3, Return_4 = np.array_split(Ticker_Values_D * 100, 4)
            Return_1, Return_2, Return_3, Return_4 = [k.mean()/Duration for k in [Return_1, Return_2, Return_3, Return_4]]
            Return_Fall_1, Return_Fall_2, Return_Fall_3, Return_Fall_4  = np.array_split(Ticker_Values_Fall_D * 100, 4)
            Return_Fall_1, Return_Fall_2, Return_Fall_3, Return_Fall_4 = [k.mean()/Duration_Fall for k in [Return_Fall_1, Return_Fall_2, Return_Fall_3, Return_Fall_4]]

            #Rel_Strength
            RSI = rsi(Price_T, End)

            #Volume
            Smooth = ss.savgol_filter(Volume_Values_Normalized, window_length=2*round((np.size(Volume_Values_Normalized)-3)/8)+3, polyorder=min(round(np.size(Volume_Values_Normalized)/3)-2,2))
            Volume_Peakiness = np.max(np.abs(np.diff(Smooth, 2)))
            Volume_Peak = np.max(Volume_Values_Normalized)
            Smooth = ss.savgol_filter(Volume_Values_Fall_Normalized, window_length=2*round(np.size((Volume_Values_Fall_Normalized)-3)/8)+3, polyorder=min(round(np.size(Volume_Values_Fall_Normalized)/3)-2,2))
            Volume_Fall_Peakiness = np.max(np.abs(np.diff(Smooth, 2)))
            Volume_Fall_Peak = np.max(Volume_Values_Fall_Normalized)

            #average peak duration
            Peak_Dur = 5283/np.size(Ext_Org_T)

            Data_Set =  np.array([Beta, Return_1, Return_2, Return_3, Return_4, Return_Fall_1, Return_Fall_2, Return_Fall_3, Return_Fall_4, RSI, Volume_Fall_Peak, Volume_Fall_Peakiness, Volume_Peakiness, Volume_Peak, int(Duration), int(Duration_Fall), Peak_Dur])
        #Price/Volume Random
            Currunt_divided = (Ext_Temp[Extrima]+Dur_Temp) // 2 #Because 2 day averaged.
            Return_S_Temp = Return_S[Ticker]
            Volume_Rate_S_Temp = Volume_Rate_S[Ticker]
            Return_6_Temp = Return_S_Temp[Currunt_divided-64:Currunt_divided-1] # We don't know the value of today yet. can be a problem if we are in second half of the 2 day set.
                                                                                #+ Because it is rate/return, it also considiered. So if day  today is 7, we can use data up to 6 of price/volume,(By upper reason) so for rate/return, we can only use 5.
            Volume_Rate_6_Temp = Volume_Rate_S_Temp[Currunt_divided-64:Currunt_divided-1]
        #History
            Extrima_Exact = Ext_Temp[Extrima]
            Month = B.Matching_Data(Extrima_Exact)
            Historical_Data = Mo_Last[str(Month-1)]
        #Next extrima
            This_Extrima_Price = Price[Ticker+"/Price"][Ext_Temp[Extrima]]
            Next_Extrima_Price = Price[Ticker+"/Price"][Ext_Temp[Extrima+1]]
            Next_Return = 100*(Next_Extrima_Price - This_Extrima_Price)/This_Extrima_Price
        #Combine
            Combined_Data = np.concatenate((Data_Set,Return_6_Temp,Volume_Rate_6_Temp,Historical_Data))

            Data.append(Standard(Combined_Data))
            Target = np.append(Target,Next_Return)
    np.save("NPZ/Training Data/Training_Source.npy", np.array(Data))
    np.save("NPZ/Training Data/Target.npy", Target)

def Random_Rough(batch_size):
    Data = np.empty((batch_size,191), dtype=np.float64)
    Data_Return = np.empty(batch_size, dtype=np.float64)
    #Recent Random
    Ticker = rng.choice(Ticker_List, batch_size)
    for i, Ticker_Temp in enumerate(Ticker):
        Ext_Temp = Ext[Ticker_Temp]
        Ext_Size = np.size(Ext_Temp)-1
        Ext_Place = rng.integers(1,Ext_Size-1) # did 1 to avoid having 0. If have 0, then [Extrima-1] just crashes.
                                                #Did Ext_Size-1 because we need next extrima data.
        Dur_Temp = rng.integers(4,np.diff(Ext_Temp)[Ext_Place]+1)#4 here is 5 actually, in indicies the recent data contains. So in real data, four day after extrima which includes extrima: [Ext, ...
        Data_Set = Recent_Data(Ticker_Temp, Ext_Place,Dur_Temp+1)
    #Price/Volume Random
        Currunt_divided = (Ext_Temp[Ext_Place]+Dur_Temp) // 2 #Because 2 day averaged.
        Return_Temp = Return[Ticker_Temp]
        Volume_Rate_Temp = Volume_Rate[Ticker_Temp]
        Return_6_Temp = Return_Temp[Currunt_divided-64:Currunt_divided-1] # We don't know the value of today yet. can be a problem if we are in second half of the 2 day set.
                                                                            #+ Because it is rate/return, it also considiered. So if day  today is 7, we can use data up to 6 of price/volume,(By upper reason) so for rate/return, we can only use 5.
        Volume_Rate_6_Temp = Volume_Rate_Temp[Currunt_divided-64:Currunt_divided-1]
    #History
        Extrima_Exact = Ext_Temp[Ext_Place]+Dur_Temp
        Month = B.Matching_Data(Extrima_Exact)
        Historical_Data = Mo_Last[str(Month-1)]
    #Next extrima
        This_Extrima_Price = Price[Ticker_Temp+"/Price"][Ext_Temp[Ext_Place]]
        Next_Extrima_Price = Price[Ticker_Temp+"/Price"][Ext_Temp[Ext_Place+1]]
        Next_Return = 100*(Next_Extrima_Price-This_Extrima_Price)/This_Extrima_Price
    #Combine
        Combined_Data = np.concatenate((Data_Set,Return_6_Temp,Volume_Rate_6_Temp,Historical_Data))

        Data[i] = Combined_Data
        Data_Return[i] = Next_Return
    return [Standard(Data), Data_Return[:, np.newaxis]]

def Recent_Data(Ticker, Extrima, Duration):
    Price_T = Price[Ticker+"/Price"]
    Ext_T = Ext[Ticker].astype(int)
    Volume_T = Volume[Ticker+"/Volume"]
    Ext_Org_T = Ext_Org[Ticker]
    Duration += -1
    End = Ext_T[Extrima] + Duration

    # In Index

    Ticker_Values = Price_T[Ext_T[Extrima]:End+1]
    Ticker_Values_Fall = Price_T[Ext_T[Extrima-1]:Ext_T[Extrima]+1]
    Volume_Values = Volume_T[Ext_T[Extrima]:End+1]
    Volume_Values_Fall = Volume_T[Ext_T[Extrima-1]:Ext_T[Extrima]+1]
    Volume_Values_Normalized = Volume_Values/np.mean(Volume_T[End-125:End+1])
    Volume_Values_Fall_Normalized = Volume_Values_Fall/np.mean(Volume_T[End-125:End+1])
    Ticker_Values_D = np.diff(Ticker_Values)/Price_T[Ext_T[Extrima]]
    Ticker_Values_Fall_D = np.diff(Ticker_Values_Fall)/Price_T[Ext_T[Extrima-1]]


    SP500_Returns = np.diff(SP500_Values_Total[End-125:End+1]) / SP500_Values_Total[End-125:End+1][:-1]
    Ticker_Returns = np.diff(Price_T[End-125:End+1]) / Price_T[End-125:End+1][:-1]

    #Beta
    Var = np.var(SP500_Returns, ddof=1)
    Cov = np.cov(Ticker_Returns, SP500_Returns)[0, 1]
    Beta=Cov/Var

    Duration_Fall = Ext_T[Extrima] - Ext_T[Extrima-1]

    #Specific Returns
    #Requirment: Days more or equal to 5.
    Return_1, Return_2, Return_3, Return_4 = np.array_split(Ticker_Values_D * 100, 4)
    Return_1, Return_2, Return_3, Return_4 = [k.mean()/Duration for k in [Return_1, Return_2, Return_3, Return_4]]
    Return_Fall_1, Return_Fall_2, Return_Fall_3, Return_Fall_4  = np.array_split(Ticker_Values_Fall_D * 100, 4)
    Return_Fall_1, Return_Fall_2, Return_Fall_3, Return_Fall_4 = [k.mean()/Duration_Fall for k in [Return_Fall_1, Return_Fall_2, Return_Fall_3, Return_Fall_4]]

    #Rel_Strength
    RSI = rsi(Price_T, End)

    #Volume
    Smooth = ss.savgol_filter(Volume_Values_Normalized, window_length=2*round((np.size(Volume_Values_Normalized)-3)/8)+3, polyorder=min(round(np.size(Volume_Values_Normalized)/3)-2,2))
    Volume_Peakiness = np.max(np.abs(np.diff(Smooth, 2)))
    Volume_Peak = np.max(Volume_Values_Normalized)
    Smooth = ss.savgol_filter(Volume_Values_Fall_Normalized, window_length=2*round(np.size((Volume_Values_Fall_Normalized)-3)/8)+3, polyorder=min(round(np.size(Volume_Values_Fall_Normalized)/3)-2,2))
    Volume_Fall_Peakiness = np.max(np.abs(np.diff(Smooth, 2)))
    Volume_Fall_Peak = np.max(Volume_Values_Fall_Normalized)

    #average peak duration
    Peak_Dur = 5283/np.size(Ext_Org_T)

    return np.array([Beta, Return_1, Return_2, Return_3, Return_4, Return_Fall_1, Return_Fall_2, Return_Fall_3, Return_Fall_4, RSI, Volume_Fall_Peak, Volume_Fall_Peakiness, Volume_Peakiness, Volume_Peak, int(Duration), int(Duration_Fall), Peak_Dur])
    #17 factors.
