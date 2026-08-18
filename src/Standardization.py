import numpy as np
import Stock_Related as SR
import Ai_Back as B

def load_npz_full(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}



def Standard(Data):
    mean = np.mean(Data, axis = 0)
    std = np.std(Data, axis = 0)
    std = np.where(std == 0, 1, std)
    return [mean,std]


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
        Data_Set = SR.Recent_Data(Ticker_Temp, Ext_Place,Dur_Temp+1)
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
    return Standard(Data)


def Find(Save, Number):
    mean = np.zeros(191, dtype=np.float64)
    M2 = np.zeros(191, dtype=np.float64)  # sum of squared deviations
    count = 0

    for n in range(Number):
        batch_mean, batch_std = Random_Smooth(Save)
        batch_count = Save

        if count == 0:
            mean = batch_mean
            M2 = (batch_std ** 2) * batch_count
            count = batch_count
        else:
            batch_M2 = (batch_std ** 2) * batch_count
            new_count = count + batch_count
            delta = batch_mean - mean

            mean += delta * batch_count / new_count
            M2 += batch_M2 + (delta ** 2) * count * batch_count / new_count
            count = new_count

        if count > 1:
            variance = M2 / count
            std = np.sqrt(variance)
        else:
            std = np.zeros_like(mean)

        # save progressive learning state
        np.savez(
            "NPZ/Mean_Std.npz",
            mean=mean,
            std=std,
        )

        print(f"Updated step {n+1}")

    return mean, std

Find(10000, 50)
