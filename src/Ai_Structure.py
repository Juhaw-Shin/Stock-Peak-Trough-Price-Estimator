import Ai_Back as Ba
import numpy as np
import Stock_Related as SR
import time
#import matplotliBa.pyplot as plt
#import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

"""
------------
Func that sepearte to batches.

Ticker-->ext number(Month)-->duration.

Lets just choose it random, not caring about what is what.
---------------
With month from above, have some function call a macro indicator, then concatenate.
-----------------
have price and volume that is averaged out for 2 day, also is smoothed/or not-->concateate.
----------------
What tommorow I will do!

for i in batch_set:
    eliminate some weight/Z by percentage.
    forward: i...
    repeat.
    """

Weight = np.load(("NPZ/Weight_6.npz"))
Bias = np.load(("NPZ/Bias_6.npz"))
Random_Valid = [0,0]
Random_Valid[0] = np.load("NPZ/Training Data/Valid_Sample.npy")
Random_Valid[1] = np.load("NPZ/Training Data/Valid_Result.npy")
Learning_Rate_Initial = np.float64(np.load("NPZ/Training Data/Learning Rate.npy"))

#Structure
#191--> 128-->64-->32-->32-->1

#Ba.Part_Back_Batch(From_To, Part_X_Currunt_Set, Part_X_Next_Set, Part_Weight_Next, Part_PDLX_Next_Set)
#Ba.Part_Next_Z_Batch(From_To, Z_Set, Weight, Bias)
#Ba.Back_Initial_Batch(From_To,Result_Set, X_Last_Set, Expected_Set, Weight_Last)


#Weight_Layer_n

def forward(Ac1_Set, Weight_Temp_1, Weight_Temp_2, Weight_Temp_3, Weight_Temp_4, Weight_Temp_5, Bias_Temp_1, Bias_Temp_2, Bias_Temp_3, Bias_Temp_4, Bias_Temp_5):
    Z = [0,0,0,0,0,0,0]
    Z[2] = Ba.Start_Z_Batch([191,128], Ac1_Set, Weight_Temp_1, Bias_Temp_1)
    Z[3] = Ba.Next_Z_Batch([128,64], Z[2], Weight_Temp_2, Bias_Temp_2)
    Z[4] = Ba.Next_Z_Batch([64,32], Z[3], Weight_Temp_3, Bias_Temp_3)
    Z[5] = Ba.Next_Z_Batch([32,32], Z[4], Weight_Temp_4, Bias_Temp_4)
    Z[6] = Ba.Next_Z_Batch([32,1], Z[5], Weight_Temp_5, Bias_Temp_5)
    return Z[6]



Weight_1 = Weight["Weight_Layer_1"]
Bias_1   = Bias["Bias_Layer_1"]
Weight_2 = Weight["Weight_Layer_2"]
Bias_2   = Bias["Bias_Layer_2"]
Weight_3 = Weight["Weight_Layer_3"]
Bias_3   = Bias["Bias_Layer_3"]
Weight_4 = Weight["Weight_Layer_4"]
Bias_4   = Bias["Bias_Layer_4"]
Weight_5 = Weight["Weight_Layer_5"]
Bias_5   = Bias["Bias_Layer_5"]
Cost_History = []

def clip_grad(grad, max_norm=1):
    arr = np.array(grad, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > max_norm:
        return True
    return False

def Machine_Learn(Learning_Rate, Repeat_Number, Batch_Size, max_norm = 1.0):
    dW = [0]*5
    dB = [0]*5
    PDLX = [0]*5
    W = [Weight_1, Weight_2, Weight_3, Weight_4, Weight_5]
    B = [Bias_1, Bias_2, Bias_3, Bias_4, Bias_5]
    Z = [0,0,0,0,0,0,0]
    for i in range(Repeat_Number):
        Random = SR.Random_Smooth(Batch_Size)

        Ac1 = Random[0]
        Z[2] = Ba.Start_Z_Batch([191,128], Ac1, W[0], B[0])
        Z[3] = Ba.Next_Z_Batch([128,64], Z[2], W[1], B[1])
        Z[4] = Ba.Next_Z_Batch([64,32], Z[3], W[2], B[2])
        Z[5] = Ba.Next_Z_Batch([32,32], Z[4], W[3], B[3])
        Z[6] = Ba.Next_Z_Batch([32,1], Z[5], W[4], B[4])

        PDLX[4],dW[4],dB[4] = Ba.Back_Initial_Batch(Batch_Size,[32,1], Random[1], Z[5], Z[6], W[4])

        PDLX[3],dW[3],dB[3] = Ba.Back_Batch(Batch_Size,[32,32], Z[4], Z[5], W[3], PDLX[4])


        PDLX[2],dW[2],dB[2] = Ba.Back_Batch(Batch_Size, [64,32], Z[3], Z[4], W[2], PDLX[3])


        PDLX[1],dW[1],dB[1] = Ba.Back_Batch(Batch_Size, [128,64], Z[2], Z[3], W[1], PDLX[2])


        PDLX[0],dW[0],dB[0] = Ba.First_Layer_Batch(Batch_Size, [191,128], Ac1, Z[2], W[0], PDLX[1])

        grads = [*dW, *dB]

        global_norm = np.sqrt(
            sum(np.sum(grad ** 2) for grad in grads)
        )

        scale = min(1.0, max_norm / (global_norm + 1e-12))

        dW = [grad * scale for grad in dW]
        dB = [grad * scale for grad in dB]

        W[4] -= Learning_Rate * dW[4]
        B[4] -= Learning_Rate * dB[4]
        W[3] -= Learning_Rate * dW[3]
        B[3] -= Learning_Rate * dB[3]
        W[2] -= Learning_Rate * dW[2]
        B[2] -= Learning_Rate * dB[2]
        W[1] -= Learning_Rate * dW[1]
        B[1] -= Learning_Rate * dB[1]
        W[0] -= Learning_Rate * dW[0]
        B[0] -= Learning_Rate * dB[0]
    print(str(Random[1][-5:].flatten()) + ", " + str(Z[6][-5:].flatten()))
    return [W[0], W[1], W[2], W[3], W[4], B[0], B[1], B[2], B[3], B[4]]

# def Grad_Dist_Check(Repeat_Number, Batch_Size):
#     W1, W2, W3, W4, W5 = Weight_1, Weight_2, Weight_3, Weight_4, Weight_5
#     B1, B2, B3, B4, B5 = Bias_1, Bias_2, Bias_3, Bias_4, Bias_5

#     Grad_Weight_Set = np.array([])
#     Grad_Bias_Set = np.array([])
#     Z = [0,0,0,0,0,0,0]
#     for i in range(Repeat_Number):
#         Random = SR.Random_Smooth(Batch_Size)

#         Ac1 = Random[0]
#         Z[2] = Ba.Start_Z_Batch([191,128], Ac1, W1, B1)
#         Z[3] = Ba.Next_Z_Batch([128,64], Z[2], W2, B2)
#         Z[4] = Ba.Next_Z_Batch([64,32], Z[3], W3, B3)
#         Z[5] = Ba.Next_Z_Batch([32,32], Z[4], W4, B4)
#         Z[6] = Ba.Next_Z_Batch([32,1], Z[5], W5, B5)

#         Back = Ba.Back_Initial_Batch(Batch_Size,[32,1], Random[1], Z[5], Z[6], W5)
#         Grad_Weight_Set = np.append(Grad_Weight_Set,Back[1])
#         Grad_Bias_Set = np.append(Grad_Bias_Set,Back[2])

#         Back = Ba.Back_Batch(Batch_Size,[32,32], Z[4], Z[5], W4, Back[0])
#         Grad_Weight_Set = np.append(Grad_Weight_Set,Back[1])
#         Grad_Bias_Set = np.append(Grad_Bias_Set,Back[2])

#         Back = Ba.Back_Batch(Batch_Size, [64,32], Z[3], Z[4], W3, Back[0])
#         Grad_Weight_Set = np.append(Grad_Weight_Set,Back[1])
#         Grad_Bias_Set = np.append(Grad_Bias_Set,Back[2])

#         Back = Ba.Back_Batch(Batch_Size, [128,64], Z[2], Z[3], W2, Back[0])
#         Grad_Weight_Set = np.append(Grad_Weight_Set,Back[1])
#         Grad_Bias_Set = np.append(Grad_Bias_Set,Back[2])

#         Back = Ba.First_Layer_Batch(Batch_Size, [191,128], Ac1, Z[2], W1, Back[0])
#         Grad_Weight_Set = np.append(Grad_Weight_Set,Back[1])
#         Grad_Bias_Set = np.append(Grad_Bias_Set,Back[2])
#     np.save("Test/Weight_Dist_"+str(Batch_Size)+".npy", Grad_Weight_Set)
#     np.save("Test/Bias_Dist_"+str(Batch_Size)+".npy", Grad_Bias_Set)


def Machine_Learn_local(Weight_Bias_Set, Learning_Rate, Repeat_Number, Batch_Size, Check_Number = 20, max_norm = 1.0):
    W = Weight_Bias_Set[:5]
    B = Weight_Bias_Set[-5:]
    dW = [0]*5
    dB = [0]*5
    PDLX = [0]*5
    Z = [0,0,0,0,0,0,0]
    Check_divide = (Repeat_Number)//Check_Number
    y = np.zeros(Check_Number)
    for j in range(Check_Number):
        for i in range(Check_divide):
            Random = SR.Random_Smooth(Batch_Size)

            Ac1 = Random[0]
            Z[2] = Ba.Start_Z_Batch([191,128], Ac1, W[0], B[0])
            Z[3] = Ba.Next_Z_Batch([128,64], Z[2], W[1], B[1])
            Z[4] = Ba.Next_Z_Batch([64,32], Z[3], W[2], B[2])
            Z[5] = Ba.Next_Z_Batch([32,32], Z[4], W[3], B[3])
            Z[6] = Ba.Next_Z_Batch([32,1], Z[5], W[4], B[4])

            PDLX[4],dW[4],dB[4] = Ba.Back_Initial_Batch(Batch_Size,[32,1], Random[1], Z[5], Z[6], W[4])

            PDLX[3],dW[3],dB[3] = Ba.Back_Batch(Batch_Size,[32,32], Z[4], Z[5], W[3], PDLX[4])


            PDLX[2],dW[2],dB[2] = Ba.Back_Batch(Batch_Size, [64,32], Z[3], Z[4], W[2], PDLX[3])


            PDLX[1],dW[1],dB[1] = Ba.Back_Batch(Batch_Size, [128,64], Z[2], Z[3], W[1], PDLX[2])


            PDLX[0],dW[0],dB[0] = Ba.First_Layer_Batch(Batch_Size, [191,128], Ac1, Z[2], W[0], PDLX[1])

            grads = [*dW, *dB]

            global_norm = np.sqrt(
                sum(np.sum(grad ** 2) for grad in grads)
            )

            scale = min(1.0, max_norm / global_norm)

            dW = [grad * scale for grad in dW]
            dB = [grad * scale for grad in dB]

            W[4] -= Learning_Rate * dW[4]
            B[4] -= Learning_Rate * dB[4]
            W[3] -= Learning_Rate * dW[3]
            B[3] -= Learning_Rate * dB[3]
            W[2] -= Learning_Rate * dW[2]
            B[2] -= Learning_Rate * dB[2]
            W[1] -= Learning_Rate * dW[1]
            B[1] -= Learning_Rate * dB[1]
            W[0] -= Learning_Rate * dW[0]
            B[0] -= Learning_Rate * dB[0]
        y[j] = np.mean((Random_Valid[1] - forward(Random_Valid[0], W[0], W[1], W[2], W[3], W[4], B[0], B[1], B[2], B[3], B[4]))** 2)
    return float(np.log(np.median(y[-(Check_Number//2):])/np.median(y[:(Check_Number//2)])))



def Each_LR_Search(Repeat_Number, Batch_Size, Learning_Rate): # check every iteration
    W1, W2, W3, W4, W5 = Weight_1, Weight_2, Weight_3, Weight_4, Weight_5
    B1, B2, B3, B4, B5 = Bias_1, Bias_2, Bias_3, Bias_4, Bias_5
    Weight_Bias_Set = [
        W1.copy(), W2.copy(), W3.copy(), W4.copy(), W5.copy(),
        B1.copy(), B2.copy(), B3.copy(), B4.copy(), B5.copy()
    ]
    return Machine_Learn_local(Weight_Bias_Set, Learning_Rate, Repeat_Number, Batch_Size)

def LR_Search(Learning_Rate_Set, Repeat_Number, Batch_Size):
    fixed_func = partial(Each_LR_Search, Repeat_Number, Batch_Size)
    with ProcessPoolExecutor(max_workers=3) as executor:
        result = list(executor.map(fixed_func, Learning_Rate_Set))
    Result_Np = np.array(result)/Learning_Rate_Set
    LR_Indice = np.argmax(-Result_Np)
    return Learning_Rate_Set[LR_Indice]

def LR_Search_Set(Learning_Rate_Set, Repeat_Number, Batch_Size):
    fixed_func = partial(Each_LR_Search, Repeat_Number, Batch_Size)
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(fixed_func, Learning_Rate_Set))










def Training(Learning_Rate_Initial, LR_Searching, Save, Repeat_Number, Batch_Size, Searching_Size = 256, Searching_Repeat = 300):
    global Weight_1, Weight_2, Weight_3, Weight_4, Weight_5
    global Bias_1, Bias_2, Bias_3, Bias_4, Bias_5
    LR_Rate = Learning_Rate_Initial
    LR_Choice = np.array([0.05, 0.11,0.25,0.4,0.6,1,1.4,2.6,3])
    if Save % LR_Searching == 0:
        Number_Search = Save//LR_Searching
    else:
        print("Searching Number Not Fit")
        return
    if Repeat_Number % Save == 0:
        Number_Save = Repeat_Number//Save
    else:
        print("Save Number Not Fit")
        return
    for j in range(LR_Searching):
        for i in range(Number_Search):
            Weight_1, Weight_2, Weight_3, Weight_4, Weight_5,Bias_1, Bias_2, Bias_3, Bias_4, Bias_5 = Machine_Learn(LR_Rate, Number_Save, Batch_Size)
            #Saving
            Return = {}
            Return["Weight_Layer_1"] = Weight_1
            Return["Weight_Layer_2"] = Weight_2
            Return["Weight_Layer_3"] = Weight_3
            Return["Weight_Layer_4"] = Weight_4
            Return["Weight_Layer_5"] = Weight_5
            np.savez("NPZ/Weight_7.npz", **Return)

            Return = {}
            Return["Bias_Layer_1"] = Bias_1
            Return["Bias_Layer_2"] = Bias_2
            Return["Bias_Layer_3"] = Bias_3
            Return["Bias_Layer_4"] = Bias_4
            Return["Bias_Layer_5"] = Bias_5
            np.savez("NPZ/Bias_7.npz", **Return)
            print(np.mean((Random_Valid[1] - forward(Random_Valid[0], Weight_1, Weight_2, Weight_3, Weight_4, Weight_5, Bias_1, Bias_2, Bias_3, Bias_4, Bias_5))** 2))
            print("Weights Saved")
        # Learning_Rate_Set = LR_Rate * LR_Choice
        # LR_Rate = LR_Search(Learning_Rate_Set, Searching_Repeat, Searching_Size)
        # if LR_Rate > 0.5:
        #     LR_Rate = 0.5
        # np.save("NPZ/Training Data/Learning Rate.npy", LR_Rate)
        # print("Learning Rate Saved")
#Learning_Rate_Set = [0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.0001,0.000125,0.00015,0.0002,0.00025,0.0003,0.0004,0.0005]
#Learning_Rate_Set_2 = [1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4,1e-3, 2.5e-3, 5e-3, 1e-4]
#All functions must be under this:
if __name__ == '__main__':
    Training(0.0001, 1, 2000, 2000000, 500)


# def clip_grad(grad, max_norm=1000.0):
#     arr = np.array(grad, dtype=float)
#     norm = np.linalg.norm(arr)
#     if norm > max_norm:
#         arr = arr * (max_norm / norm)
#     return arr


# def Machine_Learn(Repeat_Number, Batch_Size):
#     global Bias_1, Bias_2, Bias_3, Bias_4, Bias_5
#     global Weight_1, Weight_2, Weight_3, Weight_4, Weight_5

#     W1, W2, W3, W4, W5 = Weight_1, Weight_2, Weight_3, Weight_4, Weight_5
#     B1, B2, B3, B4, B5 = Bias_1, Bias_2, Bias_3, Bias_4, Bias_5


#     Z = [0,0,0,0,0,0,0]
#     for i in range(Repeat_Number):
#         Random = M1.Random_Smooth(Batch_Size)

#         Ac1 = Random[0]
#         Z[2] = M.Start_Z_Batch([191,128], Ac1, W1, B1)
#         Z[3] = M.Next_Z_Batch([128,64], Z[2], W2, B2)
#         Z[4] = M.Next_Z_Batch([64,32], Z[3], W3, B3)
#         Z[5] = M.Next_Z_Batch([32,32], Z[4], W4, B4)
#         Z[6] = M.Next_Z_Batch([32,1], Z[5], W5, B5)

#         start = time.perf_counter()
#         Back = M.Back_Initial_Batch(Batch_Size,[32,1], Random[1], Z[5], Z[6], W5)
#         W5 -= Learning_Rate * clip_grad(Back[1])
#         B5 -= Learning_Rate * clip_grad(Back[2])

#         Back = M.Back_Batch(Batch_Size,[32,32], Z[4], Z[5], W4, Back[0])
#         W4 -= Learning_Rate * clip_grad(Back[1])
#         B4 -= Learning_Rate * clip_grad(Back[2])

#         Back = M.Back_Batch(Batch_Size, [64,32], Z[3], Z[4], W3, Back[0])
#         W3 -= Learning_Rate * clip_grad(Back[1])
#         B3 -= Learning_Rate * clip_grad(Back[2])

#         Back = M.Back_Batch(Batch_Size, [128,64], Z[2], Z[3], W2, Back[0])
#         W2 -= Learning_Rate * clip_grad(Back[1])
#         B2 -= Learning_Rate * clip_grad(Back[2])

#         Back = M.First_Layer_Batch(Batch_Size, [191,128], Ac1, Z[2], W1, Back[0])
#         W1 -= Learning_Rate * clip_grad(Back[1])
#         B1 -= Learning_Rate * clip_grad(Back[2])

#         end = time.perf_counter()
#         print(f"Elapsed time: {end - start:.6f} seconds")
#     Weight_1, Weight_2, Weight_3, Weight_4, Weight_5 = W1, W2, W3, W4, W5
#     Bias_1, Bias_2, Bias_3, Bias_4, Bias_5 = B1, B2, B3, B4, B5


# def Training(Saves, Repeat_Number, Batch_Size):
#     global Cost_History
#     Cost_History = []

#     Number = Repeat_Number / Saves

#     def update_plot():
#         if len(Cost_History) < 1:
#             return

#         y = np.array(Cost_History, dtype=float)
#         x = np.arange(len(y), dtype=float)

#         if len(Cost_History) > 1:
#             m, b = np.polyfit(x, y, 1)
#             y_fit = m * x + b
#         else:
#             y_fit = np.array([], dtype=float)

#         payload = {
#             "cost": y.tolist(),
#             "fit": y_fit.tolist(),
#         }

#         html = f"""<!doctype html>
# <html>
# <head>
#   <meta charset="utf-8">
#   <meta http-equiv="refresh" content="2">
#   <title>Training Cost</title>
#   <style>
#     body {{ margin: 0; font-family: Arial, sans-serif; background: #111; color: #eee; }}
#     .wrap {{ padding: 24px; }}
#     canvas {{ width: 100%; height: 70vh; background: #181818; border: 1px solid #333; }}
#     .meta {{ margin-top: 12px; color: #bbb; }}
#   </style>
# </head>
# <body>
#   <div class="wrap">
#     <h2>Training Cost</h2>
#     <canvas id="chart" width="1400" height="700"></canvas>
#     <div class="meta" id="meta"></div>
#   </div>
#   <script>
#     const data = {json.dumps(payload)};
#     const canvas = document.getElementById("chart");
#     const ctx = canvas.getContext("2d");
#     const w = canvas.width, h = canvas.height;
#     const pad = 60;
#     const costs = data.cost;
#     const fit = data.fit;

#     function sx(i) {{
#       return costs.length <= 1 ? pad : pad + i * (w - 2 * pad) / (costs.length - 1);
#     }}
#     function sy(v, ymin, ymax) {{
#       return h - pad - (v - ymin) * (h - 2 * pad) / (ymax - ymin || 1);
#     }}
#     function drawLine(values, color, width) {{
#       if (!values.length) return;
#       ctx.beginPath();
#       values.forEach((v, i) => {{
#         const x = sx(i), y = sy(v, ymin, ymax);
#         if (i === 0) ctx.moveTo(x, y);
#         else ctx.lineTo(x, y);
#       }});
#       ctx.strokeStyle = color;
#       ctx.lineWidth = width;
#       ctx.stroke();
#     }}

#     const all = costs.concat(fit);
#     const ymin = Math.min(...all);
#     const ymax = Math.max(...all);

#     ctx.clearRect(0, 0, w, h);
#     ctx.strokeStyle = "#555";
#     ctx.lineWidth = 1;
#     ctx.beginPath();
#     ctx.moveTo(pad, pad);
#     ctx.lineTo(pad, h - pad);
#     ctx.lineTo(w - pad, h - pad);
#     ctx.stroke();

#     ctx.fillStyle = "#bbb";
#     ctx.font = "18px Arial";
#     ctx.fillText("MSE", 18, pad - 20);
#     ctx.fillText("Checkpoint", w - 170, h - 18);
#     ctx.fillText(ymax.toFixed(4), 8, pad + 6);
#     ctx.fillText(ymin.toFixed(4), 8, h - pad + 6);

#     drawLine(costs, "#4da3ff", 3);
#     drawLine(fit, "#ffcc33", 2);

#     document.getElementById("meta").textContent =
#       `points: ${{costs.length}} | latest cost: ${{costs[costs.length - 1]?.toFixed(6)}}`;
#   </script>
# </body>
# </html>
# """

#         with open("training_cost.html", "w", encoding="utf-8") as f:
#             f.write(html)

#     for i in range(Saves):

#         # =========================
#         # TRAIN STEP
#         # =========================
#         Machine_Learn(int(Number), Batch_Size)

#         # =========================
#         # SAVE WEIGHTS + BIAS
#         # =========================
#         weight_save = {
#             "Weight_Layer_1": Weight_1,
#             "Weight_Layer_2": Weight_2,
#             "Weight_Layer_3": Weight_3,
#             "Weight_Layer_4": Weight_4,
#             "Weight_Layer_5": Weight_5
#         }

#         bias_save = {
#             "Bias_Layer_1": Bias_1,
#             "Bias_Layer_2": Bias_2,
#             "Bias_Layer_3": Bias_3,
#             "Bias_Layer_4": Bias_4,
#             "Bias_Layer_5": Bias_5
#         }

#         np.savez(f"NPZ/Weight_1.npz", **weight_save)
#         np.savez(f"NPZ/Bias_1.npz", **bias_save)

#         print(f"Saved checkpoint {i}")

#         Cost = np.mean((Random_Valid[1] - forward(Random_Valid[0], Weight_1, Weight_2, Weight_3, Weight_4, Weight_5, Bias_1, Bias_2, Bias_3, Bias_4, Bias_5))** 2)
#         Cost_History.append(Cost)
#         update_plot()

#     print("Cost graph updated in training_cost.html")
