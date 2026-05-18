import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math
import csv
import sys


# 
# --- modeFRONTIER Response Surface ----------------
# Code Created by
# modeFRONTIER  - (c) ESTECO S.p.A.
# modeFRONTIER Version modeFRONTIER 2025R4 19.7.3 b20251021
# Date Tue Apr 21 12:09:04 JST 2026
# Project Name mf結果4.prj
# Operating System Windows 11 10.0 amd64
# Java (SDK/JRE) Version 21.0.7
# Java Vendor Eclipse Adoptium
# Java Vendor URL https://adoptium.net/
# User Name nakay
# 
# 
# --- DISCLAIMER - Please do not erase -------------
# NO WARRANTY ON RSM CODE
# The Response Surface Methodology ("RSM") is a code which, due to the nature of machine learning based predictive models,
# may provide inaccurate output or otherwise not always produce the intended results. Therefore it should not be relied on
# as the sole basis to implement a design, whose incorrect implementation could result in injury to person or property.
# This code is not intended for use in any inherently dangerous applications, including applications which may create a risk
# of personal injury. If you use this code without reserve, you take full responsibility to grant all appropriate fail-safe,
# backup, redundancy, and other measures to ensure its safe use.
# 
# ESTECO makes to the Customer no warranty, express or implied, with reference to the compliance of the RSM code with a particular use.
# 
# Furthermore, ESTECO:
# (i) makes no warranty, express or implied, on the merchantability and fitness of the RSM code for a particular purpose,
# (ii) does not warrant that the operation or other use of the RSM code be uninterrupted or error free or will cause damage or
# disruption to the Customer’s data, computers or networks.
# 
# 
# --------------------------------------------------
# x[0] corresponds to variable J
# x[1] corresponds to variable Je
# x[2] corresponds to variable R1
# x[3] corresponds to variable R2
# x[4] corresponds to variable R4
# x[5] corresponds to variable R5
# x[6] corresponds to variable as
# x[7] corresponds to variable al
# x[8] corresponds to variable ε1
# x[9] corresponds to variable ε2
# x[10] corresponds to variable ti
# x[11] corresponds to variable to
# --------------------------------------------------
# 
# 
# --------------------------------------------------
# Response Surface Name : t4_NN_0_pv
# Response Surface Type : Neural Networks
# --------------------------------------------------
# 

import math
import csv
import sys

class t4_NN_0_pv:
    def __init__(self):
        self.n_input = 10
        # load data from file
        try:
            # 自分のファイル(.py)があるフォルダのパスを取得し、CSVと結合する
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, 't4_NN_0_pv.csv')
            
            with open(csv_path) as csvfile:
                filereader = csv.reader(csvfile)
                next(filereader)
                next(filereader)
                self.x_range = [[0 for _ in range(2)] for _ in range(10)]
                for i in range(10):
                    self.x_range[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.y_range = [0 for _ in range(2)]
                for i in range(2):
                    self.y_range[i] = float(next(filereader)[0])
                next(filereader)
                self.out_range = [0 for _ in range(2)]
                for i in range(2):
                    self.out_range[i] = float(next(filereader)[0])
                next(filereader)
                self.w1 = [[0 for _ in range(10)] for _ in range(8)]
                for i in range(8):
                    self.w1[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.b1 = [0 for _ in range(8)]
                for i in range(8):
                    self.b1[i] = float(next(filereader)[0])
                next(filereader)
                self.w2 = [[0 for _ in range(8)] for _ in range(1)]
                for i in range(1):
                    self.w2[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.b2 = [0 for _ in range(1)]
                for i in range(1):
                    self.b2[i] = float(next(filereader)[0])
                next(filereader)
                csvfile.close()
        except OSError:
            print("ERROR: cannot open the data file")
            sys.exit(1)
        except StopIteration:
            pass

    def evaluate(self, x):
        # check input
        if len(x) != 12:
            print("ERROR - Wrong Input Vector Length")
            return math.nan
        # keep only important input variables
        xx = [x[0], x[1], x[3], x[4], x[6], x[7], x[8], x[9], x[10], x[11]]
        # warning: variable x[2] is ignored
        # warning: variable x[5] is ignored

        # normalize input
        xn = [0 for _ in range(self.n_input)]
        for i in range(self.n_input):
            xn[i] = (2 * xx[i] - self.x_range[i][0] - self.x_range[i][1]) / (self.x_range[i][1] - self.x_range[i][0])

        # perform computations
        n1 = [0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            n1[i] = self.b1[i]
            for j in range(len(self.w1[0])):
                n1[i] += self.w1[i][j] * xn[j]
        y1 = [0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            exp = math.exp(-2.0 * n1[i])
            if exp == math.inf:
                y1[i] = -1.0
            else:
                y1[i] = (1.0 - exp)/(1.0 + exp)
        n2 = [0 for _ in range(len(self.w2))]
        for i in range(len(self.w2)):
            n2[i] = self.b2[i]
            for j in range(len(self.w2[0])):
                n2[i] += self.w2[i][j] * y1[j]
        yn = [0 for _ in range(len(self.w2))]
        for i in range(len(self.w2)):
            yn[i] = n2[i]
        # scale output
        y = self.y_range[0] + (self.y_range[1] - self.y_range[0])/(self.out_range[1] - self.out_range[0]) * (yn[0] - self.out_range[0])
        return y


    def get_input_variable_names(self):
        return ["J", "Je", "R1", "R2", "R4", "R5", "as", "al", "ε1", "ε2", "ti", "to"]

    def get_output_variable_name(self):
        return "t4"





# ==========================================
# 2. Streamlit ダッシュボード UI
# ==========================================
def main():
    st.set_page_config(layout="wide", page_title="modeFRONTIER Dashboard")
    st.title("📊 modeFRONTIER RSM Dashboard (t4)")


    try:
        model = t4_NN_0_pv()
    except Exception as e:
        st.error(str(e))
        st.info("実行フォルダに 't1_NN_0.csv' が存在するか確認してください。")
        return


    st.sidebar.header("Input Parameters")
   
    # --- スライダー入力の設定 (10変数は model.x_range から範囲を取得) ---
    v_J   = st.sidebar.slider("J (Active)", float(model.x_range[0][0]), float(model.x_range[0][1]), float((model.x_range[0][0] + model.x_range[0][1])/2))
    v_Je  = st.sidebar.slider("Je (Active)", float(model.x_range[1][0]), float(model.x_range[1][1]), float((model.x_range[1][0] + model.x_range[1][1])/2))
   
    v_R1  = st.sidebar.number_input("R1 (Ignored)", value=0.04, format="%.4f")
    v_R2  = st.sidebar.slider("R2 (Active)", float(model.x_range[2][0]), float(model.x_range[2][1]), float((model.x_range[2][0] + model.x_range[2][1])/2))
    v_R4  = st.sidebar.slider("R4 (Active)", float(model.x_range[3][0]), float(model.x_range[3][1]), float((model.x_range[3][0] + model.x_range[3][1])/2))
    v_R5  = st.sidebar.number_input("R5 (Ignored)", value=0.11, format="%.4f")
   
    v_as  = st.sidebar.slider("as (Active)", float(model.x_range[4][0]), float(model.x_range[4][1]), float((model.x_range[4][0] + model.x_range[4][1])/2))
    v_al  = st.sidebar.slider("al (Active)", float(model.x_range[5][0]), float(model.x_range[5][1]), float((model.x_range[5][0] + model.x_range[5][1])/2))
    v_ε1 = st.sidebar.slider("ε1 (Active)", float(model.x_range[6][0]), float(model.x_range[6][1]), float((model.x_range[6][0] + model.x_range[6][1])/2))
    v_ε2 = st.sidebar.slider("ε2 (Active)", float(model.x_range[7][0]), float(model.x_range[7][1]), float((model.x_range[7][0] + model.x_range[7][1])/2))
    v_ti  = st.sidebar.slider("ti (Active)", float(model.x_range[8][0]), float(model.x_range[8][1]), float((model.x_range[8][0] + model.x_range[8][1])/2))
    v_to  = st.sidebar.slider("to (Active)", float(model.x_range[9][0]), float(model.x_range[9][1]), float((model.x_range[9][0] + model.x_range[9][1])/2))


    # --- 数式による派生変数の計算 ---
    st.sidebar.markdown("---")
    st.sidebar.header("Computed Parameters")
   
    # 1. SATの計算
    v_SAT = v_to + (1.0/23.0) * (v_as * v_J - v_al * v_Je)
    st.sidebar.metric("SAT (Calculated)", f"{v_SAT:.4f}")


    # 2. R3の計算 (ゼロ除算を防ぐため max(x, 1e-10) を使用)
    try:
        rad_term = (1.0 / (max(v_ε1, 1e-10)**-1 + max(v_ε2, 1e-10)**-1 - 1.0)) * 4.0 * (293.0**3) * 5.67e-8
        v_R3 = 1.0 / (25.0 + rad_term)
    except:
        v_R3 = 0.0
    st.sidebar.metric("R3 (Calculated)", f"{v_R3:.6f}")


    # --- モデル評価の実行 (t4として表示) ---
    input_vec = [v_J, v_Je, v_R1, v_R2, v_R4, v_R5, v_as, v_al, v_ε1, v_ε2, v_ti, v_to]
    t4_val = model.evaluate(input_vec)


    # --- メインパネル表示 ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Predicted Output (t4)", value=f"{t4_val:.6f}")
        st.caption("Active Inputs: 10 variables")
        st.write("---")
        st.write("**Equation-based derived values:**")
        st.write(f"- **SAT:** {v_SAT:.4f}")
        st.write(f"- **R3:** {v_R3:.6f}")


    # --- 3Dグラフ表示 ---
    with col2:
        res = 25
        # X軸: as (インデックス4), Y軸: R4 (インデックス3)
        x_axis_grid = np.linspace(model.x_range[4][0], model.x_range[4][1], res)
        y_axis_grid = np.linspace(model.x_range[3][0], model.x_range[3][1], res)
        
        Z = np.zeros((res, res))
        for i in range(res):     # i は Y軸 (R4) に対応
            for j in range(res): # j は X軸 (as) に対応
                # temp_input の 6番目(as)に x_axis_grid[j] を、4番目(R4)に y_axis_grid[i] を代入
                temp_input = [v_J, v_Je, v_R1, v_R2, y_axis_grid[i], v_R5, x_axis_grid[j], v_al, v_ε1, v_ε2, v_ti, v_to]
                Z[i, j] = model.evaluate(temp_input)

        # meshgridを使わず1次元配列を直接渡すことで、Z[i, j] の i=Y軸, j=X軸 のマッピングを確実に固定
        fig = go.Figure(data=[go.Surface(z=Z, x=x_axis_grid, y=y_axis_grid, colorscale='Viridis')])
        
        fig.update_layout(
            scene=dict(xaxis_title='as', yaxis_title='R4', zaxis_title='t4'),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
if __name__ == "__main__":
    main()
