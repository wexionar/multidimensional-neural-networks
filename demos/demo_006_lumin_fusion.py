# =============================================================
# LUMIN-DEMO 006: The Integrated Fusion Engine (Symmetry & Precision)
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# Repository: https://github.com/wexionar/multi-dimensional-neural-networks
# License: MIT License
# Date: 2026-01-24
# Description: Advanced unified engine for ingestion and
#              high-speed vectorized resolution. Synchronized
#              multi-dimensional sectors for real-world
#              benchmarking and structural synthesis.
# =============================================================

import numpy as np
import pandas as pd
import io
import time
import secrets
import os

# --- ENVIRONMENT DETECTION ---
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# =============================================================
# CORE 1: LUMIN ORIGIN
# =============================================================
class LuminOrigin:
    def __init__(self, epsilon_val=0.05, epsilon_type=1, mode_type=1):
        self.epsilon_val = epsilon_val
        self.epsilon_type = int(epsilon_type)
        self.mode = 'diversity' if mode_type == 1 else 'purity'
        self.mode_label = "DIVERSITY (Mode 1)" if mode_type == 1 else "PURITY (Mode 2)"
        self.master_sectors = []
        self.current_sector_nodes = []
        self.D = None

    def _calculate_law(self, nodes):
        if len(nodes) < 2: return None, None
        nodes_np = np.array(nodes)
        X, Y = nodes_np[:, :-1], nodes_np[:, -1]
        A_mat = np.c_[X, np.ones(X.shape[0])]
        try:
            res = np.linalg.lstsq(A_mat, Y, rcond=None)[0]
            return res[:-1], res[-1]
        except: return None, None

    def _close_sector(self):
        if len(self.current_sector_nodes) < 2: return
        nodes = np.array(self.current_sector_nodes)
        W, B = self._calculate_law(self.current_sector_nodes)
        if W is not None:
            sector = np.concatenate([
                np.min(nodes[:, :-1], axis=0),
                np.max(nodes[:, :-1], axis=0),
                W, [B]
            ])
            self.master_sectors.append(sector)

    def ingest(self, cell):
        cell_np = np.array(cell, dtype=float)
        y_real = cell_np[-1]
        if self.D is None: self.D = len(cell_np) - 1
        if len(self.current_sector_nodes) < 2:
            self.current_sector_nodes.append(cell_np.tolist())
            return
        W, B = self._calculate_law(self.current_sector_nodes)
        y_pred = np.dot(cell_np[:-1], W) + B
        error_abs = abs(y_real - y_pred)
        threshold = self.epsilon_val if self.epsilon_type == 1 else abs(y_real) * self.epsilon_val
        if error_abs <= threshold:
            self.current_sector_nodes.append(cell_np.tolist())
        else:
            self._close_sector()
            self.current_sector_nodes = [self.current_sector_nodes[-1], cell_np.tolist()] if self.mode == 'diversity' else [cell_np.tolist()]

# =============================================================
# CORE 2: LUMIN RESOLUTION
# =============================================================
class LuminResolution:
    def __init__(self, sectors):
        self.sectors = np.array(sectors)
        self.D = (self.sectors.shape[1] - 1) // 3
        self.mins = self.sectors[:, :self.D]
        self.maxs = self.sectors[:, self.D : 2*self.D]
        self.weights = self.sectors[:, 2*self.D : 3*self.D]
        self.biases = self.sectors[:, -1]

    def resolve(self, X_input):
        X = np.atleast_2d(X_input)
        inside = np.all((X[:, np.newaxis, :] >= self.mins - 1e-9) &
                        (X[:, np.newaxis, :] <= self.maxs + 1e-9), axis=2)
        has_sector = np.any(inside, axis=1)
        indices = np.argmax(inside, axis=1)
        results = np.full(X.shape[0], np.nan)
        if np.any(has_sector):
            results[has_sector] = np.einsum('ij,ij->i', X[has_sector], self.weights[indices[has_sector]]) + self.biases[indices[has_sector]]
        return results

# =============================================================
# FLOW CONTROL: LUMIN FUSION 006
# =============================================================
def start_fusion_006():
    session_token = secrets.token_hex(4).upper()
    cache_data = None  # <-- PERSISTENCE

    while True:
        raw_data = None

        # --- 1. INGESTION ---
        while raw_data is None:
            print("\n" + "═"*45 + f"\n    LUMIN FUSION 006 - SESSION [{session_token}]\n" + "═"*45)
            print(" 1: Generate Synthetic Dataset\n 2: Load CSV file")
            if cache_data is not None:
                print(f" 3: Re-use dataset in memory ({len(cache_data):,} pts)")
            print(" 0: Exit\n" + "─"*45)

            mode_select = input("Select option: ").strip() or "1"
            if mode_select == "0": return

            if mode_select == "1":
                sug = "10000, 50, 100, 2"
                print("\n📝 CONFIGURATION: Points, Dims, Range, Type(1:Pos/2:Both)")
                user_params = input(f">> [Enter for {sug}]: ").strip() or sug
                try:
                    params = [float(x.strip()) for x in user_params.split(",")]
                    N, D, R_MAX, DATA_TYPE = int(params[0]), int(params[1]), params[2], int(params[3])
                    X = np.random.uniform(-R_MAX, R_MAX, (N, D)) if DATA_TYPE == 2 else np.random.uniform(0, R_MAX, (N, D))
                    Y = (np.sum(X, axis=1) / D) + np.random.normal(0, 0.01, N)
                    raw_data = np.hstack([X, Y.reshape(-1, 1)])
                    cache_data = raw_data # Save
                except Exception as e: print(f"❌ Error: {e}")
            elif mode_select == "2":
                if not IN_COLAB: print("⚠️ Colab environment required."); continue
                uploaded = files.upload()
                if not uploaded: continue
                df = pd.read_csv(io.BytesIO(uploaded[list(uploaded.keys())[0]]))
                raw_data = df.to_numpy()
                cache_data = raw_data # Save
            elif mode_select == "3" and cache_data is not None:
                raw_data = cache_data

        # --- 2. ENGINE CONFIGURATION ---
        print("\n" + "═"*45 + "\n    ENGINE CONFIGURATION\n" + "═"*45)
        print("📝 PARAMETERS: Norm(1:Sim/2:Dir), Eps Type(1:Abs/2:Rel), Eps Val, Mode(1:Div/2:Pur)")
        cfg_sug = "1, 1, 0.05, 1"
        cfg_data = input(f">> [Enter for {cfg_sug}]: ").strip() or cfg_sug
        try:
            config = [float(x.strip()) for x in cfg_data.split(",")]
            n_type, e_type, e_val, m_type = int(config[0]), int(config[1]), config[2], int(config[3])
        except:
            n_type, e_type, e_val, m_type = 1, 1, 0.05, 1

        # --- 3. ORIGIN EXECUTION ---
        print(f"\n🚀 PROCESSING SESSION {session_token}...")
        s_min, s_max = raw_data.min(axis=0), raw_data.max(axis=0)
        s_range = np.where((s_max - s_min) == 0, 1e-9, s_max - s_min)

        if n_type == 1:
            data_norm = 2 * (raw_data - s_min) / s_range - 1
            norm_label = "SYMMETRIC [-1, 1]"
        else:
            data_norm = (raw_data - s_min) / s_range
            norm_label = "DIRECT [0, 1]"

        origin = LuminOrigin(epsilon_val=e_val, epsilon_type=e_type, mode_type=m_type)
        t_start = time.perf_counter()
        for point in data_norm: origin.ingest(point)
        t_end = time.perf_counter()

        # METRICS
        duration = t_end - t_start
        pts, dims = len(raw_data), raw_data.shape[1] - 1
        sectors_list = origin.master_sectors
        num_sectors = len(sectors_list)
        comp_ratio = (1 - num_sectors/pts) * 100 if pts > 0 else 0
        speed = pts / duration
        throughput = (pts * dims) / duration

        mae_val, fidelity = 0.0, 0.0
        if num_sectors > 0:
            res_val = LuminResolution(sectors_list)
            y_pred = res_val.resolve(data_norm[:, :-1])
            valid_mask = ~np.isnan(y_pred)
            if np.any(valid_mask):
                mae_val = np.mean(np.abs(data_norm[valid_mask, -1] - y_pred[valid_mask]))
                fidelity = max(0, (1 - mae_val) * 100)

        estimated_weight_bytes = num_sectors * (dims * 3 + 1) * 8
        weight_str = f"{estimated_weight_bytes/1024:.2f} KB" if estimated_weight_bytes < 1024*1024 else f"{estimated_weight_bytes/(1024*1024):.2f} MB"

        print("\n📊 IGNITION REPORT: " + session_token)
        print("─"*45)
        print(f"• STRATEGY:      {origin.mode_label}")
        print(f"• NORMALIZATION: {norm_label}")
        print(f"• EPSILON:       {e_val} ({'RELATIVE %' if e_type == 2 else 'ABSOLUTE'})")
        print(f"• PRECISION:     {mae_val:.5f} MAE ({fidelity:.2f}% Fidelity)")
        print(f"• Y-RANGE:       [{s_min[-1]:,.2f} to {s_max[-1]:,.2f}]")
        print("─"*45)
        print(f"• PROCESSED:     {pts:,} points | {dims}D")
        print(f"• SECTORS:       {num_sectors} detected")
        print(f"• COMPRESSION:   {comp_ratio:.2f}%")
        print(f"• MAP SIZE:      {weight_str}")
        print("─"*45)
        print(f"• LEARNING SPD:  {speed:,.2f} pts/sec")
        print(f"• THROUGHPUT:    {throughput:,.2f} ops/sec")
        print(f"• LATENCY:       {duration:.4f} sec")
        print("─"*45)

        # --- 4. RESOLUTION TEST ---
        if num_sectors > 0:
            n_stress = 1000
            print(f"\n🔥 STARTING RESOLUTION STRESS TEST ({n_stress} Points)...")
            res_engine = LuminResolution(sectors_list)
            t_min, t_max = (-1, 1) if n_type == 1 else (0, 1)
            test_points = np.random.uniform(t_min, t_max, (n_stress, dims))
            t_res_start = time.perf_counter()
            _ = res_engine.resolve(test_points)
            t_res_end = time.perf_counter()
            throughput_res = n_stress / (t_res_end - t_res_start)
            print(f"🚀 VECTORIZED THROUGHPUT: {throughput_res:,.2f} ops/sec")
            print("─"*45)

        # --- 5. EXIT ---
        print("\nWhat would you like to do now, colleague?")
        print(" 1: Return to start (New Ingestion/Parameters)")
        print(" 2: Exit and Save .npy file")
        print(" 0: Exit (Without saving)")
        print("─"*45)

        post_action = input(">> Select: ").strip() or "1"
        if post_action == "1": continue
        elif post_action == "2":
            filename = f"LUMIN_DATA_{session_token}.npy"
            np.save(filename, {
                'sectors': sectors_list,
                'token': session_token,
                'dims': dims,
                'norm_type': n_type,
                's_min': s_min,
                's_range': s_range,
                'epsilon_type': e_type,
                'epsilon_val': e_val
            })
            print(f"\n✅ SAVED: {filename}")
            if IN_COLAB: files.download(filename)
            break
        elif post_action == "0": break

start_fusion_006()
