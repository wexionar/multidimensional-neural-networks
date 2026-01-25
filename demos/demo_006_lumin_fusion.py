# =============================================================
# LUMIN-DEMO 006: The Integrated Fusion Engine (Symmetry & Precision)
# =============================================================
# Project: SLRM-nD (Lumin Core)
# Developers: Alex Kinetic & Gemini
# Repository: https://github.com/wexionar/multi-dimensional-neural-networks
# License: MIT License
# Date: 2026-01-25
# Description: Advanced unified engine for ingestion and
#                high-speed vectorized resolution. Synchronized
#                multi-dimensional sectors for real-world
#                benchmarking and structural synthesis.
# =============================================================

import numpy as np
import pandas as pd
import io
import time
import secrets
import os
from google.colab import output

# --- GLOBAL SYSTEM CACHE ---
LUMIN_CACHE_MAP = None

# --- ENVIRONMENT DETECTION ---
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# =============================================================
# CORE 1: LUMIN FUSION ORIGIN (PART A)
# =============================================================
class LuminFusionOrigin:
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
# CORE 2: LUMIN FUSION RESOLUTION (PART B)
# =============================================================
class LuminFusionResolution:
    def __init__(self, metadata):
        self.sectors = np.array(metadata['sectors'])
        self.n_type = int(metadata.get('norm_type', 1))
        self.s_min = metadata['s_min']
        self.s_range = metadata['s_range']
        self.token = metadata.get('token', 'UNKNOWN')
        self.e_val = metadata.get('epsilon_val', 'N/A')
        self.mode = metadata.get('mode', 'N/A')

        self.D = (self.sectors.shape[1] - 1) // 3
        self.mins = self.sectors[:, :self.D]
        self.maxs = self.sectors[:, self.D : 2*self.D]
        self.weights = self.sectors[:, 2*self.D : 3*self.D]
        self.biases = self.sectors[:, -1]

    def resolve(self, X_raw, tolerance=0.0):
        X_raw = np.atleast_2d(X_raw)
        if self.n_type == 1:
            X_norm = 2 * (X_raw - self.s_min[:-1]) / self.s_range[:-1] - 1
        else:
            X_norm = (X_raw - self.s_min[:-1]) / self.s_range[:-1]

        inside = np.all((X_norm[:, np.newaxis, :] >= self.mins - (tolerance + 1e-9)) &
                        (X_norm[:, np.newaxis, :] <= self.maxs + (tolerance + 1e-9)), axis=2)

        has_sector = np.any(inside, axis=1)
        indices = np.argmax(inside, axis=1)

        results_norm = np.full(X_raw.shape[0], np.nan)
        if np.any(has_sector):
            relevant_indices = indices[has_sector]
            results_norm[has_sector] = np.einsum('ij,ij->i', X_norm[has_sector],
                                               self.weights[relevant_indices]) + self.biases[relevant_indices]

        if self.n_type == 1:
            results_real = (results_norm + 1) * self.s_range[-1] / 2 + self.s_min[-1]
        else:
            results_real = results_norm * self.s_range[-1] + self.s_min[-1]
        return results_real

# =============================================================
# SYSTEM FLOWS
# =============================================================

def run_origin_flow():
    """ ORIGIN INGESTION FLOW """
    global LUMIN_CACHE_MAP
    session_token = secrets.token_hex(4).upper()
    cache_data = None

    while True:
        raw_data = None
        while raw_data is None:
            print("\n" + "═"*45 + f"\n    LUMIN FUSION 006 - SESSION [{session_token}]\n" + "═"*45)
            print(" 1: Generate Synthetic Dataset\n 2: Load CSV file")
            print(" 0: Back to Main Menu\n" + "─"*45)

            mode_select = input("Select option: ").strip() or "1"
            if mode_select == "0": return False

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
                    cache_data = raw_data
                except Exception as e: print(f"❌ Error: {e}")
            elif mode_select == "2":
                if not IN_COLAB: print("⚠️ Colab environment required."); continue
                uploaded = files.upload()
                if not uploaded: continue
                df = pd.read_csv(io.BytesIO(uploaded[list(uploaded.keys())[0]]))
                raw_data = df.to_numpy()
                cache_data = raw_data

        while True: # --- FINE-TUNING LOOP ---
            print("\n" + "═"*45 + "\n    ENGINE CONFIGURATION\n" + "═"*45)
            print("📝 PARAMETERS: Norm(1:Sim/2:Dir), Eps Type(1:Abs/2:Rel), Eps Val, Mode(1:Div/2:Pur)")
            cfg_sug = "1, 1, 0.05, 1"
            cfg_data = input(f">> [Enter for {cfg_sug}]: ").strip() or cfg_sug
            try:
                config = [float(x.strip()) for x in cfg_data.split(",")]
                n_type, e_type, e_val, m_type = int(config[0]), int(config[1]), config[2], int(config[3])
            except:
                n_type, e_type, e_val, m_type = 1, 1, 0.05, 1

            print(f"\n🚀 PROCESSING SESSION {session_token}...")
            s_min, s_max = raw_data.min(axis=0), raw_data.max(axis=0)
            s_range = np.where((s_max - s_min) == 0, 1e-9, s_max - s_min)

            if n_type == 1:
                data_norm = 2 * (raw_data - s_min) / s_range - 1
                norm_label = "SYMMETRIC [-1, 1]"
            else:
                data_norm = (raw_data - s_min) / s_range
                norm_label = "DIRECT [0, 1]"

            origin = LuminFusionOrigin(epsilon_val=e_val, epsilon_type=e_type, mode_type=m_type)
            t_start = time.perf_counter()
            for point in data_norm: origin.ingest(point)
            t_end = time.perf_counter()

            duration = t_end - t_start
            pts, dims = len(raw_data), raw_data.shape[1] - 1
            sectors_list = origin.master_sectors
            num_sectors = len(sectors_list)
            comp_ratio = (1 - num_sectors/pts) * 100 if pts > 0 else 0
            speed = pts / duration
            throughput = (pts * dims) / duration

            mae_val, fidelity = 0.0, 0.0
            metadata_report = {'sectors': sectors_list, 'norm_type': n_type, 's_min': s_min, 's_range': s_range, 'token': session_token, 'epsilon_val': e_val, 'mode': origin.mode_label}

            if num_sectors > 0:
                res_val = LuminFusionResolution(metadata_report)
                y_pred = res_val.resolve(raw_data[:, :-1])
                valid_mask = ~np.isnan(y_pred)
                if np.any(valid_mask):
                    mae_val = np.mean(np.abs(raw_data[valid_mask, -1] - y_pred[valid_mask]))
                    fidelity = max(0, (1 - mae_val) * 100)

            # UPDATE GLOBAL CACHE
            LUMIN_CACHE_MAP = metadata_report

            estimated_weight_bytes = num_sectors * (dims * 3 + 1) * 8
            weight_str = f"{estimated_weight_bytes/1024:.2f} KB" if estimated_weight_bytes < 1024*1024 else f"{estimated_weight_bytes/(1024*1024):.2f} MB"

            print("\n📊 IGNITION REPORT: " + session_token)
            print("─"*45)
            print(f"• STRATEGY:      {origin.mode_label}")
            print(f"• NORMALIZATION: {norm_label}")
            print(f"• EPSILON:        {e_val} ({'RELATIVE %' if e_type == 2 else 'ABSOLUTE'})")
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

            if num_sectors > 0:
                n_stress = 1000
                print(f"\n🔥 STARTING RESOLUTION STRESS TEST ({n_stress} Points)...")
                res_engine = LuminFusionResolution(metadata_report)
                t_min, t_max = (-1, 1) if n_type == 1 else (0, 1)
                test_points = np.random.uniform(t_min, t_max, (n_stress, dims))
                t_res_start = time.perf_counter()
                _ = res_engine.resolve(test_points)
                t_res_end = time.perf_counter()
                throughput_res = n_stress / (t_res_end - t_res_start)
                print(f"🚀 VECTORIZED THROUGHPUT: {throughput_res:,.2f} ops/sec")
                print("─"*45)

            print("\nSelect next action:")
            print(" 1: Adjust Parameters (Fine-Tuning)")
            print(" 2: New Dataset (Reset)")
            print(" 3: Save .npy & Exit")
            print(" 0: Back to Main Menu")
            print("─"*45)

            post_action = input(">> Select: ").strip() or "1"
            if post_action == "1": continue
            elif post_action == "2": break
            elif post_action == "3":
                filename = f"lumin_fusion_dataset_{session_token.lower()}.npy"
                np.save(filename, {**metadata_report, 'token': session_token.lower(), 'dims': dims, 'epsilon_type': e_type})
                print(f"\n✅ SAVED: {filename}")
                if IN_COLAB:
                    files.download(filename)
                    time.sleep(3)
                return True
            elif post_action == "0": return False

def run_resolution_flow():
    """ RESOLUTION INFERENCE FLOW """
    global LUMIN_CACHE_MAP
    print("\n" + "═"*45 + "\n    LUMIN INDEPENDENT RESOLUTION ENGINE\n" + "═"*45)

    metadata = None
    if LUMIN_CACHE_MAP is not None:
        print(f"💡 Active Session Detected: [{LUMIN_CACHE_MAP.get('token')}]")
        print(" 1: Upload new .npy file")
        print(" 2: Use .npy from Cache")
        res_choice = input("\nSELECT SOURCE >> ").strip() or "2"
        if res_choice == "2":
            metadata = LUMIN_CACHE_MAP

    if metadata is None:
        if IN_COLAB:
            print("Please upload your .npy map file...")
            uploaded = files.upload()
            if uploaded:
                npy_files = [f for f in uploaded.keys() if f.endswith('.npy')]
                if npy_files: metadata = np.load(npy_files[0], allow_pickle=True).item()
        else:
            print("⚠️ Cache empty and manual upload required.")

    if metadata is None: return False
    engine = LuminFusionResolution(metadata)

    print(f"\n✅ ENGINE LOADED [Session: {engine.token}]")
    print(f"• Dimensions:    {engine.D}D")
    print(f"• Sectors:       {len(engine.sectors)}")
    print(f"• DNA Norm:      {'SYMMETRIC' if engine.n_type == 1 else 'STANDARD'}")
    print(f"• Mode:          {engine.mode}")
    print(f"• Precision:     ε = {engine.e_val}")
    print(f"• Input Range:   [{engine.s_min[0]:.2f} to {(engine.s_min[0]+engine.s_range[0]):.2f}]")
    print(f"• Output Scale: [{engine.s_min[-1]:.2f} to {(engine.s_min[-1]+engine.s_range[-1]):.2f}]")
    print("─"*45)

    while True:
        print("\nMAIN MENU:")
        print(" [1] INDIVIDUAL INFERENCE (Manual)")
        print(" [2] CSV BATCH PROCESSING (Massive)")
        print(" [0] EXIT TO MAIN FUSION")
        cmd = input("\nSELECT ACTION >> ").strip().lower() or "1"

        if cmd == '0': return False

        if cmd == '1':
            try:
                sug_tol = float(engine.e_val) * 0.02
                t_in = input(f"\nSet tolerance (Sug: {sug_tol:.4f}): ").strip()
            except:
                t_in = input("\nSet tolerance (Default 0.0): ").strip()
            
            try: tol = float(t_in) if t_in else 0.0
            except: tol = 0.0

            cal_norm_x = (engine.mins[0] + engine.maxs[0]) / 2
            if engine.n_type == 1:
                cal_real_x = ((cal_norm_x + 1) / 2) * engine.s_range[:-1] + engine.s_min[:-1]
            else:
                cal_real_x = (cal_norm_x * engine.s_range[:-1]) + engine.s_min[:-1]
            cal_str = ",".join([f"{v:.4f}" for v in cal_real_x])

            while True:
                print("\nCOMMANDS: [coords] | 's' = Sample | 'm' = Menu | 'e' = Exit")
                u_input = input("ENTER INPUT >> ").strip().lower()

                if u_input == 'm': break
                if u_input == 'e': return True

                target_point = None
                if u_input == 's':
                    if IN_COLAB:
                        try:
                            output.eval_js(f'navigator.clipboard.writeText("{cal_str}")')
                            print("📋 Sample copied to clipboard...")
                        except:
                            pass
                    target_point = cal_real_x
                else:
                    try:
                        target_point = np.array([float(x) for x in u_input.split(",")])
                    except:
                        print("❌ ERROR: Use val1,val2,val3..."); continue

                if target_point is not None:
                    if len(target_point) != engine.D:
                        print(f"❌ ERROR: Dimension mismatch"); continue

                    if len(target_point) > 10:
                        p1 = ", ".join([f"{v:.4f}" for v in target_point[:5]])
                        p2 = ", ".join([f"{v:.4f}" for v in target_point[-5:]])
                        print(f"📋 Input: [{p1} ... {p2}]")
                    else:
                        print(f"📋 Input: [{', '.join([f'{v:.4f}' for v in target_point])}]")

                    t0 = time.perf_counter()
                    res = engine.resolve(target_point, tol)
                    dt = (time.perf_counter() - t0)*1e6
                    if np.isnan(res[0]): print("⚠️  OUT OF BOUNDS")
                    else: print(f"🔮 PREDICTION: {res[0]:,.4f} | ⚡ {dt:.2f} us")

        elif cmd == '2':
            t_in = input("\nCSV Mode - Set tolerance (Default 0.0): ").strip()
            try: tol = float(t_in) if t_in else 0.0
            except: tol = 0.0
            if IN_COLAB:
                print("Upload CSV file for batch processing...")
                uploaded_csv = files.upload()
                if uploaded_csv:
                    csv_name = list(uploaded_csv.keys())[0]
                    raw_data = np.genfromtxt(io.BytesIO(uploaded_csv[csv_name]), delimiter=',', filling_values=0)
                    if len(raw_data.shape) == 1: raw_data = raw_data.reshape(1, -1)
                    if np.isnan(raw_data[0,0]): raw_data = raw_data[1:]

                    print(f"⚙️  Processing {len(raw_data)} rows...")
                    t0 = time.perf_counter()
                    batch_results = engine.resolve(raw_data, tol)
                    total_time = time.perf_counter() - t0

                    out_name = f"RESULTS_{engine.token}.csv"
                    np.savetxt(out_name, batch_results, delimiter=',', fmt='%.6f', header='prediction', comments='')
                    print(f"✅ Downloaded {out_name} in {total_time:.4f}s")
                    files.download(out_name)
                    time.sleep(3)
                    return True

def main_fusion_controller():
    while True:
        print("\n" + "═"*45)
        print("    LUMIN FUSION SYSTEM - INTEGRATED CONTROL")
        print("═"*45)
        print(" [1] ORIGIN: Create Map from Data (Ingest) [DEFAULT]")
        print(" [2] RESOLUTION: Use Existing Map (Inference)")
        print(" [0] EXIT SYSTEM")
        print("─"*45)

        choice = input("SELECT ACTION >> ").strip() or "1"
        should_exit = False
        if choice == "1": should_exit = run_origin_flow()
        elif choice == "2": should_exit = run_resolution_flow()
        elif choice == "0": break
        if should_exit: break

if __name__ == "__main__":
    main_fusion_controller()
    
