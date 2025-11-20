# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from algorithms import solve_fcfs, solve_sjf, solve_rr
from openai import OpenAI
import os
from dotenv import load_dotenv
import random

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(page_title="OS Process Scheduler", layout="wide")

st.title("Process Scheduling Simulator")
st.markdown("Define parameters, generate workloads, and analyze CPU scheduling algorithms.")

# --- SESSION STATE INITIALIZATION ---
if 'processes' not in st.session_state:
    st.session_state.processes = [
        {'id': 'P1', 'arrival': 0, 'burst': 5},
        {'id': 'P2', 'arrival': 1, 'burst': 3},
        {'id': 'P3', 'arrival': 2, 'burst': 2}
    ]

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("System Configuration")
    
    # API Key Handling
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        st.info("API Key loaded from environment.")
        api_key = env_key
    else:
        api_key = st.text_input("OpenRouter API Key", type="password")
    
    st.divider()
    
    # Algorithm Selection
    st.subheader("Algorithm Settings")
    algo_type = st.selectbox("Algorithm Strategy", ["FCFS", "SJF", "Round Robin"])
    
    quantum = 2
    if algo_type == "Round Robin":
        quantum = st.number_input("Time Quantum (ms)", min_value=1, value=2)
    
    st.divider()

    # --- SANDBOX TOOLS: RANDOM GENERATOR ---
    st.subheader("Workload Generator")
    st.markdown("Generate random process batches for stress testing.")
    
    num_procs = st.slider("Number of Processes", min_value=2, max_value=20, value=5)
    max_arrival = st.slider("Max Arrival Time", min_value=0, max_value=20, value=10)
    max_burst = st.slider("Max Burst Time", min_value=1, max_value=20, value=10)
    
    if st.button("Generate Random Workload"):
        new_data = []
        for i in range(num_procs):
            new_data.append({
                'id': f"P{i+1}",
                'arrival': random.randint(0, max_arrival),
                'burst': random.randint(1, max_burst)
            })
        # Ordenamos por llegada para limpieza visual, aunque el algoritmo lo maneja igual
        new_data.sort(key=lambda x: x['arrival'])
        st.session_state.processes = new_data
        st.rerun()

    if st.button("Clear All Processes"):
        st.session_state.processes = []
        st.rerun()

# --- MAIN AREA ---

# 1. DATA EDITOR (SANDBOX CORE)
st.subheader("1. Process Queue Configuration")
st.markdown("Edit values directly in the table below.")

if len(st.session_state.processes) > 0:
    # Convert to DataFrame
    df_input = pd.DataFrame(st.session_state.processes)
    
    # Data Editor allows direct manipulation of the grid
    edited_df = st.data_editor(
        df_input,
        num_rows="dynamic", # Allow adding/deleting rows directly in UI
        use_container_width=True,
        column_config={
            "id": "Process ID",
            "arrival": st.column_config.NumberColumn("Arrival Time (ms)", min_value=0, max_value=100),
            "burst": st.column_config.NumberColumn("Burst Time (ms)", min_value=1, max_value=100)
        }
    )
    
    # Update session state with edits
    # We convert back to list of dicts to keep compatibility with algorithms
    st.session_state.processes = edited_df.to_dict('records')
else:
    st.info("No processes defined. Use the sidebar generator or add rows manually.")

# 2. SIMULATION EXECUTION
st.divider()

if st.button("RUN SIMULATION", type="primary") and len(st.session_state.processes) > 0:
    
    # A. ALGORITHM EXECUTION
    with st.spinner('Processing scheduling logic...'):
        timeline = []
        # Ensure clean data types
        clean_processes = [
            {'id': str(p['id']), 'arrival': int(p['arrival']), 'burst': int(p['burst'])} 
            for p in st.session_state.processes
        ]

        if algo_type == "FCFS":
            timeline = solve_fcfs(clean_processes)
        elif algo_type == "SJF":
            timeline = solve_sjf(clean_processes)
        elif algo_type == "Round Robin":
            timeline = solve_rr(clean_processes, quantum)
            
        # B. METRICS CALCULATION
        df_timeline = pd.DataFrame(timeline)
        
        # Group by Process to find the LAST finish time (essential for RR)
        df_metrics = df_timeline.groupby('Process')['Finish'].max().reset_index()
        
        # Merge with original input to get Arrival and Burst
        df_original = pd.DataFrame(clean_processes)
        df_metrics = df_metrics.merge(df_original, left_on='Process', right_on='id')
        
        # Calculate OS Metrics
        df_metrics['Turnaround Time'] = df_metrics['Finish'] - df_metrics['arrival']
        df_metrics['Waiting Time'] = df_metrics['Turnaround Time'] - df_metrics['burst']
        
        # Averages
        avg_tat = df_metrics['Turnaround Time'].mean()
        avg_wt = df_metrics['Waiting Time'].mean()
        total_duration = df_timeline['Finish'].max()
        cpu_utilization = (df_metrics['burst'].sum() / total_duration) * 100 if total_duration > 0 else 0

        # --- OUTPUT SECTION ---
        st.subheader("2. Simulation Results")
        
        # KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Time", f"{total_duration} ms")
        kpi2.metric("Avg Turnaround", f"{avg_tat:.2f} ms")
        kpi3.metric("Avg Waiting", f"{avg_wt:.2f} ms")
        kpi4.metric("CPU Utilization", f"{cpu_utilization:.1f}%")
        
        # Gantt Chart
        st.markdown("### Gantt Chart Visualization")
        fig = px.timeline(df_timeline, x_start="Start", x_end="Finish", y="Process", color="Process", 
                          height=350)
        fig.update_yaxes(autorange="reversed") 
        fig.update_layout(
            xaxis_title="Time (ms)", 
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Metrics Table
        st.markdown("### Detailed Metrics")
        st.dataframe(
            df_metrics[['Process', 'arrival', 'burst', 'Finish', 'Turnaround Time', 'Waiting Time']], 
            use_container_width=True
        )

        # --- AI ANALYSIS ---
        st.subheader("3. AI Performance Analysis")
        
        if api_key:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            
            prompt = f"""
            Role: Senior Operating Systems Engineer.
            Task: Analyze the efficiency of the {algo_type} algorithm for the provided workload.
            
            Simulation Metrics:
            - Algorithm: {algo_type} (Quantum: {quantum if algo_type == 'Round Robin' else 'N/A'})
            - Avg Waiting Time: {avg_wt:.2f} ms
            - Avg Turnaround Time: {avg_tat:.2f} ms
            - CPU Utilization: {cpu_utilization:.1f}%
            
            Process Data (CSV format):
            {df_metrics[['Process', 'arrival', 'burst', 'Waiting Time']].to_csv(index=False)}
            
            Analysis Requirements:
            1. Identify bottlenecks or starvation issues.
            2. Comment on the suitability of {algo_type} for this specific burst/arrival distribution.
            3. Suggest an alternative algorithm if this one performed poorly.
            
            Format: Professional Markdown. Keep it concise.
            """
            
            try:
                with st.spinner('Generating engineering report...'):
                    completion = client.chat.completions.create(
                        model="meta-llama/llama-3.1-70b-instruct:free",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.markdown(completion.choices[0].message.content)
            except Exception as e:
                st.error(f"API Error: {e}")
        else:
            st.info("Configure API Key in sidebar to enable AI analysis.")

