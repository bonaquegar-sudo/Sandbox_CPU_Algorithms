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
    if "OPENROUTER_API_KEY" in st.secrets:
        env_key = st.secrets["OPENROUTER_API_KEY"]
    else:
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
        
    # --- NUEVO: SELECCIÓN DE IDIOMA ---
    st.divider()
    st.subheader("AI Analysis Settings")
    language = st.selectbox("Report Language / Idioma", ["Español", "English", "Français"])
    
    st.divider()

    # --- SANDBOX TOOLS ---
    st.subheader("Workload Generator")
    
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
        new_data.sort(key=lambda x: x['arrival'])
        st.session_state.processes = new_data
        st.rerun()

    if st.button("Clear All Processes"):
        st.session_state.processes = []
        st.rerun()

# --- MAIN AREA ---

# 1. DATA EDITOR
st.subheader("1. Process Queue Configuration")

if len(st.session_state.processes) > 0:
    df_input = pd.DataFrame(st.session_state.processes)
    edited_df = st.data_editor(
        df_input,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": "Process ID",
            "arrival": st.column_config.NumberColumn("Arrival Time (ms)", min_value=0, max_value=100),
            "burst": st.column_config.NumberColumn("Burst Time (ms)", min_value=1, max_value=100)
        }
    )
    st.session_state.processes = edited_df.to_dict('records')
else:
    st.info("No processes defined.")

# 2. SIMULATION EXECUTION
st.divider()

if st.button("RUN SIMULATION", type="primary") and len(st.session_state.processes) > 0:
    
    with st.spinner('Processing scheduling logic...'):
        # A. ALGORITHM LOGIC (Igual que antes)
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
            
        # B. METRICS & VISUALIZATION
        df_timeline = pd.DataFrame(timeline)
        df_timeline['Duration'] = df_timeline['Finish'] - df_timeline['Start']
        
        df_metrics = df_timeline.groupby('Process')['Finish'].max().reset_index()
        df_original = pd.DataFrame(clean_processes)
        df_metrics = df_metrics.merge(df_original, left_on='Process', right_on='id')
        
        df_metrics['Turnaround Time'] = df_metrics['Finish'] - df_metrics['arrival']
        df_metrics['Waiting Time'] = df_metrics['Turnaround Time'] - df_metrics['burst']
        
        avg_tat = df_metrics['Turnaround Time'].mean()
        avg_wt = df_metrics['Waiting Time'].mean()
        total_duration = df_timeline['Finish'].max()
        cpu_utilization = (df_metrics['burst'].sum() / total_duration) * 100 if total_duration > 0 else 0

        # --- OUTPUT ---
        st.subheader("2. Simulation Results")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Time", f"{total_duration} ms")
        kpi2.metric("Avg Turnaround", f"{avg_tat:.2f} ms")
        kpi3.metric("Avg Waiting", f"{avg_wt:.2f} ms")
        kpi4.metric("CPU Utilization", f"{cpu_utilization:.1f}%")
        
        st.markdown("### Gantt Chart Visualization")
        fig = px.bar(
            df_timeline, x="Duration", y="Process", base="Start", 
            orientation='h', color="Process", text="Duration", height=350
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Time (ms)", showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Detailed Metrics")
        st.dataframe(
            df_metrics[['Process', 'arrival', 'burst', 'Finish', 'Turnaround Time', 'Waiting Time']], 
            use_container_width=True
        )

        # --- AI ANALYSIS (UPDATED) ---
        st.subheader("3. AI Performance Analysis")
        
        if api_key:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            
            # Prompt adaptado al idioma seleccionado
            prompt = f"""
            Role: Senior Operating Systems Engineer.
            Task: Analyze the efficiency of the {algo_type} algorithm for the provided workload.
            
            LANGUAGE REQUIREMENT: Please provide the response strictly in {language}.
            
            Simulation Metrics:
            - Algorithm: {algo_type}
            - Avg Waiting Time: {avg_wt:.2f} ms
            - CPU Utilization: {cpu_utilization:.1f}%
            
            Process Data:
            {df_metrics[['Process', 'arrival', 'burst', 'Waiting Time']].to_csv(index=False)}
            
            Analysis Requirements:
            1. Identify bottlenecks.
            2. Comment on efficiency.
            3. Suggest improvements.
            
            Format: Use clear Markdown headers (###), bullet points, and bold text for emphasis. Do NOT use code blocks for normal text.
            """
            
            try:
                with st.spinner(f'Generating report in {language}...'):
                    completion = client.chat.completions.create(
                        model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    # Extracción y Limpieza del contenido
                    content = completion.choices[0].message.content
                    
                    # FIX VISUAL: A veces los modelos DeepSeek ponen <think>...</think>. Lo limpiamos.
                    if "<think>" in content:
                        content = content.split("</think>")[-1].strip()
                    
                    # Renderizado seguro de Markdown
                    st.markdown(content, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"API Error: {e}")
        else:
            st.info("Configure API Key to enable AI analysis.")
