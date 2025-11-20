# app.py - Version 1.2.1 (Fixed Function Order)
import streamlit as st
import pandas as pd
import plotly.express as px
from algorithms import solve_fcfs, solve_sjf, solve_rr
from openai import OpenAI
import os
from dotenv import load_dotenv
import random
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

load_dotenv()

# --- DEFINICIÓN DE FUNCIONES (Ahora está al principio para evitar errores) ---
def generate_pdf_report(results):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph(f"<b>Scheduling Analysis Report: {results['algo']}</b>", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.3*inch))
    
    # Metrics Summary
    metrics_text = f"""
    <b>Performance Metrics:</b><br/>
    - Total Execution Time: {results['total_duration']} ms<br/>
    - Average Turnaround Time: {results['avg_tat']:.2f} ms<br/>
    - Average Waiting Time: {results['avg_wt']:.2f} ms<br/>
    - CPU Utilization: {results['cpu_util']:.1f}%
    """
    story.append(Paragraph(metrics_text, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Gantt Chart Image (Matplotlib para PDF)
    fig, ax = plt.subplots(figsize=(8, 4))
    df = results['timeline']
    
    # Convertir colores de Plotly a algo que Matplotlib entienda o usar default
    colors_list = plt.cm.Paired.colors # Paleta de colores
    unique_procs = df['Process'].unique()
    color_map = {proc: colors_list[i % len(colors_list)] for i, proc in enumerate(unique_procs)}
    
    for idx, row in df.iterrows():
        ax.barh(row['Process'], row['Duration'], left=row['Start'], height=0.5, color=color_map[row['Process']])
    
    ax.set_xlabel('Time (ms)')
    ax.set_title('Gantt Chart Timeline')
    ax.invert_yaxis() # P1 arriba
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    story.append(Image(img_buffer, width=6*inch, height=3*inch))
    plt.close()
    
    story.append(Spacer(1, 0.2*inch))
    
    # Metrics Table
    data = [['Process', 'Arrival', 'Burst', 'Finish', 'TAT', 'WT']]
    for _, row in results['metrics'].iterrows():
        data.append([
            row['Process'], row['arrival'], row['burst'], 
            row['Finish'], row['Turnaround Time'], row['Waiting Time']
        ])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(table)
    
    # AI Analysis (if available)
    if 'ai_report' in results:
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("<b>AI Analysis:</b>", styles['Heading2']))
        # Basic Markdown cleanup for PDF (ReportLab doesn't support full MD natively)
        clean_text = results['ai_report'].replace('#', '').replace('**', '').replace('*', '•')
        # Split paragraphs by newlines to keep structure
        for para in clean_text.split('\n'):
            if para.strip():
                story.append(Paragraph(para, styles['BodyText']))
                story.append(Spacer(1, 0.05*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="OS Process Scheduler", layout="wide")

st.title("Process Scheduling Simulator")
st.markdown("Define parameters, generate workloads, and analyze CPU scheduling algorithms.")

# --- SESSION STATE ---
if 'processes' not in st.session_state:
    st.session_state.processes = [
        {'id': 'P1', 'arrival': 0, 'burst': 5},
        {'id': 'P2', 'arrival': 1, 'burst': 3},
        {'id': 'P3', 'arrival': 2, 'burst': 2}
    ]

# --- SIDEBAR ---
with st.sidebar:
    st.header("System Configuration")
    
    # API Key
    if "OPENROUTER_API_KEY" in st.secrets:
        env_key = st.secrets["OPENROUTER_API_KEY"]
    else:
        env_key = os.getenv("OPENROUTER_API_KEY")

    if env_key:
        st.info("API Key loaded.")
        api_key = env_key
    else:
        api_key = st.text_input("OpenRouter API Key", type="password")
    
    st.divider()
    
    # Algorithm Selection
    st.subheader("Algorithm Settings")
    algo_type = st.selectbox("Algorithm Strategy", ["FCFS", "SJF", "Round Robin", "Custom Algorithm"])
    
    quantum = 2
    if algo_type == "Round Robin":
        quantum = st.number_input("Time Quantum (ms)", min_value=1, value=2)
    
    # Language
    st.divider()
    st.subheader("AI Settings")
    language = st.selectbox("Report Language", ["Español", "English", "Français"])
    
    st.divider()

    # Workload Generator
    st.subheader("Workload Generator")
    num_procs = st.slider("Number of Processes", 2, 20, 5)
    max_arrival = st.slider("Max Arrival Time", 0, 20, 10)
    max_burst = st.slider("Max Burst Time", 1, 20, 10)
    
    if st.button("Generate Random Workload"):
        new_data = [{'id': f"P{i+1}", 'arrival': random.randint(0, max_arrival), 
                     'burst': random.randint(1, max_burst)} for i in range(num_procs)]
        new_data.sort(key=lambda x: x['arrival'])
        st.session_state.processes = new_data
        st.rerun()

    if st.button("Clear All Processes"):
        st.session_state.processes = []
        st.rerun()

# --- MAIN AREA ---

# Custom Algorithm Editor
if algo_type == "Custom Algorithm":
    st.subheader("Custom Algorithm Editor")
    st.markdown("Write your own scheduling algorithm. Your function must accept `processes` (list of dicts) and return `timeline` (list of dicts with 'Process', 'Start', 'Finish').")
    
    default_code = """def custom_scheduler(processes):
    # Example: Simple FCFS clone
    processes.sort(key=lambda x: x['arrival'])
    timeline = []
    current_time = 0
    
    for p in processes:
        if current_time < p['arrival']:
            current_time = p['arrival']
        start = current_time
        end = start + p['burst']
        timeline.append({'Process': p['id'], 'Start': start, 'Finish': end})
        current_time = end
    
    return timeline
"""
    
    custom_code = st.text_area("Python Code", value=default_code, height=300)
    st.session_state.custom_code = custom_code

# Data Editor
st.subheader("1. Process Queue Configuration")

if len(st.session_state.processes) > 0:
    df_input = pd.DataFrame(st.session_state.processes)
    edited_df = st.data_editor(
        df_input, num_rows="dynamic", use_container_width=True,
        column_config={
            "id": "Process ID",
            "arrival": st.column_config.NumberColumn("Arrival (ms)", min_value=0, max_value=100),
            "burst": st.column_config.NumberColumn("Burst (ms)", min_value=1, max_value=100)
        }
    )
    st.session_state.processes = edited_df.to_dict('records')
else:
    st.info("No processes defined.")

st.divider()

# --- SIMULATION ---
if st.button("RUN SIMULATION", type="primary") and len(st.session_state.processes) > 0:
    
    with st.spinner('Processing...'):
        clean_processes = [
            {'id': str(p['id']), 'arrival': int(p['arrival']), 'burst': int(p['burst'])} 
            for p in st.session_state.processes
        ]

        # Execute Algorithm
        try:
            if algo_type == "FCFS":
                timeline = solve_fcfs(clean_processes)
            elif algo_type == "SJF":
                timeline = solve_sjf(clean_processes)
            elif algo_type == "Round Robin":
                timeline = solve_rr(clean_processes, quantum)
            elif algo_type == "Custom Algorithm":
                namespace = {}
                exec(st.session_state.custom_code, namespace)
                timeline = namespace['custom_scheduler'](clean_processes)
        except Exception as e:
            st.error(f"Algorithm Error: {e}")
            st.stop()
            
        # Metrics Calculation
        df_timeline = pd.DataFrame(timeline)
        if not df_timeline.empty:
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

            # Store results
            st.session_state.results = {
                'algo': algo_type,
                'timeline': df_timeline,
                'metrics': df_metrics,
                'avg_tat': avg_tat,
                'avg_wt': avg_wt,
                'total_duration': total_duration,
                'cpu_util': cpu_utilization
            }

            # --- OUTPUT ---
            st.subheader("2. Simulation Results")
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Time", f"{total_duration} ms")
            kpi2.metric("Avg Turnaround", f"{avg_tat:.2f} ms")
            kpi3.metric("Avg Waiting", f"{avg_wt:.2f} ms")
            kpi4.metric("CPU Utilization", f"{cpu_utilization:.1f}%")
            
            st.markdown("### Gantt Chart")
            fig = px.bar(df_timeline, x="Duration", y="Process", base="Start", 
                         orientation='h', color="Process", text="Duration", height=350)
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(xaxis_title="Time (ms)", showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Detailed Metrics")
            st.dataframe(df_metrics[['Process', 'arrival', 'burst', 'Finish', 'Turnaround Time', 'Waiting Time']], 
                         use_container_width=True)

            # --- AI ANALYSIS ---
            st.subheader("3. AI Performance Analysis")
            
            ai_report = ""
            if api_key:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                
                prompt = f"""
                Role: Senior OS Engineer. Analyze {algo_type}.
                LANGUAGE: {language}
                
                Metrics: TAT={avg_tat:.2f}ms, WT={avg_wt:.2f}ms, CPU={cpu_utilization:.1f}%
                Data: {df_metrics[['Process', 'Waiting Time']].to_csv(index=False)}
                
                Provide: 1) Bottlenecks 2) Efficiency 3) Suggestions
                Format: Markdown (###, bullets, bold). NO code blocks.
                """
                
                try:
                    with st.spinner(f'Generating report ({language})...'):
                        completion = client.chat.completions.create(
                            model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        ai_report = completion.choices[0].message.content
                        if "<think>" in ai_report:
                            ai_report = ai_report.split("</think>")[-1].strip()
                        st.markdown(ai_report, unsafe_allow_html=True)
                        st.session_state.results['ai_report'] = ai_report
                except Exception as e:
                    st.error(f"API Error: {e}")
            else:
                st.info("Configure API Key for AI analysis.")

            # --- PDF EXPORT BUTTON ---
            st.divider()
            # Ahora la función ya está definida arriba, así que esto funcionará
            pdf_buffer = generate_pdf_report(st.session_state.results)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"scheduling_report_{algo_type}.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No processes scheduled. Check your parameters.")
