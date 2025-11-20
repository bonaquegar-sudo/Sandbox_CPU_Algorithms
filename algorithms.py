# algorithms.py
from collections import deque

def solve_fcfs(processes):
    # Ordenamos por llegada
    processes.sort(key=lambda x: x['arrival'])
    current_time = 0
    timeline = [] # Guardaremos: (Proceso, Inicio, Fin)
    
    for p in processes:
        # Si la CPU está libre pero el proceso no ha llegado, avanzamos el tiempo
        if current_time < p['arrival']:
            current_time = p['arrival']
        
        start = current_time
        end = start + p['burst']
        timeline.append({'Process': p['id'], 'Start': start, 'Finish': end})
        current_time = end
        
    return timeline

def solve_sjf(processes):
    # SJF No expropiativo según tu PDF
    # Necesitamos controlar quién está disponible en la cola
    n = len(processes)
    completed = 0
    current_time = 0
    timeline = []
    # Hacemos una copia para no modificar el original
    pending = [p.copy() for p in processes] 
    pending.sort(key=lambda x: x['arrival']) # Orden inicial por llegada

    while completed < n:
        # Filtrar procesos que ya llegaron y no han sido atendidos
        available = [p for p in pending if p['arrival'] <= current_time and 'done' not in p]
        
        if not available:
            # Si no hay nadie, avanzamos el reloj al siguiente que llegue
            future_processes = [p for p in pending if 'done' not in p]
            if future_processes:
                current_time = min(p['arrival'] for p in future_processes)
            continue

        # LA CLAVE SJF: Elegir el de menor ráfaga (burst) de los disponibles
        shortest = min(available, key=lambda x: x['burst'])
        
        start = current_time
        end = start + shortest['burst']
        timeline.append({'Process': shortest['id'], 'Start': start, 'Finish': end})
        
        # Marcar como completado
        shortest['done'] = True
        completed += 1
        current_time = end
        
    return timeline

def solve_rr(processes, quantum):
    # Round Robin
    # Cola de procesos listos
    queue = deque()
    timeline = []
    current_time = 0
    
    # Copia con tiempo restante
    procs = {p['id']: {'arrival': p['arrival'], 'burst': p['burst'], 'rem': p['burst']} for p in processes}
    
    # Procesos ordenados por llegada para ir metiéndolos
    incoming = sorted(processes, key=lambda x: x['arrival'])
    incoming_idx = 0
    
    # Lógica inicial: meter el primero
    if incoming:
        current_time = incoming[0]['arrival']
        queue.append(incoming[0]['id'])
        incoming_idx += 1
    
    while queue or incoming_idx < len(incoming):
        # Si la cola está vacía pero faltan procesos por llegar
        if not queue and incoming_idx < len(incoming):
            current_time = incoming[incoming_idx]['arrival']
            queue.append(incoming[incoming_idx]['id'])
            incoming_idx += 1
            
        # Sacamos el siguiente proceso
        pid = queue.popleft()
        
        # Ejecutamos (Quantum o lo que le quede)
        time_slice = min(quantum, procs[pid]['rem'])
        start = current_time
        end = start + time_slice
        
        timeline.append({'Process': pid, 'Start': start, 'Finish': end})
        
        procs[pid]['rem'] -= time_slice
        current_time = end
        
        # IMPORTANTE: Verificar si llegaron nuevos procesos MIENTRAS ejecutábamos este
        while incoming_idx < len(incoming) and incoming[incoming_idx]['arrival'] <= current_time:
            queue.append(incoming[incoming_idx]['id'])
            incoming_idx += 1
            
        # Si al proceso le queda tiempo, vuelve a la cola (al final)
        if procs[pid]['rem'] > 0:
            queue.append(pid)
            
    return timeline
