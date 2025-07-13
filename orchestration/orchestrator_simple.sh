#!/bin/bash
# ORQUESTADOR SIMPLE SDN - Versión base para ML

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SIMULATION_DURATION=${1:-300}

# Directorios
DATA_DIR="$PROJECT_DIR/data"
LOGS_DIR="$PROJECT_DIR/logs"

# PIDs
RYU_PID=""
TOPOLOGY_PID=""

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Limpieza de procesos
cleanup() {
    log "� Ejecutando limpieza..."
    
    if [[ ! -z "$TOPOLOGY_PID" ]] && kill -0 $TOPOLOGY_PID 2>/dev/null; then
        info "Terminando topología (PID: $TOPOLOGY_PID)"
        sudo kill -TERM $TOPOLOGY_PID 2>/dev/null
    fi
    
    if [[ ! -z "$RYU_PID" ]] && kill -0 $RYU_PID 2>/dev/null; then
        info "Terminando Ryu (PID: $RYU_PID)"
        kill -TERM $RYU_PID 2>/dev/null
    fi
    
    sudo mn -c &>/dev/null || true
    sudo pkill -f "ryu-manager" 2>/dev/null || true
    
    # Limpiar archivos temporales
    rm -f "$PROJECT_DIR/controller/sla_monitor_ml_temp.py" 2>/dev/null || true
    rm -f "$PROJECT_DIR/topology/temp_topology.py" 2>/dev/null || true
    
    log "✅ Limpieza completada"
}

# Configurar trap
trap cleanup EXIT INT TERM

# Verificar prerrequisitos
check_prerequisites() {
    log "� Verificando prerrequisitos..."
    
    # Verificar comandos
    for cmd in mn ryu-manager python3; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Comando no encontrado: $cmd"
            exit 1
        fi
    done
    
    # Verificar archivos
    if [[ ! -f "$PROJECT_DIR/topology/custom_topo.py" ]]; then
        error "No se encuentra: topology/custom_topo.py"
        exit 1
    fi
    
    # Detectar controlador a usar
    if [[ -f "$PROJECT_DIR/controller/sla_monitor_ml.py" ]]; then
        CONTROLLER_FILE="sla_monitor_ml.py"
        log "✅ Usando controlador ML"
    elif [[ -f "$PROJECT_DIR/controller/sla_monitor.py" ]]; then
        CONTROLLER_FILE="sla_monitor.py"
        log "✅ Usando controlador estándar"
    else
        error "No se encuentra controlador SLA"
        exit 1
    fi
    
    # Verificar sudo
    if ! sudo -n true 2>/dev/null; then
        error "Se requieren permisos sudo"
        exit 1
    fi
    
    log "✅ Prerrequisitos verificados"
}

# Configurar entorno
setup_environment() {
    log "� Configurando entorno..."
    
    # Crear directorios
    mkdir -p "$DATA_DIR" "$LOGS_DIR"
    
    # Configurar permisos
    sudo chown -R $USER:$USER "$DATA_DIR" "$LOGS_DIR" 2>/dev/null || true
    chmod -R 755 "$DATA_DIR" "$LOGS_DIR" 2>/dev/null || true
    
    # Limpiar procesos anteriores
    sudo mn -c &>/dev/null || true
    sudo pkill -f "ryu-manager" 2>/dev/null || true
    
    log "✅ Entorno configurado"
}

# Iniciar controlador Ryu
start_ryu() {
    log "� Iniciando controlador Ryu..."
    
    cd "$PROJECT_DIR/controller"
    
    # Crear copia temporal con CSV único
    local csv_file="../data/metrics_${TIMESTAMP}.csv"
    local temp_controller="${CONTROLLER_FILE%.py}_temp.py"
    
    cp "$CONTROLLER_FILE" "$temp_controller"
    sed -i "s|../data/metrics.csv|${csv_file}|g" "$temp_controller"
    
    # Iniciar Ryu
    ryu-manager "$temp_controller" --verbose > "$LOGS_DIR/ryu_${TIMESTAMP}.log" 2>&1 &
    RYU_PID=$!
    
    # Verificar inicio
    sleep 3
    if kill -0 $RYU_PID 2>/dev/null; then
        log "✅ Ryu iniciado correctamente (PID: $RYU_PID)"
    else
        error "Error iniciando Ryu"
        cat "$LOGS_DIR/ryu_${TIMESTAMP}.log"
        exit 1
    fi
}

# Iniciar topología
start_topology() {
    log "� Iniciando topología Mininet..."
    
    cd "$PROJECT_DIR/topology"
    
    # Crear script temporal para topología automatizada
    cat > temp_topology.py << 'TOPO'
#!/usr/bin/env python3
import sys
import os
sys.path.append('.')
from custom_topo import CustomSDNTopo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.log import setLogLevel
from mininet.link import TCLink
import time

def run_automated(duration):
    setLogLevel('info')
    topo = CustomSDNTopo()
    net = Mininet(
        topo=topo,
        controller=RemoteController,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )
    
    net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)
    net.start()
    
    print(f"� Topología iniciada por {duration} segundos")
    
    # Test de conectividad
    print("� Testing conectividad...")
    result = net.pingAll()
    print(f"� Resultado: {result}% packet loss")
    
    # Mantener activa
    print(f"⏱️  Manteniendo topología activa...")
    time.sleep(duration)
    
    print("� Cerrando topología...")
    net.stop()

if __name__ == '__main__':
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run_automated(duration)
TOPO
    
    # Ejecutar topología
    python3 temp_topology.py $SIMULATION_DURATION > "$LOGS_DIR/mininet_${TIMESTAMP}.log" 2>&1 &
    TOPOLOGY_PID=$!
    
    # Verificar inicio
    sleep 5
    if kill -0 $TOPOLOGY_PID 2>/dev/null; then
        log "✅ Topología iniciada (PID: $TOPOLOGY_PID)"
    else
        error "Error iniciando topología"
        cat "$LOGS_DIR/mininet_${TIMESTAMP}.log"
        exit 1
    fi
}

# Monitorear simulación
monitor_simulation() {
    log "�️  Monitoreando simulación..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + SIMULATION_DURATION))
    
    info "⏱️  Duración: $SIMULATION_DURATION segundos"
    info "� Inicio: $(date)"
    echo ""
    
    while [[ $(date +%s) -lt $end_time ]]; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local remaining=$((end_time - current_time))
        local progress=$((elapsed * 100 / SIMULATION_DURATION))
        
        # Barra de progreso simple
        local bar=""
        local bar_length=30
        local filled=$((progress * bar_length / 100))
        
        for ((i=0; i<filled; i++)); do bar+="█"; done
        for ((i=filled; i<bar_length; i++)); do bar+="░"; done
        
        printf "\r${BLUE}[PROGRESO]${NC} ${bar} ${progress}%% | ${elapsed}s/${SIMULATION_DURATION}s | ${remaining}s restantes"
        
        sleep 5
    done
    
    printf "\n"
    log "✅ Simulación completada"
}

# Mostrar resumen final
show_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                � SIMULACIÓN COMPLETADA                  ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    log "� Simulación SDN finalizada exitosamente"
    info "⏱️  Duración: $SIMULATION_DURATION segundos"
    info "�️  Sesión: $TIMESTAMP"
    
    echo ""
    echo -e "${YELLOW}� ARCHIVOS GENERADOS:${NC}"
    echo "  � Métricas:  data/metrics_${TIMESTAMP}.csv"
    echo "  � Logs:      logs/ryu_${TIMESTAMP}.log"
    
    # Verificar métricas generadas
    local metrics_file="$DATA_DIR/metrics_${TIMESTAMP}.csv"
    if [[ -f "$metrics_file" ]]; then
        local record_count=$(tail -n +2 "$metrics_file" | wc -l)
        echo ""
        echo -e "${YELLOW}� MÉTRICAS GENERADAS:${NC}"
        echo "  Total registros: $record_count"
        
        if [[ $record_count -gt 0 ]]; then
            echo "  Últimas 5 líneas:"
            tail -5 "$metrics_file" | while IFS= read -r line; do
                echo "    $line"
            done
        fi
    else
        echo ""
        echo -e "${YELLOW}⚠️  No se encontraron métricas generadas${NC}"
    fi
}

# Función principal
main() {
    info "� Iniciando Orquestador SDN Simple"
    info "� Proyecto: $PROJECT_DIR"
    info "⏱️  Duración: $SIMULATION_DURATION segundos"
    info "�️  Timestamp: $TIMESTAMP"
    echo ""
    
    check_prerequisites
    setup_environment
    start_ryu
    sleep 2
    start_topology
    sleep 3
    
    monitor_simulation
    
    # Esperar a que termine la topología
    if [[ ! -z "$TOPOLOGY_PID" ]]; then
        wait $TOPOLOGY_PID 2>/dev/null || true
    fi
    
    show_summary
}

# Mostrar ayuda
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "USO: $0 [DURACIÓN_EN_SEGUNDOS]"
    echo ""
    echo "EJEMPLOS:"
    echo "  $0         # Simulación de 5 minutos (300s)"
    echo "  $0 60      # Test rápido de 1 minuto"
    echo "  $0 600     # Simulación larga de 10 minutos"
    echo ""
    echo "ARCHIVOS GENERADOS:"
    echo "  - data/metrics_TIMESTAMP.csv"
    echo "  - logs/ryu_TIMESTAMP.log"
    exit 0
fi

# Ejecutar
main
