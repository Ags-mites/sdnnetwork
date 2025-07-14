#!/bin/bash
# Script de Simulación SDN Unificado

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

SIMULATION_DURATION=${1:-300}
TARGET_RECORDS=${2:-3000}
MODE=${3:-"single"} 

DATA_DIR="$PROJECT_DIR/data"
LOGS_DIR="$PROJECT_DIR/logs"
RESULTS_DIR="$PROJECT_DIR/results"

CONTROLLER="$PROJECT_DIR/controller/sla_monitor.py"
TOPOLOGY="$PROJECT_DIR/topology/custom_topo.py"
FINAL_DATASET="$DATA_DIR/sla_dataset_${TIMESTAMP}.csv"

RYU_PID=""
TOPOLOGY_PID=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

show_banner() {
    clear
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                     SIMULADOR SDN                          ║"
    echo "║          Machine Learning + SLA Monitoring                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo "Modo: $MODE | Duración: ${SIMULATION_DURATION}s | Target: $TARGET_RECORDS registros"
    echo ""
}

cleanup() {
    log "Ejecutando limpieza..."
    
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
    
    rm -f "$PROJECT_DIR/controller"/*_temp.py 2>/dev/null || true
    rm -f "$PROJECT_DIR/topology"/temp_*.py 2>/dev/null || true
    
    log "Limpieza completada"
}

trap cleanup EXIT INT TERM

check_prerequisites() {
    step "Verificando prerrequisitos..."
    
    for cmd in mn ryu-manager python3; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Comando no encontrado: $cmd"
            exit 1
        fi
    done
    
    if [[ ! -f "$CONTROLLER" ]]; then
        error "Controlador no encontrado: $CONTROLLER"
        exit 1
    fi
    
    if [[ ! -f "$TOPOLOGY" ]]; then
        error "Topología no encontrada: $TOPOLOGY"
        exit 1
    fi
    
    if ! sudo -n true 2>/dev/null; then
        error "Se requieren permisos sudo"
        exit 1
    fi
    
    log "Prerrequisitos verificados"
}

setup_environment() {
    step "Configurando entorno..."
    
    mkdir -p "$DATA_DIR" "$LOGS_DIR" "$RESULTS_DIR"
    
    sudo chown -R $USER:$USER "$DATA_DIR" "$LOGS_DIR" "$RESULTS_DIR" 2>/dev/null || true
    chmod -R 755 "$DATA_DIR" "$LOGS_DIR" "$RESULTS_DIR" 2>/dev/null || true
    
    sudo mn -c &>/dev/null || true
    sudo pkill -f "ryu-manager" 2>/dev/null || true
    
    log "Entorno configurado"
}

start_ryu_controller() {
    step "Iniciando controlador Ryu..."
    
    cd "$PROJECT_DIR/controller"
    
    export SLA_TARGET_RECORDS="$TARGET_RECORDS"
    export SLA_CSV_FILE="$FINAL_DATASET"
    
    ryu-manager sla_monitor.py --verbose > "$LOGS_DIR/ryu_${TIMESTAMP}.log" 2>&1 &
    RYU_PID=$!
    
    sleep 3
    if kill -0 $RYU_PID 2>/dev/null; then
        log "Ryu iniciado correctamente (PID: $RYU_PID)"
    else
        error "Error iniciando Ryu"
        cat "$LOGS_DIR/ryu_${TIMESTAMP}.log"
        exit 1
    fi
}

start_mininet_topology() {
    step "Iniciando topología Mininet..."
    
    cd "$PROJECT_DIR/topology"
    
    cat > temp_topology_${TIMESTAMP}.py << 'TOPO_EOF'
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

def run_simulation(duration):
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
    
    print(f"Topología iniciada por {duration} segundos")
    
    # Test inicial de conectividad
    print("Ejecutando test de conectividad...")
    result = net.pingAll()
    print(f"Conectividad: {result}% packet loss")
    
    # Mantener topología activa para recolección de métricas
    print(f"Recolectando métricas SLA por {duration}s...")
    time.sleep(duration)
    
    print("Cerrando topología...")
    net.stop()

if __name__ == '__main__':
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run_simulation(duration)
TOPO_EOF
    
    python3 temp_topology_${TIMESTAMP}.py $SIMULATION_DURATION > "$LOGS_DIR/mininet_${TIMESTAMP}.log" 2>&1 &
    TOPOLOGY_PID=$!
    
    sleep 5
    if kill -0 $TOPOLOGY_PID 2>/dev/null; then
        log "Topología iniciada (PID: $TOPOLOGY_PID)"
    else
        error "Error iniciando topología"
        cat "$LOGS_DIR/mininet_${TIMESTAMP}.log"
        exit 1
    fi
}

monitor_simulation() {
    step "Monitoreando simulación..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + SIMULATION_DURATION))
    
    info "Duración: $SIMULATION_DURATION segundos"
    info "Target de registros: $TARGET_RECORDS"
    echo ""
    
    while [[ $(date +%s) -lt $end_time ]]; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local remaining=$((end_time - current_time))
        local progress=$((elapsed * 100 / SIMULATION_DURATION))
        
        local current_records=0
        if [[ -f "$FINAL_DATASET" ]]; then
            current_records=$(tail -n +2 "$FINAL_DATASET" 2>/dev/null | wc -l)
        fi
        
        local bar=""
        local bar_length=30
        local filled=$((progress * bar_length / 100))
        
        for ((i=0; i<filled; i++)); do bar+="█"; done
        for ((i=filled; i<bar_length; i++)); do bar+="░"; done
        
        printf "\r${BLUE}[PROGRESO]${NC} ${bar} ${progress}%% | ${elapsed}s/${SIMULATION_DURATION}s | Registros: ${current_records}/${TARGET_RECORDS}"
        
        if [[ $current_records -ge $TARGET_RECORDS ]]; then
            printf "\n"
            log "Target de registros alcanzado: $current_records"
            break
        fi
        
        sleep 5
    done
    
    printf "\n"
    log "Simulación completada"
}

run_batch_simulation() {
    step "Ejecutando simulación en modo batch..."
    
    local batch_size=600
    local batch_duration=240
    local total_batches=$(( (TARGET_RECORDS + batch_size - 1) / batch_size ))
    local successful_batches=0
    
    info "Configuración batch: $total_batches lotes de $batch_size registros"
    
    for ((batch=1; batch<=total_batches; batch++)); do
        log "Ejecutando lote $batch/$total_batches..."
        
        export SLA_TARGET_RECORDS="$batch_size"
        export SLA_CSV_FILE="$DATA_DIR/batch_${batch}_${TIMESTAMP}.csv"
        
        if run_single_simulation $batch_duration; then
            successful_batches=$((successful_batches + 1))
            log "Lote $batch completado exitosamente"
        else
            error "Lote $batch falló"
        fi
        
        # Pausa entre lotes
        if [[ $batch -lt $total_batches ]]; then
            sleep 10
        fi
    done
    
    consolidate_batch_results $successful_batches
}

run_single_simulation() {
    local duration=${1:-$SIMULATION_DURATION}
    
    start_ryu_controller
    sleep 2
    start_mininet_topology
    sleep 3
    
    if [[ ! -z "$TOPOLOGY_PID" ]]; then
        wait $TOPOLOGY_PID 2>/dev/null || true
    fi
    
    return 0
}

consolidate_batch_results() {
    local successful_batches=$1
    
    step "Consolidando resultados de $successful_batches lotes..."
    
    echo "timestamp,src_host,dst_host,latency_ms,jitter_ms,packet_loss_percent,throughput_mbps,sla_compliant,path_type,traffic_pattern,hour_of_day,is_weekend,batch_id" > "$FINAL_DATASET"
    
    local total_records=0
    for ((batch=1; batch<=successful_batches; batch++)); do
        local batch_file="$DATA_DIR/batch_${batch}_${TIMESTAMP}.csv"
        if [[ -f "$batch_file" ]]; then
            tail -n +2 "$batch_file" | while IFS= read -r line; do
                echo "${line},batch_${batch}" >> "$FINAL_DATASET"
            done
            local batch_records=$(tail -n +2 "$batch_file" | wc -l)
            total_records=$((total_records + batch_records))
            rm -f "$batch_file"
        fi
    done
    
    log "Dataset consolidado: $total_records registros en $FINAL_DATASET"
}

show_results_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                  SIMULACIÓN COMPLETADA                       ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    log "Simulación SDN finalizada exitosamente"
    info "Modo de ejecución: $MODE"
    info "Duración total: $SIMULATION_DURATION segundos"
    info "Sesión: $TIMESTAMP"
    
    echo ""
    echo -e "${YELLOW}ARCHIVOS GENERADOS:${NC}"
    echo "  Dataset: $FINAL_DATASET"
    echo "  Logs: $LOGS_DIR/*_${TIMESTAMP}.log"
    
    # Verificar dataset generado
    if [[ -f "$FINAL_DATASET" ]]; then
        local record_count=$(tail -n +2 "$FINAL_DATASET" | wc -l)
        local sla_true=$(tail -n +2 "$FINAL_DATASET" | cut -d',' -f8 | grep -c "True" || echo "0")
        local sla_false=$(tail -n +2 "$FINAL_DATASET" | cut -d',' -f8 | grep -c "False" || echo "0")
        
        echo ""
        echo -e "${YELLOW}ESTADÍSTICAS DEL DATASET:${NC}"
        echo "  Total registros: $record_count"
        echo "  SLA Cumple (True): $sla_true"
        echo "  SLA Viola (False): $sla_false"
        
        if [[ $((sla_true + sla_false)) -gt 0 ]]; then
            local sla_ratio=$((sla_true * 100 / (sla_true + sla_false)))
            echo "  Ratio SLA: ${sla_ratio}% cumple"
        fi
        
        if [[ $record_count -ge $TARGET_RECORDS ]]; then
            echo -e "  ${GREEN}✅ TARGET ALCANZADO${NC}"
        else
            echo -e "  ${YELLOW}⚠️ Target parcial: $record_count/$TARGET_RECORDS${NC}"
        fi
    else
        echo -e "${RED}❌ No se generó el dataset${NC}"
    fi
    
    echo ""
    log "Listo para Fase 4: Evaluación ML"
    info "Ejecuta: ./evaluate_ml.sh $FINAL_DATASET"
}

show_help() {
    echo "SIMULADOR SDN UNIFICADO"
    echo "======================="
    echo ""
    echo "USO: $0 [DURACIÓN] [TARGET_RECORDS] [MODO]"
    echo ""
    echo "PARÁMETROS:"
    echo "  DURACIÓN       Duración en segundos (default: 300)"
    echo "  TARGET_RECORDS Número de registros objetivo (default: 3000)"
    echo "  MODO          Modo de ejecución: single|batch|evaluation (default: single)"
    echo ""
    echo "EJEMPLOS:"
    echo "  $0                    # Simulación básica 5 min, 3000 registros"
    echo "  $0 600 5000 batch    # Simulación batch 10 min, 5000 registros"
    echo "  $0 120 1000 single   # Test rápido 2 min, 1000 registros"
    echo ""
    echo "MODOS:"
    echo "  single     - Simulación continua estándar"
    echo "  batch      - Simulación por lotes para datasets grandes"
    echo "  evaluation - Modo optimizado para evaluación ML"
    echo ""
    echo "ARCHIVOS GENERADOS:"
    echo "  data/sla_dataset_TIMESTAMP.csv - Dataset principal"
    echo "  logs/ryu_TIMESTAMP.log         - Log del controlador"
    echo "  logs/mininet_TIMESTAMP.log     - Log de Mininet"
    echo ""
    exit 0
}

main() {
    show_banner
    
    log "Iniciando simulación SDN unificada"
    info "Proyecto: $PROJECT_DIR"
    info "Target: $TARGET_RECORDS registros en $SIMULATION_DURATION segundos"
    echo ""
    
    check_prerequisites
    setup_environment
    
    case $MODE in
        "single")
            run_single_simulation
            monitor_simulation
            ;;
        "batch")
            run_batch_simulation
            ;;
        "evaluation")
            export SLA_TARGET_RECORDS="$TARGET_RECORDS"
            export SLA_CSV_FILE="$FINAL_DATASET"
            run_single_simulation
            ;;
        *)
            error "Modo desconocido: $MODE"
            show_help
            ;;
    esac
    
    show_results_summary
}

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
fi

main