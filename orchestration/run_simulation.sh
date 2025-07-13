#!/bin/bash
# Script de orquestación completa para simulación SDN con SLA
# Archivo: orchestration/run_simulation.sh

# Configuración global
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOPOLOGY_FILE="$PROJECT_DIR/topology/custom_topo.py"
CONTROLLER_FILE="$PROJECT_DIR/controller/sla_monitor.py"
TRAFFIC_SCRIPT="$PROJECT_DIR/traffic/traffic_gen.sh"
DATA_DIR="$PROJECT_DIR/data"
LOG_DIR="$PROJECT_DIR/logs"

# Variables de control
RYU_PID=""
MININET_PID=""
SIMULATION_DURATION=300  # 5 minutos por defecto
CLEANUP_ON_EXIT=true

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Función de logging
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${GREEN}[INFO]${NC}  ${timestamp} - ${message}" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  ${timestamp} - ${message}" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} ${timestamp} - ${message}" ;;
        "DEBUG") echo -e "${CYAN}[DEBUG]${NC} ${timestamp} - ${message}" ;;
    esac
    
    # También escribir a archivo de log
    echo "[$level] $timestamp - $message" >> "$LOG_DIR/simulation.log"
}

# Función de limpieza
cleanup() {
    log "INFO" "� Iniciando limpieza..."
    
    # Matar procesos de Ryu
    if [ ! -z "$RYU_PID" ]; then
        log "INFO" "Terminando controlador Ryu (PID: $RYU_PID)"
        kill -TERM $RYU_PID 2>/dev/null
        wait $RYU_PID 2>/dev/null
    fi
    
    # Limpiar Mininet
    log "INFO" "Limpiando Mininet..."
    sudo mn -c &>/dev/null
    
    # Matar cualquier proceso residual
    sudo pkill -f "ryu-manager" 2>/dev/null
    sudo pkill -f "custom_topo.py" 2>/dev/null
    
    log "INFO" "✅ Limpieza completada"
}

# Configurar trap para limpieza al salir
trap cleanup EXIT INT TERM

# Función para verificar prerrequisitos
check_prerequisites() {
    log "INFO" "� Verificando prerrequisitos..."
    
    local missing_deps=()
    
    # Verificar comandos necesarios
    local required_commands=("mn" "ryu-manager" "python3" "iperf3" "hping3")
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=($cmd)
        fi
    done
    
    # Verificar archivos del proyecto
    local required_files=("$TOPOLOGY_FILE" "$CONTROLLER_FILE" "$TRAFFIC_SCRIPT")
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log "ERROR" "Archivo requerido no encontrado: $file"
            missing_deps+=("$(basename $file)")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log "ERROR" "Dependencias faltantes: ${missing_deps[*]}"
        log "INFO" "Ejecuta: sudo apt install -y mininet ryu-controller iperf3 hping3"
        exit 1
    fi
    
    log "INFO" "✅ Todos los prerrequisitos verificados"
}

# Función para crear directorios necesarios
setup_directories() {
    log "INFO" "� Configurando directorios..."
    
    # Crear directorios si no existen
    mkdir -p "$DATA_DIR" "$LOG_DIR"
    
    # Limpiar logs anteriores si se solicita
    if [ "$1" = "clean" ]; then
        log "INFO" "� Limpiando datos anteriores..."
        rm -f "$DATA_DIR"/*.csv "$LOG_DIR"/*.log
    fi
    
    log "INFO" "✅ Directorios configurados"
}

# Función para iniciar el controlador Ryu
start_controller() {
    log "INFO" "� Iniciando controlador Ryu..."
    
    # Cambiar al directorio del controlador
    cd "$PROJECT_DIR/controller"
    
    # Iniciar Ryu en background
    ryu-manager sla_monitor.py --verbose > "$LOG_DIR/ryu.log" 2>&1 &
    RYU_PID=$!
    
    # Verificar que Ryu se inició correctamente
    sleep 3
    
    if kill -0 $RYU_PID 2>/dev/null; then
        log "INFO" "✅ Controlador Ryu iniciado (PID: $RYU_PID)"
        log "INFO" "� Monitoreo SLA activado"
    else
        log "ERROR" "❌ Error iniciando controlador Ryu"
        cat "$LOG_DIR/ryu.log"
        exit 1
    fi
}

# Función para iniciar la topología Mininet
start_topology() {
    log "INFO" "� Iniciando topología Mininet..."
    
    # Cambiar al directorio de topología
    cd "$PROJECT_DIR/topology"
    
    # Hacer el archivo ejecutable
    chmod +x custom_topo.py
    
    # Iniciar topología en background
    sudo python3 custom_topo.py > "$LOG_DIR/mininet.log" 2>&1 &
    MININET_PID=$!
    
    # Esperar a que la topología se estabilice
    log "INFO" "⏳ Esperando estabilización de la topología..."
    sleep 10
    
    # Verificar conectividad básica
    log "INFO" "� Verificando conectividad básica..."
    
    # Test de conectividad simple
    if sudo mn --test pingall &>/dev/null; then
        log "INFO" "✅ Topología iniciada y conectividad verificada"
    else
        log "WARN" "⚠️  Topología iniciada, verificación manual requerida"
    fi
}

# Función para ejecutar generación de tráfico
run_traffic_generation() {
    local duration=$1
    log "INFO" "� Iniciando generación de tráfico (duración: ${duration}s)..."
    
    # Cambiar al directorio de tráfico
    cd "$PROJECT_DIR/traffic"
    
    # Hacer el script ejecutable
    chmod +x traffic_gen.sh
    
    # Calcular número de ciclos basado en duración
    local cycles=$((duration / 60))  # Un ciclo por minuto
    if [ $cycles -lt 1 ]; then
        cycles=1
    fi
    
    log "INFO" "� Ejecutando $cycles ciclos de generación de tráfico..."
    
    # Ejecutar generación de tráfico
    ./traffic_gen.sh $cycles
    
    log "INFO" "✅ Generación de tráfico completada"
}

# Función para generar reporte final
generate_report() {
    log "INFO" "� Generando reporte final..."
    
    local report_file="$DATA_DIR/simulation_report.txt"
    local csv_file="$DATA_DIR/metrics.csv"
    
    # Crear reporte
    cat > "$report_file" << EOF
# REPORTE DE SIMULACIÓN SDN - $(date)
====================================

## CONFIGURACIÓN
- Duración: ${SIMULATION_DURATION}s
- Topología: 6 hosts, 3 subredes, 2 switches
- Controlador: Ryu con monitoreo SLA

## ARCHIVOS GENERADOS
- Métricas: $csv_file
- Logs Ryu: $LOG_DIR/ryu.log
- Logs Mininet: $LOG_DIR/mininet.log
- Log simulación: $LOG_DIR/simulation.log

## ESTADÍSTICAS
EOF

    # Agregar estadísticas si el CSV existe
    if [ -f "$csv_file" ]; then
        local total_records=$(tail -n +2 "$csv_file" | wc -l)
        local ok_records=$(grep ",OK$" "$csv_file" | wc -l)
        local warn_records=$(grep ",WARN$" "$csv_file" | wc -l)
        local violated_records=$(grep ",VIOLATED$" "$csv_file" | wc -l)
        
        cat >> "$report_file" << EOF
- Total de registros: $total_records
- SLA OK: $ok_records ($(( ok_records * 100 / total_records ))%)
- SLA WARN: $warn_records ($(( warn_records * 100 / total_records ))%)
- SLA VIOLATED: $violated_records ($(( violated_records * 100 / total_records ))%)

## UMBRALES SLA CONFIGURADOS
- Latencia: < 50ms
- Jitter: < 10ms
- Pérdida de paquetes: < 1%
- Throughput: > 10 Mbps
EOF
    else
        echo "- No se generaron métricas CSV" >> "$report_file"
    fi
    
    log "INFO" "� Reporte generado: $report_file"
}

# Función de monitoreo en tiempo real
monitor_simulation() {
    local duration=$1
    log "INFO" "�️  Iniciando monitoreo en tiempo real..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + duration))
    
    while [ $(date +%s) -lt $end_time ]; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local remaining=$((end_time - current_time))
        
        # Mostrar progreso
        printf "\r${BLUE}⏱️  Progreso: %d/%d segundos (quedan %d)${NC}" $elapsed $duration $remaining
        
        # Verificar si los procesos siguen corriendo
        if [ ! -z "$RYU_PID" ] && ! kill -0 $RYU_PID 2>/dev/null; then
            printf "\n"
            log "ERROR" "❌ Controlador Ryu se detuvo inesperadamente"
            break
        fi
        
        sleep 5
    done
    
    printf "\n"
    log "INFO" "✅ Monitoreo completado"
}

# Función de ayuda
show_help() {
    cat << EOF
� SIMULADOR SDN CON MONITOREO SLA
=================================

Uso: $0 [opciones]

OPCIONES:
  -d, --duration SECONDS    Duración de la simulación en segundos (default: 300)
  -c, --clean               Limpiar datos anteriores antes de iniciar
  -h, --help                Mostrar esta ayuda
  -v, --verbose             Modo verbose (más logs)
  --no-cleanup              No ejecutar limpieza al salir

EJEMPLOS:
  $0                        # Simulación de 5 minutos
  $0 -d 600                 # Simulación de 10 minutos
  $0 -c -d 180              # Simulación de 3 minutos con limpieza
  $0 --verbose              # Simulación con logs detallados

ARCHIVOS GENERADOS:
  data/metrics.csv          # Métricas de red con clasificación SLA
  data/simulation_report.txt # Reporte final de la simulación
  logs/ryu.log              # Logs del controlador Ryu
  logs/mininet.log          # Logs de Mininet
  logs/simulation.log       # Logs de orquestación

EOF
}

# Función principal
main() {
    # Banner de inicio
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════╗"
    echo "║        SIMULADOR SDN CON SLA           ║"
    echo "║      Mininet 2.3.0 + Ryu + ML         ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Parsear argumentos
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--duration)
                SIMULATION_DURATION="$2"
                shift 2
                ;;
            -c|--clean)
                CLEAN_DATA=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            --no-cleanup)
                CLEANUP_ON_EXIT=false
                shift
                ;;
            *)
                log "ERROR" "Opción desconocida: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    log "INFO" "� Iniciando simulación SDN con monitoreo SLA"
    log "INFO" "� Directorio del proyecto: $PROJECT_DIR"
    log "INFO" "⏱️  Duración configurada: ${SIMULATION_DURATION}s"
    
    # Ejecutar pasos de la simulación
    check_prerequisites
    setup_directories ${CLEAN_DATA:+clean}
    start_controller
    start_topology
    
    # Ejecutar tráfico y monitoreo en paralelo
    run_traffic_generation $SIMULATION_DURATION &
    monitor_simulation $SIMULATION_DURATION
    
    # Esperar a que termine la generación de tráfico
    wait
    
    # Generar reporte final
    generate_report
    
    log "INFO" "� Simulación completada exitosamente"
    log "INFO" "� Revisa los resultados en: $DATA_DIR/"
    
    # Preguntar si mantener la topología activa
    if [ "$CLEANUP_ON_EXIT" = true ]; then
        log "INFO" "⏳ Limpieza automática en 5 segundos... (Ctrl+C para cancelar)"
        sleep 5
    else
        log "INFO" "� Topología mantenida activa. Ejecuta 'sudo mn -c' para limpiar manualmente."
        log "INFO" "� Controlador Ryu sigue corriendo en PID: $RYU_PID"
        trap - EXIT INT TERM  # Deshabilitar limpieza automática
    fi
}

# Ejecutar función principal
main "$@"
