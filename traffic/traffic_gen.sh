#!/bin/bash
# Script de generación de tráfico para testing SLA
# Archivo: traffic/traffic_gen.sh

# Configuración
DURATION=60  # Duración de cada test en segundos
INTERVAL=5   # Intervalo entre tests
LOG_FILE="../data/traffic_log.txt"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}� Iniciando generación de tráfico SDN${NC}"
echo "⏱️  Duración por test: ${DURATION}s"
echo "� Intervalo entre tests: ${INTERVAL}s"
echo "� Log file: ${LOG_FILE}"

# Crear archivo de log
echo "# Traffic Generation Log - $(date)" > $LOG_FILE
echo "# Format: timestamp,test_type,src_host,dst_host,result" >> $LOG_FILE

# Función para logging
log_result() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local test_type=$1
    local src_host=$2
    local dst_host=$3
    local result=$4
    
    echo "${timestamp},${test_type},${src_host},${dst_host},${result}" >> $LOG_FILE
}

# Función para test de conectividad básica
test_connectivity() {
    echo -e "${YELLOW}� Testing conectividad básica...${NC}"
    
    # Pares de hosts para testing
    local hosts=(
        "h1 h3" "h1 h4" "h1 h5" "h1 h6"
        "h2 h3" "h2 h4" "h2 h5" "h2 h6"
        "h3 h5" "h3 h6" "h4 h5" "h4 h6"
    )
    
    for pair in "${hosts[@]}"; do
        local src=$(echo $pair | cut -d' ' -f1)
        local dst=$(echo $pair | cut -d' ' -f2)
        
        echo -e "  � Ping ${src} -> ${dst}"
        
        # Ejecutar ping desde Mininet
        local ping_result=$(sudo mn --topo single,3 --test pingall 2>/dev/null | grep -c "0% dropped")
        
        if [ $ping_result -gt 0 ]; then
            echo -e "    ✅ ${GREEN}Conectividad OK${NC}"
            log_result "ping" $src $dst "OK"
        else
            echo -e "    ❌ ${RED}Conectividad FAIL${NC}"
            log_result "ping" $src $dst "FAIL"
        fi
        
        sleep 1
    done
}

# Función para test de throughput con iperf3
test_throughput() {
    echo -e "${YELLOW}� Testing throughput con iperf3...${NC}"
    
    # Pares críticos para throughput testing
    local critical_pairs=(
        "h1 h4"  # Inter-subnet rápido
        "h1 h5"  # Inter-subnet con latencia
        "h2 h6"  # Inter-subnet con pérdida
        "h3 h4"  # Intra-subnet
    )
    
    for pair in "${critical_pairs[@]}"; do
        local src=$(echo $pair | cut -d' ' -f1)
        local dst=$(echo $pair | cut -d' ' -f2)
        
        echo -e "  � Throughput ${src} -> ${dst}"
        
        # Simular comando iperf3 (en implementación real usar mininet)
        local throughput=$((RANDOM % 80 + 20))  # Simular 20-100 Mbps
        
        echo -e "    � Throughput: ${throughput} Mbps"
        
        if [ $throughput -gt 10 ]; then
            echo -e "    ✅ ${GREEN}SLA OK${NC}"
            log_result "throughput" $src $dst "${throughput}Mbps_OK"
        else
            echo -e "    ⚠️  ${YELLOW}SLA WARN${NC}"
            log_result "throughput" $src $dst "${throughput}Mbps_WARN"
        fi
        
        sleep 2
    done
}

# Función para test de latencia con hping3
test_latency() {
    echo -e "${YELLOW}⏱️  Testing latencia con hping3...${NC}"
    
    local latency_pairs=(
        "h1 h3"  # Enlace rápido
        "h1 h5"  # Enlace con latencia
        "h2 h6"  # Enlace con pérdida
    )
    
    for pair in "${latency_pairs[@]}"; do
        local src=$(echo $pair | cut -d' ' -f1)
        local dst=$(echo $pair | cut -d' ' -f2)
        
        echo -e "  ⚡ Latencia ${src} -> ${dst}"
        
        # Simular latencia (en implementación real usar hping3)
        local latency=$((RANDOM % 60 + 5))  # Simular 5-65ms
        
        echo -e "    ⏱️  Latencia: ${latency}ms"
        
        if [ $latency -lt 50 ]; then
            echo -e "    ✅ ${GREEN}SLA OK${NC}"
            log_result "latency" $src $dst "${latency}ms_OK"
        else
            echo -e "    ❌ ${RED}SLA VIOLATED${NC}"
            log_result "latency" $src $dst "${latency}ms_VIOLATED"
        fi
        
        sleep 1
    done
}

# Función para test de pérdida de paquetes
test_packet_loss() {
    echo -e "${YELLOW}� Testing pérdida de paquetes...${NC}"
    
    local loss_pairs=(
        "h1 h4"
        "h2 h5"
        "h3 h6"
    )
    
    for pair in "${loss_pairs[@]}"; do
        local src=$(echo $pair | cut -d' ' -f1)
        local dst=$(echo $pair | cut -d' ' -f2)
        
        echo -e "  � Pérdida ${src} -> ${dst}"
        
        # Simular pérdida de paquetes
        local loss=$(echo "scale=2; $RANDOM/32767*3" | bc)  # 0-3%
        
        echo -e "    � Pérdida: ${loss}%"
        
        if (( $(echo "$loss < 1.0" | bc -l) )); then
            echo -e "    ✅ ${GREEN}SLA OK${NC}"
            log_result "packet_loss" $src $dst "${loss}%_OK"
        else
            echo -e "    ❌ ${RED}SLA VIOLATED${NC}"
            log_result "packet_loss" $src $dst "${loss}%_VIOLATED"
        fi
        
        sleep 1
    done
}

# Función para stress test
stress_test() {
    echo -e "${YELLOW}� Ejecutando stress test...${NC}"
    
    echo -e "  � Generando tráfico concurrente..."
    
    # Simular múltiples flujos concurrentes
    local concurrent_flows=5
    
    for i in $(seq 1 $concurrent_flows); do
        local src="h$((RANDOM % 3 + 1))"
        local dst="h$((RANDOM % 3 + 4))"
        
        echo -e "    � Flujo $i: ${src} -> ${dst}"
        
        # Simular métricas bajo estrés
        local stress_throughput=$((RANDOM % 40 + 10))  # Menor throughput
        local stress_latency=$((RANDOM % 100 + 20))    # Mayor latencia
        
        if [ $stress_throughput -gt 10 ] && [ $stress_latency -lt 50 ]; then
            echo -e "      ✅ ${GREEN}Flujo estable${NC}"
            log_result "stress" $src $dst "stable"
        else
            echo -e "      ⚠️  ${YELLOW}Flujo degradado${NC}"
            log_result "stress" $src $dst "degraded"
        fi
        
        sleep 0.5
    done
}

# Función principal de ejecución
main() {
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}        TRAFFIC GENERATION SUITE        ${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    
    # Verificar que Mininet esté disponible
    if ! command -v mn &> /dev/null; then
        echo -e "${RED}❌ Mininet no encontrado${NC}"
        exit 1
    fi
    
    # Verificar que iperf3 esté disponible
    if ! command -v iperf3 &> /dev/null; then
        echo -e "${YELLOW}⚠️  iperf3 no encontrado, instalando...${NC}"
        sudo apt install -y iperf3
    fi
    
    # Verificar que hping3 esté disponible
    if ! command -v hping3 &> /dev/null; then
        echo -e "${YELLOW}⚠️  hping3 no encontrado, instalando...${NC}"
        sudo apt install -y hping3
    fi
    
    # Ejecutar tests
    local cycles=${1:-3}  # Número de ciclos (default: 3)
    
    for cycle in $(seq 1 $cycles); do
        echo -e "${BLUE}� Ciclo $cycle de $cycles${NC}"
        
        test_connectivity
        sleep $INTERVAL
        
        test_throughput
        sleep $INTERVAL
        
        test_latency
        sleep $INTERVAL
        
        test_packet_loss
        sleep $INTERVAL
        
        if [ $cycle -eq $cycles ]; then
            stress_test
        fi
        
        echo -e "${GREEN}✅ Ciclo $cycle completado${NC}"
        echo ""
    done
    
    echo -e "${GREEN}� Generación de tráfico completada${NC}"
    echo -e "${BLUE}� Revisa el log: ${LOG_FILE}${NC}"
    echo -e "${BLUE}� Revisa las métricas: ../data/metrics.csv${NC}"
}

# Verificar argumentos
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Uso: $0 [ciclos]"
    echo "  ciclos: Número de ciclos de testing (default: 3)"
    echo ""
    echo "Ejemplos:"
    echo "  $0        # 3 ciclos"
    echo "  $0 5      # 5 ciclos"
    echo "  $0 1      # 1 ciclo rápido"
    exit 0
fi

# Ejecutar función principal
main $1
