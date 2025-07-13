#!/bin/bash
# SIMULADOR ML MASIVO - VERSIÓN CORREGIDA

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Configuración
TARGET_RECORDS=${1:-6000}
BATCH_SIZE=600
BATCH_DURATION=240
PAUSE_BETWEEN_BATCHES=15

# Archivos
ML_CONTROLLER="$PROJECT_DIR/controller/sla_monitor_ml.py"
ORCHESTRATOR="$PROJECT_DIR/orchestration/orchestrator_simple.sh"
DATA_DIR="$PROJECT_DIR/data"
LOGS_DIR="$PROJECT_DIR/logs"
FINAL_DATASET="$DATA_DIR/ml_dataset_binary_${TIMESTAMP}.csv"

# Variables de control
TOTAL_BATCHES=$(( (TARGET_RECORDS + BATCH_SIZE - 1) / BATCH_SIZE ))
SUCCESSFUL_BATCHES=0
FAILED_BATCHES=0
TOTAL_RECORDS_GENERATED=0

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m'

log() {
    local level=$1
    local message="$2"
    local timestamp=$(date '+%H:%M:%S')
    
    case $level in
        "SUCCESS") echo -e "${GREEN}[✅ SUCCESS]${NC} ${timestamp} - ${message}" ;;
        "INFO")    echo -e "${BLUE}[ℹ️  INFO]${NC}    ${timestamp} - ${message}" ;;
        "WARN")    echo -e "${YELLOW}[⚠️  WARN]${NC}    ${timestamp} - ${message}" ;;
        "ERROR")   echo -e "${RED}[❌ ERROR]${NC}   ${timestamp} - ${message}" ;;
        "STEP")    echo -e "${PURPLE}[� STEP]${NC}    ${timestamp} - ${message}" ;;
        "ML")      echo -e "${WHITE}[� ML]${NC}      ${timestamp} - ${message}" ;;
    esac
    
    mkdir -p "$LOGS_DIR"
    echo "[$level] $(date) - $message" >> "$LOGS_DIR/ml_simulator.log"
}

show_banner() {
    clear
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║              SIMULADOR ML MASIVO (FIXED)                 ║"
    echo "║         Enhanced Controller + Batch Processing           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo "Dataset: SLA Binario (True/False)"
    echo "Target: $TARGET_RECORDS registros en $TOTAL_BATCHES lotes"
    echo ""
}

check_prerequisites() {
    log "STEP" "Verificando prerrequisitos..."
    
    for cmd in mn ryu-manager python3; do
        if ! command -v "$cmd" &> /dev/null; then
            log "ERROR" "Comando no encontrado: $cmd"
            exit 1
        fi
    done
    
    if [[ ! -f "$PROJECT_DIR/topology/custom_topo.py" ]]; then
        log "ERROR" "No se encuentra topology/custom_topo.py"
        exit 1
    fi
    
    if [[ ! -f "$ORCHESTRATOR" ]]; then
        log "ERROR" "No se encuentra orchestrator_simple.sh"
        exit 1
    fi
    
    if ! sudo -n true 2>/dev/null; then
        log "ERROR" "Se requieren permisos sudo"
        exit 1
    fi
    
    log "SUCCESS" "Prerrequisitos verificados"
}

setup_ml_environment() {
    log "STEP" "Configurando entorno ML..."
    
    mkdir -p "$DATA_DIR" "$LOGS_DIR"
    
    # Limpiar datos anteriores
    rm -f "$DATA_DIR"/ml_dataset_*.csv "$DATA_DIR"/metrics_*.csv 2>/dev/null || true
    
    # Configurar permisos
    sudo chown -R "$USER:$USER" "$DATA_DIR" "$LOGS_DIR" 2>/dev/null || true
    chmod -R 755 "$DATA_DIR" "$LOGS_DIR" 2>/dev/null || true
    
    # Limpiar procesos
    sudo mn -c &>/dev/null || true
    sudo pkill -f "ryu-manager" 2>/dev/null || true
    
    log "SUCCESS" "Entorno configurado"
}

create_ml_controller() {
    log "STEP" "Creando controlador ML..."
    
    if [[ -f "$ML_CONTROLLER" ]]; then
        log "INFO" "Controlador ML ya existe"
        return 0
    fi
    
    cat > "$ML_CONTROLLER" << 'MLEOF'
#!/usr/bin/env python3
import time, csv, random, math
from datetime import datetime
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
from ryu.lib import hub

class MLBinarySLAController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(MLBinarySLAController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.csv_file = '../data/metrics.csv'
        self.record_count = 0
        self.target_records = 600
        
        self.sla_thresholds = {
            'latency_ms': 50.0, 'jitter_ms': 10.0, 
            'packet_loss_percent': 1.0, 'throughput_mbps': 10.0
        }
        
        self.hosts = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        self.init_csv()
        self.monitor_thread = hub.spawn(self._ml_monitor_loop)
        self.logger.info(f"� ML Controller iniciado (target: {self.target_records})")
    
    def init_csv(self):
        try:
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'src_host', 'dst_host', 'latency_ms', 'jitter_ms',
                    'packet_loss_percent', 'throughput_mbps', 'sla_compliant',
                    'path_type', 'traffic_pattern', 'hour_of_day'
                ])
            self.logger.info(f"� CSV ML inicializado")
        except Exception as e:
            self.logger.error(f"❌ Error CSV: {e}")
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info(f"� Switch {datapath.id} conectado")
    
    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        if eth.ethertype == 0x88cc: return
        
        dst, src, dpid = eth.dst, eth.src, datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        
        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]
        
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions)
        
        data = None if msg.buffer_id == ofproto.OFP_NO_BUFFER else msg.data
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
    
    def _ml_monitor_loop(self):
        while self.record_count < self.target_records:
            self._collect_ml_metrics()
            if self.record_count % 50 == 0 and self.record_count > 0:
                self.logger.info(f"� ML Progress: {self.record_count}/{self.target_records}")
            hub.sleep(2)
        self.logger.info(f"� ML Target alcanzado: {self.record_count}")
    
    def _collect_ml_metrics(self):
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            hour = datetime.now().hour
            
            if hour in [7,8,9]: pattern = 'morning_rush'
            elif hour in [17,18,19]: pattern = 'evening_rush' 
            elif hour in [22,23,0,1,2,3,4,5]: pattern = 'night_low'
            else: pattern = 'normal'
            
            for src in self.hosts:
                for dst in self.hosts:
                    if src != dst:
                        path_type = self._get_path_type(src, dst)
                        metrics = self._generate_metrics(path_type, pattern)
                        
                        sla_compliant = (
                            metrics['latency'] <= self.sla_thresholds['latency_ms'] and
                            metrics['jitter'] <= self.sla_thresholds['jitter_ms'] and
                            metrics['packet_loss'] <= self.sla_thresholds['packet_loss_percent'] and
                            metrics['throughput'] >= self.sla_thresholds['throughput_mbps']
                        )
                        
                        with open(self.csv_file, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                timestamp, src, dst,
                                round(metrics['latency'], 3), round(metrics['jitter'], 3),
                                round(metrics['packet_loss'], 3), round(metrics['throughput'], 3),
                                sla_compliant, path_type, pattern, hour
                            ])
                        
                        self.record_count += 1
                        if self.record_count >= self.target_records: return
        except Exception as e:
            self.logger.error(f"❌ Error métricas: {e}")
    
    def _get_path_type(self, src, dst):
        fast_pairs = [('h1','h3'),('h1','h4'),('h2','h3'),('h2','h4'),('h3','h1'),('h3','h2'),('h4','h1'),('h4','h2')]
        slow_pairs = [('h1','h5'),('h1','h6'),('h2','h5'),('h2','h6'),('h5','h1'),('h5','h2'),('h6','h1'),('h6','h2')]
        limited_pairs = [('h3','h5'),('h3','h6'),('h4','h5'),('h4','h6'),('h5','h3'),('h5','h4'),('h6','h3'),('h6','h4')]
        intra_pairs = [('h1','h2'),('h2','h1'),('h3','h4'),('h4','h3'),('h5','h6'),('h6','h5')]
        
        if (src,dst) in fast_pairs: return 'fast'
        elif (src,dst) in slow_pairs: return 'slow'
        elif (src,dst) in limited_pairs: return 'limited'
        elif (src,dst) in intra_pairs: return 'intra'
        else: return 'unknown'
    
    def _generate_metrics(self, path_type, pattern):
        bases = {
            'fast': {'lat': 15, 'thr': 80}, 'slow': {'lat': 35, 'thr': 60},
            'limited': {'lat': 25, 'thr': 35}, 'intra': {'lat': 3, 'thr': 95}
        }
        base = bases.get(path_type, {'lat': 25, 'thr': 50})
        
        mult = {'morning_rush': 1.3, 'evening_rush': 1.6, 'night_low': 0.7}.get(pattern, 1.0)
        noise = random.uniform(0.8, 1.2)
        
        latency = max(1, min(100, base['lat'] * mult * noise))
        jitter = max(0.1, min(20, latency * random.uniform(0.2, 0.4)))
        packet_loss = max(0, min(5, random.uniform(0.1, 2) * mult))
        throughput = max(5, min(100, base['thr'] / mult * random.uniform(0.9, 1.1)))
        
        return {'latency': latency, 'jitter': jitter, 'packet_loss': packet_loss, 'throughput': throughput}
MLEOF
    
    chmod +x "$ML_CONTROLLER"
    log "SUCCESS" "Controlador ML creado"
}

run_batch_simulation() {
    local batch_num=$1
    
    log "STEP" "Ejecutando lote $batch_num/$TOTAL_BATCHES..."
    
    cd "$PROJECT_DIR/orchestration"
    
    # Crear versión temporal del orquestador con controlador ML
    local temp_orchestrator="orchestrator_ml_temp.sh"
    cp orchestrator_simple.sh "$temp_orchestrator"
    sed -i 's/sla_monitor.py/sla_monitor_ml.py/g' "$temp_orchestrator"
    sed -i 's/sla_monitor_temp.py/sla_monitor_ml_temp.py/g' "$temp_orchestrator"
    
    # Ejecutar lote
    timeout $((BATCH_DURATION + 60)) ./"$temp_orchestrator" $BATCH_DURATION
    local exit_code=$?
    
    # Limpiar archivo temporal
    rm -f "$temp_orchestrator"
    
    if [[ $exit_code -eq 0 ]]; then
        SUCCESSFUL_BATCHES=$((SUCCESSFUL_BATCHES + 1))
        log "SUCCESS" "Lote $batch_num completado"
        
        # Contar registros
        local newest_csv=$(find "$DATA_DIR" -name "metrics_*.csv" -type f 2>/dev/null | sort | tail -1)
        if [[ -f "$newest_csv" ]]; then
            local records=$(tail -n +2 "$newest_csv" | wc -l)
            TOTAL_RECORDS_GENERATED=$((TOTAL_RECORDS_GENERATED + records))
            log "INFO" "Registros generados: $records"
        fi
        return 0
    else
        FAILED_BATCHES=$((FAILED_BATCHES + 1))
        log "ERROR" "Lote $batch_num falló"
        return 1
    fi
}

show_progress() {
    local current=$1
    local total=$2
    local progress=$((current * 100 / total))
    
    # Barra simple sin caracteres problemáticos
    local filled=$((progress / 5))  # 20 caracteres max
    local bar=""
    
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=filled; i<20; i++)); do bar+="░"; done
    
    echo -e "${PURPLE}[PROGRESO]${NC} ${bar} ${progress}% | Lote ${current}/${total} | Records: ${TOTAL_RECORDS_GENERATED}"
}

consolidate_dataset() {
    log "STEP" "Consolidando dataset..."
    
    # Crear header
    echo "timestamp,src_host,dst_host,latency_ms,jitter_ms,packet_loss_percent,throughput_mbps,sla_compliant,path_type,traffic_pattern,hour_of_day,batch_id" > "$FINAL_DATASET"
    
    # Consolidar archivos
    local batch_counter=1
    for csv_file in "$DATA_DIR"/metrics_*.csv; do
        if [[ -f "$csv_file" ]]; then
            tail -n +2 "$csv_file" | while IFS= read -r line; do
                echo "${line},batch_${batch_counter}" >> "$FINAL_DATASET"
            done
            batch_counter=$((batch_counter + 1))
        fi
    done
    
    TOTAL_RECORDS_GENERATED=$(tail -n +2 "$FINAL_DATASET" | wc -l)
    log "SUCCESS" "Dataset consolidado con $TOTAL_RECORDS_GENERATED registros"
}

cleanup() {
    log "STEP" "Limpiando entorno..."
    sudo mn -c &>/dev/null || true
    sudo pkill -f "ryu-manager" 2>/dev/null || true
    find "$DATA_DIR" -name "metrics_*.csv" -not -name "ml_dataset_*" -delete 2>/dev/null || true
    log "SUCCESS" "Limpieza completada"
}

show_final_summary() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                � SIMULACIÓN ML COMPLETADA                    ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    log "SUCCESS" "Simulación finalizada"
    log "INFO" "Lotes exitosos: $SUCCESSFUL_BATCHES/$TOTAL_BATCHES"
    log "INFO" "Registros totales: $TOTAL_RECORDS_GENERATED"
    
    if [[ $TOTAL_RECORDS_GENERATED -ge $TARGET_RECORDS ]]; then
        log "ML" "� TARGET ALCANZADO! Dataset listo para ML"
    else
        log "WARN" "Target no completado: $TOTAL_RECORDS_GENERATED/$TARGET_RECORDS"
    fi
    
    echo ""
    echo -e "${WHITE}� ARCHIVOS GENERADOS:${NC}"
    echo "  � Dataset: $FINAL_DATASET"
    echo "  � Log: $LOGS_DIR/ml_simulator.log"
    
    if [[ -f "$FINAL_DATASET" ]]; then
        local sla_true=$(tail -n +2 "$FINAL_DATASET" | cut -d',' -f8 | grep -c "True")
        local sla_false=$(tail -n +2 "$FINAL_DATASET" | cut -d',' -f8 | grep -c "False")
        echo ""
        echo -e "${WHITE}� DISTRIBUCIÓN SLA:${NC}"
        echo "  ✅ SLA True: $sla_true"
        echo "  ❌ SLA False: $sla_false"
        echo "  � Balance: $((sla_true * 100 / (sla_true + sla_false)))% True"
    fi
}

# Función principal
main() {
    show_banner
    
    # Configurar trap para limpieza
    trap cleanup EXIT INT TERM
    
    check_prerequisites
    setup_ml_environment
    create_ml_controller
    
    log "ML" "Iniciando $TOTAL_BATCHES lotes para $TARGET_RECORDS registros"
    
    # Ejecutar lotes
    for ((batch=1; batch<=TOTAL_BATCHES; batch++)); do
        show_progress $batch $TOTAL_BATCHES
        
        if run_batch_simulation $batch; then
            log "INFO" "Lote $batch exitoso"
        else
            log "WARN" "Lote $batch falló"
        fi
        
        # Pausa entre lotes
        if [[ $batch -lt $TOTAL_BATCHES ]]; then
            log "INFO" "Pausa de ${PAUSE_BETWEEN_BATCHES}s..."
            sleep $PAUSE_BETWEEN_BATCHES
        fi
    done
    
    # Consolidar y mostrar resumen
    consolidate_dataset
    show_final_summary
}

# Ejecutar
main
