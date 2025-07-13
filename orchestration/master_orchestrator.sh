#!/bin/bash
# SIMULADOR HÍBRIDO EN LOTES - Enhanced + Batch para ML Masivo
# Archivo: orchestration/ml_mass_simulator.sh

# =====================================================
# CONFIGURACIÓN GLOBAL
# =====================================================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Configuración por defecto
TARGET_RECORDS=6000
BATCH_SIZE=600          # Registros por lote (30 pares × 20 ciclos)
BATCH_DURATION=240      # 4 minutos por lote (30 pares × 2s × 20 ciclos)
MONITORING_INTERVAL=2   # Monitoreo cada 2 segundos
PAUSE_BETWEEN_BATCHES=15 # Pausa entre lotes

# Archivos y directorios
ML_CONTROLLER="$PROJECT_DIR/controller/sla_monitor_ml.py"
ORCHESTRATOR="$PROJECT_DIR/orchestration/orchestrator_simple.sh"
DATA_DIR="$PROJECT_DIR/data"
LOGS_DIR="$PROJECT_DIR/logs"
RESULTS_DIR="$PROJECT_DIR/ml_results_$TIMESTAMP"
FINAL_DATASET="$DATA_DIR/ml_dataset_binary_${TIMESTAMP}.csv"

# Variables de control
TOTAL_BATCHES=0
SUCCESSFUL_BATCHES=0
FAILED_BATCHES=0
TOTAL_RECORDS_GENERATED=0

# =====================================================
# COLORES Y LOGGING
# =====================================================
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
        "DEBUG")   echo -e "${CYAN}[� DEBUG]${NC}   ${timestamp} - ${message}" ;;
        "STEP")    echo -e "${PURPLE}[� STEP]${NC}    ${timestamp} - ${message}" ;;
        "ML")      echo -e "${WHITE}[� ML]${NC}      ${timestamp} - ${message}" ;;
    esac
    
    # Log a archivo
    mkdir -p "$LOGS_DIR"
    echo "[$level] $(date '+%Y-%m-%d %H:%M:%S') - $message" >> "$LOGS_DIR/ml_simulator.log"
}

# =====================================================
# FUNCIONES DE BANNER Y AYUDA
# =====================================================
show_banner() {
    clear
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║              SIMULADOR ML MASIVO HÍBRIDO                 ║
║         Enhanced Controller + Batch Processing           ║
║                                                           ║
║  � ML Ready  |  � SLA Binario  |  � Alto Volumen      ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo -e "${WHITE}Dataset: SLA Binario (cumple/no_cumple)${NC}"
    echo -e "${WHITE}Features: 30 pares × latencia/jitter/loss/throughput + contexto${NC}"
    echo -e "${WHITE}Objetivo: Máximo rendimiento para modelos ML${NC}"
    echo ""
}

show_help() {
    cat << EOF
${WHITE}SIMULADOR ML MASIVO HÍBRIDO${NC}
==========================

${YELLOW}DESCRIPCIÓN:${NC}
  Genera datasets masivos para ML usando controlador enhanced
  con SLA binario y simulación en lotes automatizada.

${YELLOW}CARACTERÍSTICAS:${NC}
  • SLA binario: True/False (cumple/no cumple)
  • 30 pares bidireccionales de hosts
  • Frecuencia: cada 2 segundos
  • Features ML: path_type, traffic_pattern, congestion_level
  • Balanceado automático de clases
  • Limpieza automática de procesos

${YELLOW}USO:${NC}
  $0 [OPCIONES]

${YELLOW}OPCIONES:${NC}
  -t, --target NUM      Registros objetivo (default: 6000)
  -b, --batch-size NUM  Registros por lote (default: 600)
  -d, --duration SEC    Duración por lote (default: 240s)
  -p, --pause SEC       Pausa entre lotes (default: 15s)
  -c, --clean           Limpiar datos anteriores
  -v, --verbose         Modo debug detallado
  -h, --help            Mostrar esta ayuda

${YELLOW}EJEMPLOS:${NC}
  $0                    # 6000 registros, 10 lotes
  $0 -t 10000          # 10000 registros
  $0 -b 1200 -d 480    # Lotes grandes (8 min cada uno)
  $0 -t 3000 -c        # Dataset pequeño con limpieza

${YELLOW}SALIDA:${NC}
  � ml_dataset_binary_TIMESTAMP.csv    (dataset consolidado)
  � ml_analysis_TIMESTAMP.txt          (análisis estadístico)
  � ml_balance_report.txt              (balance de clases)
  � logs/ml_simulator.log              (log detallado)

EOF
}

# =====================================================
# FUNCIONES DE VERIFICACIÓN
# =====================================================
check_prerequisites() {
    log "STEP" "Verificando prerrequisitos para ML..."
    
    local missing=()
    
    # Verificar comandos
    for cmd in mn ryu-manager python3; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done
    
    # Verificar archivos base
    if [[ ! -f "$PROJECT_DIR/topology/custom_topo.py" ]]; then
        missing+=("topology/custom_topo.py")
    fi
    
    if [[ ! -f "$ORCHESTRATOR" ]]; then
        missing+=("orchestrator_simple.sh")
    fi
    
    # Verificar permisos sudo
    if ! sudo -n true 2>/dev/null; then
        log "ERROR" "Se requieren permisos sudo para Mininet"
        exit 1
    fi
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log "ERROR" "Componentes faltantes: ${missing[*]}"
        exit 1
    fi
    
    log "SUCCESS" "Prerrequisitos verificados"
}

calculate_batches() {
    TOTAL_BATCHES=$(( (TARGET_RECORDS + BATCH_SIZE - 1) / BATCH_SIZE ))  # Redondeo hacia arriba
    
    log "ML" "Configuración calculada:"
    log "INFO" "  Target: $TARGET_RECORDS registros"
    log "INFO" "  Lotes: $TOTAL_BATCHES × $BATCH_SIZE registros"
    log "INFO" "  Duración por lote: $BATCH_DURATION segundos"
    log "INFO" "  Tiempo estimado: $(( (BATCH_DURATION + PAUSE_BETWEEN_BATCHES) * TOTAL_BATCHES / 60 )) minutos"
}

# =====================================================
# FUNCIONES DE CONFIGURACIÓN
# =====================================================
setup_ml_environment() {
    log "STEP" "Configurando entorno ML..."
    
    # Crear directorios
    mkdir -p "$DATA_DIR" "$LOGS_DIR" "$RESULTS_DIR"
    
    # Limpiar datos anteriores si se solicita
    if [[ "$CLEAN_DATA" == "true" ]]; then
        log "INFO" "Limpiando datos anteriores..."
        rm -f "$DATA_DIR"/ml_dataset_*.csv "$DATA_DIR"/metrics_*.csv 2>/dev/null || true
        rm -rf "$PROJECT_DIR"/ml_results_* "$PROJECT_DIR"/results_* 2>/dev/null || true
    fi
    
    # Configurar permisos
    sudo chown -R "$USER:$USER" "$DATA_DIR" "$LOGS_DIR" "$RESULTS_DIR" 2>/dev/null || true
    chmod -R 755 "$DATA_DIR" "$LOGS_DIR" "$RESULTS_DIR" 2>/dev/null || true
    
    # Limpiar procesos anteriores
    sudo mn -c &>/dev/null || true
    sudo pkill -f "ryu-manager" 2>/dev/null || true
    sudo pkill -f "sla_monitor" 2>/dev/null || true
    
    log "SUCCESS" "Entorno ML configurado"
}

setup_ml_controller() {
    log "STEP" "Configurando controlador ML..."
    
    # Verificar si existe el controlador ML
    if [[ ! -f "$ML_CONTROLLER" ]]; then
        log "WARN" "Controlador ML no encontrado, creando..."
        create_ml_controller
    fi
    
    # Modificar orquestador para usar controlador ML
    local temp_orchestrator="${ORCHESTRATOR}.ml_temp"
    cp "$ORCHESTRATOR" "$temp_orchestrator"
    
    # Reemplazar controlador en el orquestador temporal
    sed -i 's/sla_monitor.py/sla_monitor_ml.py/g' "$temp_orchestrator"
    sed -i 's/sla_monitor_temp.py/sla_monitor_ml_temp.py/g' "$temp_orchestrator"
    
    log "SUCCESS" "Controlador ML configurado"
}

create_ml_controller() {
    log "INFO" "Creando controlador ML básico..."
    
    cat > "$ML_CONTROLLER" << 'MLCONTROLLER'
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
        self.target_records = 600  # Por lote
        
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
            self.logger.info(f"� CSV ML inicializado: {self.csv_file}")
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
            if self.record_count % 50 == 0:
                self.logger.info(f"� ML: {self.record_count}/{self.target_records}")
            hub.sleep(2)
        self.logger.info(f"� ML Target alcanzado: {self.record_count}")
    
    def _collect_ml_metrics(self):
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            hour = datetime.now().hour
            
            # Patrones de tráfico
            if hour in [7,8,9]: pattern = 'morning_rush'
            elif hour in [17,18,19]: pattern = 'evening_rush' 
            elif hour in [22,23,0,1,2,3,4,5]: pattern = 'night_low'
            else: pattern = 'normal'
            
            for src in self.hosts:
                for dst in self.hosts:
                    if src != dst:
                        # Generar métricas según tipo de ruta
                        path_type = self._get_path_type(src, dst)
                        metrics = self._generate_metrics(path_type, pattern)
                        
                        # SLA binario
                        sla_compliant = (
                            metrics['latency'] <= self.sla_thresholds['latency_ms'] and
                            metrics['jitter'] <= self.sla_thresholds['jitter_ms'] and
                            metrics['packet_loss'] <= self.sla_thresholds['packet_loss_percent'] and
                            metrics['throughput'] >= self.sla_thresholds['throughput_mbps']
                        )
                        
                        # Escribir CSV
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
MLCONTROLLER
    
    chmod +x "$ML_CONTROLLER"
    log "SUCCESS" "Controlador ML básico creado"
}

# =====================================================
# FUNCIONES DE EJECUCIÓN DE LOTES
# =====================================================
run_batch_simulation() {
    local batch_num=$1
    local batch_start_time=$(date +%s)
    
    log "STEP" "Ejecutando lote ML $batch_num/$TOTAL_BATCHES..."
    
    cd "$PROJECT_DIR/orchestration"
    
    # Usar orquestador temporal con controlador ML
    local temp_orchestrator="${ORCHESTRATOR}.ml_temp"
    
    # Ejecutar con timeout de seguridad
    timeout $((BATCH_DURATION + 120)) ./"$(basename "$temp_orchestrator")" $BATCH_DURATION
    local exit_code=$?
    
    local batch_end_time=$(date +%s)
    local batch_elapsed=$((batch_end_time - batch_start_time))
    
    if [[ $exit_code -eq 0 ]]; then
        SUCCESSFUL_BATCHES=$((SUCCESSFUL_BATCHES + 1))
        log "SUCCESS" "Lote $batch_num completado en ${batch_elapsed}s"
        
        # Contar registros generados
        local csv_file=$(find "$DATA_DIR" -name "metrics_*.csv" -newer /tmp/batch_start_marker 2>/dev/null | head -1)
        if [[ -f "$csv_file" ]]; then
            local records=$(tail -n +2 "$csv_file" | wc -l)
            TOTAL_RECORDS_GENERATED=$((TOTAL_RECORDS_GENERATED + records))
            log "DEBUG" "  � Registros generados: $records"
        fi
        
        return 0
    else
        FAILED_BATCHES=$((FAILED_BATCHES + 1))
        if [[ $exit_code -eq 124 ]]; then
            log "WARN" "Lote $batch_num terminado por timeout (${batch_elapsed}s)"
        else
            log "ERROR" "Lote $batch_num falló (código: $exit_code, tiempo: ${batch_elapsed}s)"
        fi
        return 1
    fi
}

show_batch_progress() {
    local current=$1
    local total=$2
    local start_time=$3
    
    local elapsed=$(($(date +%s) - start_time))
    local progress=$((current * 100 / total))
    
    # Calcular ETA
    local eta=0
    if [[ $current -gt 0 ]]; then
        local avg_time=$((elapsed / current))
        local remaining=$((total - current))
        eta=$((remaining * avg_time))
    fi
    
    # Barra de progreso
    local bar_length=40
    local filled=$((progress * bar_length / 100))
    local bar=""
    
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=filled; i<bar_length; i++)); do bar+="░"; done
    
    printf "\r${PURPLE}[� PROGRESO LOTES]${NC} ${bar} ${progress}%% | ${current}/${total} | ETA: $((eta/60))m${((eta%60))}s | Records: ${TOTAL_RECORDS_GENERATED}"
}

# =====================================================
# FUNCIONES DE CONSOLIDACIÓN Y ANÁLISIS
# =====================================================
consolidate_ml_dataset() {
    log "STEP" "Consolidando dataset ML..."
    
    # Buscar todos los archivos CSV de métricas
    local csv_files=($(find "$DATA_DIR" -name "metrics_*.csv" -type f 2>/dev/null))
    
    if [[ ${#csv_files[@]} -eq 0 ]]; then
        log "ERROR" "No se encontraron archivos CSV para consolidar"
        return 1
    fi
    
    # Crear header del dataset final
    echo "timestamp,src_host,dst_host,latency_ms,jitter_ms,packet_loss_percent,throughput_mbps,sla_compliant,path_type,traffic_pattern,hour_of_day,batch_id" > "$FINAL_DATASET"
    
    local batch_counter=1
    local total_records=0
    
    for csv_file in "${csv_files[@]}"; do
        if [[ -f "$csv_file" ]]; then
            log "DEBUG" "Procesando: $(basename "$csv_file")"
            
            # Agregar datos con batch_id
            tail -n +2 "$csv_file" | while IFS= read -r line; do
                echo "${line},batch_${batch_counter}" >> "$FINAL_DATASET"
            done
            
            local records=$(tail -n +2 "$csv_file" | wc -l)
            total_records=$((total_records + records))
            batch_counter=$((batch_counter + 1))
        fi
    done
    
    TOTAL_RECORDS_GENERATED=$total_records
    
    log "SUCCESS" "Dataset consolidado: $FINAL_DATASET"
    log "ML" "Total de registros: $TOTAL_RECORDS_GENERATED"
    
    return 0
}

analyze_ml_dataset() {
    log "STEP" "Analizando dataset ML..."
    
    if [[ ! -f "$FINAL_DATASET" ]]; then
        log "ERROR" "Dataset no encontrado para análisis"
        return 1
    fi
    
    local analysis_file="$RESULTS_DIR/ml_analysis_${TIMESTAMP}.txt"
    local balance_file="$RESULTS_DIR/ml_balance_report.txt"
    
    # Análisis básico
    local total_records=$(tail -n +2 "$FINAL_DATASET" | wc -l)
    local sla_true=$(tail -n +2 "$FINAL_DATASET" | cut -d',' -f8 | grep -c "True")
    local sla_false=$(tail -n +2 "$FINAL_DATASET" | cut -d',' -f8 | grep -c "False")
    
    # Calcular porcentajes
    local sla_true_pct=0
    local sla_false_pct=0
    if [[ $total_records -gt 0 ]]; then
        sla_true_pct=$((sla_true * 100 / total_records))
        sla_false_pct=$((sla_false * 100 / total_records))
    fi
    
    # Generar reporte de análisis
    cat > "$analysis_file" << ANALYSIS
# ANÁLISIS DATASET ML - SLA BINARIO
==================================
Fecha: $(date)
Dataset: $FINAL_DATASET

## ESTADÍSTICAS GENERALES
Total de registros: $total_records
Target original: $TARGET_RECORDS
Lotes ejecutados: $TOTAL_BATCHES
Lotes exitosos: $SUCCESSFUL_BATCHES
Lotes fallidos: $FAILED_BATCHES

## DISTRIBUCIÓN SLA BINARIA
SLA Cumple (True):    $sla_true registros (${sla_true_pct}%)
SLA No Cumple (False): $sla_false registros (${sla_false_pct}%)

## BALANCE DE CLASES
$(if [[ $sla_true_pct -ge 40 && $sla_true_pct -le 60 ]]; then echo "✅ BIEN BALANCEADO"; elif [[ $sla_true_pct -ge 30 && $sla_true_pct -le 70 ]]; then echo "⚠️  MODERADAMENTE BALANCEADO"; else echo "❌ DESBALANCEADO - Considerar ajustar umbrales"; fi)
Ratio True/False: $(echo "scale=2; $sla_true / $sla_false" | bc 2>/dev/null || echo "N/A")

## DISTRIBUCIÓN POR TIPO DE RUTA
$(tail -n +2 "$FINAL_DATASET" | cut -d',' -f9 | sort | uniq -c | sed 's/^ *//')

## DISTRIBUCIÓN POR PATRÓN DE TRÁFICO
$(tail -n +2 "$FINAL_DATASET" | cut -d',' -f10 | sort | uniq -c | sed 's/^ *//')

## ESTADÍSTICAS MÉTRICAS
Latencia promedio: $(tail -n +2 "$FINAL_DATASET" | cut -d',' -f4 | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f ms", sum/count; else print "N/A"}')
Throughput promedio: $(tail -n +2 "$FINAL_DATASET" | cut -d',' -f7 | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f Mbps", sum/count; else print "N/A"}')

ANALYSIS
    
    # Generar reporte de balance específico
    cat > "$balance_file" << BALANCE
# REPORTE DE BALANCE PARA ML
===========================

## DISTRIBUCIÓN OBJETIVO PARA ML
Para modelos de clasificación binaria, se recomienda:
- Clases balanceadas: 40-60% cada una
- Mínimo 1000 ejemplos por clase
- Diversidad en features

## ESTADO ACTUAL
✅ Total registros: $total_records
$(if [[ $sla_true -ge 1000 && $sla_false -ge 1000 ]]; then echo "✅"; else echo "❌"; fi) Ejemplos por clase: True($sla_true), False($sla_false)
$(if [[ $sla_true_pct -ge 30 && $sla_true_pct -le 70 ]]; then echo "✅"; else echo "❌"; fi) Balance: ${sla_true_pct}% True, ${sla_false_pct}% False

## RECOMENDACIONES PARA ML
$(if [[ $total_records -ge $TARGET_RECORDS && $sla_true_pct -ge 30 && $sla_true_pct -le 70 ]]; then
echo "� DATASET LISTO PARA ML
- Tamaño adecuado: $total_records registros
- Balance aceptable: ${sla_true_pct}%/${sla_false_pct}%
- Proceder con train/test split (80/20)
- Probar algoritmos: RandomForest, SVM, XGBoost"
else
echo "⚠️  DATASET NECESITA AJUSTES"
[[ $total_records -lt $TARGET_RECORDS ]] && echo "- Incrementar registros (actual: $total_records, target: $TARGET_RECORDS)"
[[ $sla_true_pct -lt 30 || $sla_true_pct -gt 70 ]] && echo "- Ajustar umbrales SLA para mejor balance"
echo "- Considerar técnicas de balanceo: SMOTE, undersampling"
fi)

## COMANDOS PYTHON PARA ML
# Cargar dataset
import pandas as pd
df = pd.read_csv('$FINAL_DATASET')
print(f"Shape: {df.shape}")
print("SLA distribution:")
print(df['sla_compliant'].value_counts())

# Preparar features
features = ['latency_ms', 'jitter_ms', 'packet_loss_percent', 'throughput_mbps', 'hour_of_day']
X = df[features]
y = df['sla_compliant']

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

BALANCE
    
    log "SUCCESS" "Análisis completado:"
    log "INFO" "  � Análisis: $analysis_file"
    log "INFO" "  ⚖️  Balance: $balance_file"
    log "ML" "Dataset con $total_records registros (${sla_true_pct}% True, ${sla_false_pct}% False)"
}

# =====================================================
# FUNCIONES DE LIMPIEZA
# =====================================================
cleanup_ml_environment() {
    log "STEP" "Limpiando entorno ML..."
    
    # Limpiar procesos
    sudo mn -c &>/dev/null || true
    sudo pkill -f "ryu-manager" 2>/dev/null || true
    sudo pkill -f "sla_monitor" 2>/dev/null || true
    
    # Restaurar orquestador original
    local temp_orchestrator="${ORCHESTRATOR}.ml_temp"
    [[ -f "$temp_orchestrator" ]] && rm -f "$temp_orchestrator"
    
    # Limpiar archivos temporales
    find "$DATA_DIR" -name "metrics_*.csv" -not -name "ml_dataset_*" -delete 2>/dev/null || true
    find "$PROJECT_DIR" -name "results_*" -maxdepth 1 -type d -exec rm -rf {} \; 2>/dev/null || true
    
    # Limpiar controladores temporales
    find "$PROJECT_DIR/controller" -name "*_temp.py" -delete 2>/dev/null || true
    
    log "SUCCESS" "Limpieza ML completada"
}

# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================
main() {
    local start_time=$(date +%s)
    
    show_banner
    
    log "ML" "� Iniciando Simulador ML Masivo Híbrido"
    log "INFO" "� Target: $TARGET_RECORDS registros SLA binario"
    log "INFO" "� Estrategia: Enhanced Controller + Batch Processing"
    echo ""
    
    # Configurar trap para limpieza
    trap cleanup_ml_environment EXIT INT TERM
    
    # Crear marcador para archivos nuevos
    touch /tmp/batch_start_marker
    
    # Fases de ejecución
    check_prerequisites
    calculate_batches
    setup_ml_environment
    setup_ml_controller
    
    echo ""
    log "STEP" "Iniciando simulación en lotes..."
    
    # Ejecutar lotes
    for ((batch=1; batch<=TOTAL_BATCHES; batch++)); do
        show_batch_progress $batch $TOTAL_BATCHES $start_time
        echo ""
        
        if run_batch_simulation $batch; then
            log "DEBUG" "Lote $batch exitoso"
        else
            log "WARN" "Lote $batch falló, continuando..."
        fi
        
        # Pausa entre lotes (excepto el último)
        if [[ $batch -lt $TOTAL_BATCHES ]]; then
            log "DEBUG" "Pausa de ${PAUSE_BETWEEN_BATCHES}s entre lotes..."
            sleep $PAUSE_BETWEEN_BATCHES
        fi
    done
    
    printf "\n"
    
    # Post-procesamiento
    consolidate_ml_dataset
    analyze_ml_dataset
    
    # Mostrar resumen final
    show_final_summary $start_time
}

show_final_summary() {
    local start_time=$1
    local total_time=$(( $(date +%s) - start_time ))
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                � SIMULACIÓN ML MASIVA COMPLETADA             ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    log "SUCCESS" "� Simulación ML finalizada exitosamente"
    log "INFO" "⏱️  Tiempo total: $((total_time / 60))m $((total_time % 60))s"
    log "INFO" "✅ Lotes exitosos: $SUCCESSFUL_BATCHES/$TOTAL_BATCHES"
    log "INFO" "� Registros generados: $TOTAL_RECORDS_GENERATED"
    
    if [[ $TOTAL_RECORDS_GENERATED -ge $TARGET_RECORDS ]]; then
        log "ML" "� ¡TARGET ALCANZADO! Dataset listo para ML"
    else
        log "WARN" "⚠️  Target no completado. Generados: $TOTAL_RECORDS_GENERATED/$TARGET_RECORDS"
    fi
    
    echo ""
    echo -e "${WHITE}� ARCHIVOS GENERADOS:${NC}"
    echo "  � Dataset principal: $FINAL_DATASET"
    echo "  � Análisis ML: $RESULTS_DIR/ml_analysis_${TIMESTAMP}.txt"
    echo "  ⚖️  Balance report: $RESULTS_DIR/ml_balance_report.txt"
    echo "  � Log completo: $LOGS_DIR/ml_simulator.log"
    
    echo ""
    echo -e "${WHITE}� SIGUIENTE PASO - ENTRENAMIENTO:${NC}"
    echo "  1. import pandas as pd"
    echo "  2. df = pd.read_csv('$FINAL_DATASET')"
    echo "  3. print(df['sla_compliant'].value_counts())"
    echo "  4. # Entrenar modelo de clasificación binaria"
    
    echo ""
    log "ML" "� Dataset ML con SLA binario listo para algoritmos de machine learning"
}

# =====================================================
# PARSEO DE ARGUMENTOS Y EJECUCIÓN
# =====================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET_RECORDS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -d|--duration)
            BATCH_DURATION="$2"
            shift 2
            ;;
        -p|--pause)
            PAUSE_BETWEEN_BATCHES="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_DATA=true
            shift
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log "ERROR" "Opción desconocida: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validaciones
[[ $TARGET_RECORDS -lt 100 ]] && { log "ERROR" "Target mínimo: 100"; exit 1; }
[[ $BATCH_SIZE -lt 50 ]] && { log "ERROR" "Batch size mínimo: 50"; exit 1; }
[[ $BATCH_DURATION -lt 60 ]] && { log "ERROR" "Duración mínima: 60s"; exit 1; }

# Ejecutar
main