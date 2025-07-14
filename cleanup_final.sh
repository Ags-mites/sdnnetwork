#!/bin/bash
# Script de limpieza final - Eliminar archivos innecesarios
# Archivo: cleanup_final.sh

PROJECT_DIR="$(pwd)"

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[CLEANUP]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[REMOVE]${NC} $1"
}

info() {
    echo -e "${BLUE}[KEEP]${NC} $1"
}

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════╗"
echo "║         LIMPIEZA FINAL DEL PROYECTO            ║"
echo "║    Eliminando archivos innecesarios            ║"
echo "╚════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
log "🗂️  Analizando archivos del proyecto..."

# ============================================================================
# ARCHIVOS COMPLETAMENTE INNECESARIOS (ELIMINAR)
# ============================================================================

echo ""
echo -e "${RED}📁 ARCHIVOS INNECESARIOS (SERÁN ELIMINADOS):${NC}"

# 1. Script de limpieza actual (este mismo)
if [[ -f "cleanup_project.sh" ]]; then
    warn "cleanup_project.sh - Script de limpieza obsoleto"
    rm -f cleanup_project.sh
fi

# 2. Logs antiguos (mantener solo los más recientes)
echo ""
warn "🗑️  Limpiando logs antiguos..."
cd logs 2>/dev/null && {
    # Mantener solo el log más reciente de cada tipo
    for prefix in mininet ryu; do
        files=($(ls -t ${prefix}_*.log 2>/dev/null | tail -n +2))  # Todo excepto el más reciente
        if [[ ${#files[@]} -gt 0 ]]; then
            rm -f "${files[@]}"
            warn "  Eliminados ${#files[@]} logs antiguos de $prefix"
        fi
    done
    
    # Eliminar logs específicos innecesarios
    rm -f traffic_20250713_070650.log orchestrator.log 2>/dev/null
    cd ..
}

# 3. Resultados de evaluación duplicados (mantener solo el más reciente)
echo ""
warn "🗑️  Limpiando resultados duplicados..."
cd evaluation_results 2>/dev/null && {
    # Mantener solo el modelo más reciente
    models=($(ls -t best_sdn_model_*.joblib 2>/dev/null | tail -n +2))
    if [[ ${#models[@]} -gt 0 ]]; then
        rm -f "${models[@]}"
        warn "  Eliminados ${#models[@]} modelos antiguos"
    fi
    
    # Mantener solo el reporte detallado más reciente
    reports=($(ls -t detailed_evaluation_*.txt 2>/dev/null | tail -n +2))
    if [[ ${#reports[@]} -gt 0 ]]; then
        rm -f "${reports[@]}"
        warn "  Eliminados ${#reports[@]} reportes antiguos"
    fi
    
    # Mantener solo el resumen más reciente
    summaries=($(ls -t evaluation_summary_*.txt 2>/dev/null | tail -n +2))
    if [[ ${#summaries[@]} -gt 0 ]]; then
        rm -f "${summaries[@]}"
        warn "  Eliminados ${#summaries[@]} resúmenes antiguos"
    fi
    cd ..
}

# 4. Directorio results vacío
if [[ -d "results" ]] && [[ -z "$(ls -A results)" ]]; then
    warn "results/ - Directorio vacío"
    rmdir results
fi

# 5. Archivos de tráfico que no se usan
cd traffic 2>/dev/null && {
    if [[ -f "generate_csv.py" ]]; then
        # Este archivo no se usa en el flujo principal
        warn "traffic/generate_csv.py - No se usa en el flujo principal"
        # rm -f generate_csv.py  # Comentado por si se quiere mantener
    fi
    cd ..
}

# ============================================================================
# ARCHIVOS ESENCIALES (MANTENER)
# ============================================================================

echo ""
echo -e "${GREEN}📁 ARCHIVOS ESENCIALES (SE MANTIENEN):${NC}"

info "🎮 controller/sla_monitor.py - Controlador SDN estándar"
info "🤖 controller/sla_monitor_ml.py - Controlador ML para SLA"
info "🌐 topology/custom_topo.py - Topología de red SDN"
info "🔧 orchestration/orchestrator_simple.sh - Orquestador básico"
info "🚀 orchestration/ml_mass_simulator_fixed.sh - Simulador ML masivo"
info "📊 orchestration/run_ml_evaluation.sh - Evaluación ML"
info "🤖 evaluation/ml_sdn_evaluator.py - Evaluador ML"
info "📈 data/ml_dataset_binary_20250713_181954.csv - Dataset principal"
info "🎯 evaluation_results/[más reciente] - Modelo y reportes finales"
info "📝 logs/[más recientes] - Logs de la última ejecución"

# ============================================================================
# RESUMEN FINAL
# ============================================================================

echo ""
log "✅ Limpieza completada"

echo ""
echo -e "${BLUE}📋 ESTRUCTURA FINAL OPTIMIZADA:${NC}"
echo "controller/               # Controladores SDN"
echo "├── sla_monitor.py       # Controlador estándar"
echo "└── sla_monitor_ml.py    # Controlador ML"
echo ""
echo "topology/"
echo "└── custom_topo.py       # Topología 6h/3s/2sw"
echo ""
echo "orchestration/           # Scripts de ejecución"
echo "├── orchestrator_simple.sh"
echo "├── ml_mass_simulator_fixed.sh"
echo "└── run_ml_evaluation.sh"
echo ""
echo "evaluation/"
echo "└── ml_sdn_evaluator.py  # Evaluador ML"
echo ""
echo "data/"
echo "└── ml_dataset_*.csv     # Dataset principal"
echo ""
echo "evaluation_results/      # Resultados finales"
echo "├── best_sdn_model_*.joblib"
echo "├── detailed_evaluation_*.txt"
echo "└── evaluation_summary_*.txt"
echo ""
echo "logs/                    # Logs recientes"
echo ""
echo "traffic/                 # Herramientas auxiliares"
echo "└── traffic_gen.sh"

echo ""
log "🎯 Proyecto optimizado y listo para uso/presentación"