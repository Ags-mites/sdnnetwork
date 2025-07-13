#!/usr/bin/env python3
"""
Script de procesamiento y análisis de métricas CSV
Archivo: traffic/generate_csv.py

Funcionalidades:
- Procesamiento de métricas de red
- Análisis estadístico de SLA
- Generación de datasets para ML
- Visualización de resultados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import os
import sys

class SLAAnalyzer:
    """Analizador de métricas SLA para datasets ML"""
    
    def __init__(self, csv_file='../data/metrics.csv'):
        self.csv_file = csv_file
        self.df = None
        self.sla_thresholds = {
            'latency_ms': 50,
            'jitter_ms': 10,
            'packet_loss_percent': 1.0,
            'throughput_mbps': 10.0
        }
    
    def load_data(self):
        """Cargar datos desde CSV"""
        try:
            if not os.path.exists(self.csv_file):
                print(f"❌ Archivo CSV no encontrado: {self.csv_file}")
                return False
            
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Datos cargados: {len(self.df)} registros")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando CSV: {e}")
            return False
    
    def analyze_sla_distribution(self):
        """Analizar distribución de estados SLA"""
        if self.df is None:
            print("❌ No hay datos cargados")
            return
        
        print("\n� DISTRIBUCIÓN DE ESTADOS SLA")
        print("=" * 40)
        
        sla_counts = self.df['sla_status'].value_counts()
        total = len(self.df)
        
        for status, count in sla_counts.items():
            percentage = (count / total) * 100
            print(f"{status:>10}: {count:>6} ({percentage:>5.1f}%)")
        
        return sla_counts
    
    def analyze_metrics_by_host_pair(self):
        """Analizar métricas por par de hosts"""
        if self.df is None:
            return
        
        print("\n� ANÁLISIS POR PAR DE HOSTS")
        print("=" * 40)
        
        # Crear columna de par de hosts
        self.df['host_pair'] = self.df['src_host'] + ' -> ' + self.df['dst_host']
        
        # Agrupar por par de hosts
        host_pairs = self.df.groupby('host_pair').agg({
            'latency_ms': ['mean', 'std', 'min', 'max'],
            'jitter_ms': ['mean', 'std'],
            'packet_loss_percent': ['mean', 'max'],
            'throughput_mbps': ['mean', 'min'],
            'sla_status': lambda x: (x == 'VIOLATED').sum()
        }).round(2)
        
        # Aplanar nombres de columnas
        host_pairs.columns = [f"{col[0]}_{col[1]}" for col in host_pairs.columns]
        host_pairs.columns = ['lat_mean', 'lat_std', 'lat_min', 'lat_max',
                             'jit_mean', 'jit_std', 'loss_mean', 'loss_max',
                             'thr_mean', 'thr_min', 'violations']
        
        print(host_pairs)
        return host_pairs
    
    def identify_problematic_paths(self):
        """Identificar rutas problemáticas"""
        if self.df is None:
            return
        
        print("\n⚠️  RUTAS PROBLEMÁTICAS (>50% violaciones)")
        print("=" * 50)
        
        # Calcular porcentaje de violaciones por par de hosts
        violations = self.df.groupby(['src_host', 'dst_host']).agg({
            'sla_status': [
                'count',
                lambda x: (x == 'VIOLATED').sum(),
                lambda x: (x == 'VIOLATED').sum() / len(x) * 100
            ]
        })
        
        violations.columns = ['total', 'violations', 'violation_rate']
        violations = violations.round(1)
        
        # Filtrar rutas con >50% violaciones
        problematic = violations[violations['violation_rate'] > 50]
        
        if len(problematic) > 0:
            print(problematic)
        else:
            print("✅ No se encontraron rutas problemáticas")
        
        return problematic
    
    def generate_time_series_analysis(self):
        """Análisis de series temporales"""
        if self.df is None:
            return
        
        print("\n� ANÁLISIS TEMPORAL")
        print("=" * 30)
        
        # Convertir timestamp a datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Resamplear por minutos
        time_series = self.df.set_index('timestamp').resample('1T').agg({
            'latency_ms': 'mean',
            'jitter_ms': 'mean',
            'packet_loss_percent': 'mean',
            'throughput_mbps': 'mean',
            'sla_status': lambda x: (x == 'VIOLATED').sum()
        }).round(2)
        
        print("Métricas promedio por minuto:")
        print(time_series.head(10))
        
        return time_series
    
    def export_ml_dataset(self, output_file='../data/ml_dataset.csv'):
        """Exportar dataset preparado para ML"""
        if self.df is None:
            return False
        
        print(f"\n� PREPARANDO DATASET PARA ML")
        print("=" * 35)
        
        # Crear features adicionales
        ml_df = self.df.copy()
        
        # Codificar host pairs como features numéricas
        ml_df['src_host_num'] = ml_df['src_host'].str.extract('(\d+)').astype(int)
        ml_df['dst_host_num'] = ml_df['dst_host'].str.extract('(\d+)').astype(int)
        
        # Feature engineering
        ml_df['latency_jitter_ratio'] = ml_df['latency_ms'] / (ml_df['jitter_ms'] + 0.1)
        ml_df['quality_score'] = (
            (ml_df['latency_ms'] < self.sla_thresholds['latency_ms']).astype(int) +
            (ml_df['jitter_ms'] < self.sla_thresholds['jitter_ms']).astype(int) +
            (ml_df['packet_loss_percent'] < self.sla_thresholds['packet_loss_percent']).astype(int) +
            (ml_df['throughput_mbps'] > self.sla_thresholds['throughput_mbps']).astype(int)
        )
        
        # Codificar SLA status como target numérico
        sla_mapping = {'OK': 2, 'WARN': 1, 'VIOLATED': 0}
        ml_df['sla_target'] = ml_df['sla_status'].map(sla_mapping)
        
        # Seleccionar features para ML
        feature_columns = [
            'src_host_num', 'dst_host_num',
            'latency_ms', 'jitter_ms', 'packet_loss_percent', 'throughput_mbps',
            'latency_jitter_ratio', 'quality_score', 'sla_target'
        ]
        
        ml_dataset = ml_df[feature_columns].copy()
        
        # Guardar dataset
        ml_dataset.to_csv(output_file, index=False)
        
        print(f"✅ Dataset ML exportado: {output_file}")
        print(f"� Features: {len(feature_columns)-1}")  # -1 porque target no es feature
        print(f"� Registros: {len(ml_dataset)}")
        print(f"� Target distribution:")
        print(ml_dataset['sla_target'].value_counts().sort_index())
        
        return True
    
    def create_visualizations(self, output_dir='../data/plots'):
        """Crear visualizaciones de las métricas"""
        if self.df is None:
            return
        
        print(f"\n� GENERANDO VISUALIZACIONES")
        print("=" * 35)
        
        # Crear directorio de plots
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig_size = (12, 8)
        
        # 1. Distribución de SLA
        plt.figure(figsize=(8, 6))
        sla_counts = self.df['sla_status'].value_counts()
        colors = ['green', 'orange', 'red']
        plt.pie(sla_counts.values, labels=sla_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Distribución de Estados SLA')
        plt.savefig(f'{output_dir}/sla_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Métricas por host pair
        plt.figure(figsize=fig_size)
        host_pair_data = self.df.groupby(['src_host', 'dst_host'])['latency_ms'].mean().reset_index()
        pivot_data = host_pair_data.pivot(index='src_host', columns='dst_host', values='latency_ms')
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn_r', fmt='.1f')
        plt.title('Latencia Promedio por Par de Hosts (ms)')
        plt.savefig(f'{output_dir}/latency_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Distribución de métricas
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['latency_ms', 'jitter_ms', 'packet_loss_percent', 'throughput_mbps']
        titles = ['Latencia (ms)', 'Jitter (ms)', 'Pérdida de Paquetes (%)', 'Throughput (Mbps)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2
            axes[row, col].hist(self.df[metric], bins=30, alpha=0.7, edgecolor='black')
            axes[row, col].axvline(self.sla_thresholds.get(metric, 0), 
                                  color='red', linestyle='--', label='Umbral SLA')
            axes[row, col].set_xlabel(title)
            axes[row, col].set_ylabel('Frecuencia')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizaciones guardadas en: {output_dir}/")
    
    def generate_summary_report(self, output_file='../data/analysis_report.txt'):
        """Generar reporte resumen del análisis"""
        if self.df is None:
            return
        
        with open(output_file, 'w') as f:
            f.write("# REPORTE DE ANÁLISIS SLA\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            # Estadísticas generales
            f.write("## ESTADÍSTICAS GENERALES\n")
            f.write(f"Total de registros: {len(self.df)}\n")
            f.write(f"Período: {self.df['timestamp'].min()} - {self.df['timestamp'].max()}\n")
            f.write(f"Hosts únicos: {self.df['src_host'].nunique()}\n\n")
            
            # Distribución SLA
            f.write("## DISTRIBUCIÓN SLA\n")
            sla_counts = self.df['sla_status'].value_counts()
            total = len(self.df)
            for status, count in sla_counts.items():
                percentage = (count / total) * 100
                f.write(f"{status}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Métricas promedio
            f.write("## MÉTRICAS PROMEDIO\n")
            for metric in ['latency_ms', 'jitter_ms', 'packet_loss_percent', 'throughput_mbps']:
                mean_val = self.df[metric].mean()
                std_val = self.df[metric].std()
                f.write(f"{metric}: {mean_val:.2f} ± {std_val:.2f}\n")
            f.write("\n")
            
            # Recomendaciones
            f.write("## RECOMENDACIONES\n")
            violation_rate = (sla_counts.get('VIOLATED', 0) / total) * 100
            
            if violation_rate > 20:
                f.write("⚠️  Alto porcentaje de violaciones SLA (>20%)\n")
                f.write("- Revisar configuración de red\n")
                f.write("- Optimizar rutas de forwarding\n")
            elif violation_rate > 10:
                f.write("⚠️  Violaciones SLA moderadas (10-20%)\n")
                f.write("- Monitorear rutas problemáticas\n")
            else:
                f.write("✅ Bajo porcentaje de violaciones SLA (<10%)\n")
                f.write("- Red funcionando dentro de parámetros aceptables\n")
        
        print(f"✅ Reporte de análisis guardado: {output_file}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Analizador de métricas SLA')
    parser.add_argument('-f', '--file', default='../data/metrics.csv',
                       help='Archivo CSV de entrada')
    parser.add_argument('-o', '--output', default='../data',
                       help='Directorio de salida')
    parser.add_argument('--no-plots', action='store_true',
                       help='No generar visualizaciones')
    parser.add_argument('--ml-only', action='store_true',
                       help='Solo generar dataset ML')
    
    args = parser.parse_args()
    
    print("� ANALIZADOR DE MÉTRICAS SLA")
    print("=" * 35)
    
    # Crear analizador
    analyzer = SLAAnalyzer(args.file)
    
    # Cargar datos
    if not analyzer.load_data():
        sys.exit(1)
    
    if args.ml_only:
        # Solo generar dataset ML
        ml_file = os.path.join(args.output, 'ml_dataset.csv')
        analyzer.export_ml_dataset(ml_file)
        return
    
    # Análisis completo
    print("\n" + "="*50)
    print("           ANÁLISIS COMPLETO")
    print("="*50)
    
    # Análisis de distribución SLA
    analyzer.analyze_sla_distribution()
    
    # Análisis por pares de hosts
    analyzer.analyze_metrics_by_host_pair()
    
    # Identificar rutas problemáticas
    analyzer.identify_problematic_paths()
    
    # Análisis temporal
    analyzer.generate_time_series_analysis()
    
    # Exportar dataset ML
    ml_file = os.path.join(args.output, 'ml_dataset.csv')
    analyzer.export_ml_dataset(ml_file)
    
    # Crear visualizaciones
    if not args.no_plots:
        plots_dir = os.path.join(args.output, 'plots')
        analyzer.create_visualizations(plots_dir)
    
    # Generar reporte resumen
    report_file = os.path.join(args.output, 'analysis_report.txt')
    analyzer.generate_summary_report(report_file)
    
    print(f"\n� Análisis completado")
    print(f"� Archivos generados en: {args.output}/")

if __name__ == '__main__':
    main()
