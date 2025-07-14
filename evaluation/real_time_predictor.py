#!/usr/bin/env python3
"""
Predictor de SLA en Tiempo Real
Usa el modelo ML entrenado para predicciones en vivo
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import time
import argparse

class RealTimeSLAPredictor:
    """Predictor de violaciones SLA en tiempo real usando modelo ML"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or self._find_latest_model()
        self.load_model()
        
        # Configurar features esperadas
        self.feature_names = [
            'latency_ms', 'jitter_ms', 'packet_loss_percent', 'throughput_mbps',
            'hour_of_day', 'traffic_multiplier', 'latency_jitter_ratio', 'is_weekend'
        ]
        
        print(f"🤖 Predictor SLA inicializado")
        print(f"📊 Modelo: {os.path.basename(self.model_path) if self.model_path else 'No encontrado'}")
    
    def _find_latest_model(self):
        """Encontrar modelo más reciente"""
        import glob
        
        model_files = glob.glob('../results/sdn_model_*.joblib')
        if model_files:
            return max(model_files, key=os.path.getctime)
        return None
    
    def load_model(self):
        """Cargar modelo ML"""
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"❌ Modelo no encontrado: {self.model_path}")
            return False
        
        try:
            self.model = joblib.load(self.model_path)
            print(f"✅ Modelo cargado exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def prepare_features(self, metrics_dict):
        """Preparar features para predicción"""
        try:
            # Obtener contexto temporal
            now = datetime.now()
            hour = now.hour
            is_weekend = now.weekday() >= 5
            
            # Determinar multiplicador de tráfico basado en hora
            if hour in [7, 8, 9]:
                traffic_mult = 1.4  # morning_rush
            elif hour in [17, 18, 19]:
                traffic_mult = 1.7  # evening_rush
            elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:
                traffic_mult = 0.7  # night_low
            else:
                traffic_mult = 1.0  # normal
            
            if is_weekend:
                traffic_mult *= 0.8  # weekend adjustment
            
            # Calcular ratio latencia/jitter
            latency_jitter_ratio = metrics_dict['latency_ms'] / (metrics_dict.get('jitter_ms', 1) + 0.1)
            
            # Crear vector de features
            features = [
                metrics_dict['latency_ms'],
                metrics_dict['jitter_ms'],
                metrics_dict['packet_loss_percent'],
                metrics_dict['throughput_mbps'],
                hour,
                traffic_mult,
                latency_jitter_ratio,
                1.0 if is_weekend else 0.0
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"❌ Error preparando features: {e}")
            return None
    
    def predict_sla_violation(self, metrics_dict):
        """Predecir violación SLA"""