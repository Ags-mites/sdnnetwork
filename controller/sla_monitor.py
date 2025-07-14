#!/usr/bin/env python3
"""
Controlador SDN Unificado con Monitoreo SLA y Soporte ML
Consolidación de sla_monitor.py y sla_monitor_ml.py
"""

import time
import csv
import random
import math
import os
from datetime import datetime
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
from ryu.lib import hub

class UnifiedSLAController(app_manager.RyuApp):
    """Controlador SDN unificado para generación de datasets ML con SLA binario"""
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(UnifiedSLAController, self).__init__(*args, **kwargs)
        
        # Configuración básica
        self.mac_to_port = {}
        self.monitoring_interval = 2
        self.record_count = 0
        
        # Configuración desde variables de entorno o defaults
        self.target_records = int(os.environ.get('SLA_TARGET_RECORDS', 3000))
        self.csv_file = os.environ.get('SLA_CSV_FILE', '../data/metrics.csv')
        
        # Umbrales SLA - TODOS deben cumplirse para SLA=True
        self.sla_thresholds = {
            'latency_ms': 50.0,
            'jitter_ms': 10.0,
            'packet_loss_percent': 1.0,
            'throughput_mbps': 10.0
        }
        
        # Configuración de hosts y topología
        self.hosts = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        self.path_configs = self._init_path_configs()
        
        # Patrones de tráfico realistas
        self.traffic_patterns = {
            'morning_rush': {'multiplier': 1.4, 'hours': [7, 8, 9]},
            'lunch_time': {'multiplier': 1.1, 'hours': [12, 13]},
            'evening_rush': {'multiplier': 1.7, 'hours': [17, 18, 19]},
            'night_low': {'multiplier': 0.7, 'hours': [22, 23, 0, 1, 2, 3, 4, 5]},
            'weekend': {'multiplier': 0.8, 'hours': list(range(24))}
        }
        
        # Inicializar
        self.init_csv()
        self.monitor_thread = hub.spawn(self._monitoring_loop)
        self.logger.info(f"Unified SLA Controller iniciado (Target: {self.target_records})")
    
    def _init_path_configs(self):
        """Configurar características de rutas según topología 6h/3s"""
        return {
            # Rutas rápidas 
            ('h1', 'h3'): {'type': 'fast', 'base_latency': 10, 'base_throughput': 85},
            ('h1', 'h4'): {'type': 'fast', 'base_latency': 12, 'base_throughput': 80},
            ('h2', 'h3'): {'type': 'fast', 'base_latency': 11, 'base_throughput': 82},
            ('h2', 'h4'): {'type': 'fast', 'base_latency': 13, 'base_throughput': 78},
            ('h3', 'h1'): {'type': 'fast', 'base_latency': 10, 'base_throughput': 83},
            ('h3', 'h2'): {'type': 'fast', 'base_latency': 11, 'base_throughput': 81},
            ('h4', 'h1'): {'type': 'fast', 'base_latency': 12, 'base_throughput': 79},
            ('h4', 'h2'): {'type': 'fast', 'base_latency': 13, 'base_throughput': 77},
            
            # Rutas lentas
            ('h1', 'h5'): {'type': 'slow', 'base_latency': 28, 'base_throughput': 70},
            ('h1', 'h6'): {'type': 'slow', 'base_latency': 32, 'base_throughput': 65},
            ('h2', 'h5'): {'type': 'slow', 'base_latency': 30, 'base_throughput': 68},
            ('h2', 'h6'): {'type': 'slow', 'base_latency': 35, 'base_throughput': 62},
            ('h5', 'h1'): {'type': 'slow', 'base_latency': 29, 'base_throughput': 69},
            ('h5', 'h2'): {'type': 'slow', 'base_latency': 31, 'base_throughput': 67},
            ('h6', 'h1'): {'type': 'slow', 'base_latency': 33, 'base_throughput': 64},
            ('h6', 'h2'): {'type': 'slow', 'base_latency': 36, 'base_throughput': 61},
            
            # Rutas limitadas
            ('h3', 'h5'): {'type': 'limited', 'base_latency': 18, 'base_throughput': 45},
            ('h3', 'h6'): {'type': 'limited', 'base_latency': 22, 'base_throughput': 40},
            ('h4', 'h5'): {'type': 'limited', 'base_latency': 20, 'base_throughput': 42},
            ('h4', 'h6'): {'type': 'limited', 'base_latency': 25, 'base_throughput': 38},
            ('h5', 'h3'): {'type': 'limited', 'base_latency': 19, 'base_throughput': 44},
            ('h5', 'h4'): {'type': 'limited', 'base_latency': 21, 'base_throughput': 41},
            ('h6', 'h3'): {'type': 'limited', 'base_latency': 23, 'base_throughput': 39},
            ('h6', 'h4'): {'type': 'limited', 'base_latency': 26, 'base_throughput': 37},
            
            # Rutas intra-switch
            ('h1', 'h2'): {'type': 'intra', 'base_latency': 2, 'base_throughput': 95},
            ('h2', 'h1'): {'type': 'intra', 'base_latency': 2, 'base_throughput': 95},
            ('h3', 'h4'): {'type': 'intra', 'base_latency': 2, 'base_throughput': 95},
            ('h4', 'h3'): {'type': 'intra', 'base_latency': 2, 'base_throughput': 95},
            ('h5', 'h6'): {'type': 'intra', 'base_latency': 2, 'base_throughput': 95},
            ('h6', 'h5'): {'type': 'intra', 'base_latency': 2, 'base_throughput': 95},
        }
    
    def init_csv(self):
        """Inicializar CSV con headers optimizados para ML"""
        try:
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp',
                    'src_host',
                    'dst_host',
                    'latency_ms',
                    'jitter_ms', 
                    'packet_loss_percent',
                    'throughput_mbps',
                    'sla_compliant',
                    'path_type',
                    'traffic_pattern',
                    'hour_of_day',
                    'is_weekend'
                ])
            self.logger.info(f"CSV ML inicializado: {self.csv_file}")
        except Exception as e:
            self.logger.error(f"Error inicializando CSV: {e}")
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Configurar switch al conectarse"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info(f"Switch {datapath.id} conectado")
    
    def add_flow(self, datapath, priority, match, actions):
        """Agregar regla de flujo"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                              match=match, instructions=inst)
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """Manejar paquetes entrantes - forwarding L2 básico"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        if eth.ethertype == 0x88cc:  # LLDP
            return
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]
        
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions)
        
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
    
    def _monitoring_loop(self):
        """Loop principal de monitoreo cada 2 segundos"""
        self.logger.info(f"Iniciando monitoreo ML (target: {self.target_records})")
        
        while self.record_count < self.target_records:
            self._collect_sla_metrics()
            
            if self.record_count % 100 == 0 and self.record_count > 0:
                progress = (self.record_count * 100) // self.target_records
                self.logger.info(f"Progreso: {self.record_count}/{self.target_records} ({progress}%)")
            
            hub.sleep(self.monitoring_interval)
        
        self.logger.info(f"Target alcanzado: {self.record_count} registros generados")
    
    def _get_current_context(self):
        """Obtener contexto temporal actual"""
        now = datetime.now()
        hour = now.hour
        is_weekend = now.weekday() >= 5
        
        # Determinar patrón de tráfico
        pattern_name = 'normal'
        traffic_multiplier = 1.0
        
        if is_weekend:
            pattern_name = 'weekend'
            traffic_multiplier = self.traffic_patterns['weekend']['multiplier']
        else:
            for pattern, config in self.traffic_patterns.items():
                if pattern != 'weekend' and hour in config['hours']:
                    pattern_name = pattern
                    traffic_multiplier = config['multiplier']
                    break
        
        return {
            'hour': hour,
            'is_weekend': is_weekend,
            'pattern_name': pattern_name,
            'traffic_multiplier': traffic_multiplier
        }
    
    def _generate_realistic_metrics(self, src, dst, context, simulation_time):
        """Generar métricas realistas basadas en topología y contexto"""
        path_config = self.path_configs.get((src, dst), {
            'type': 'unknown',
            'base_latency': 25,
            'base_throughput': 50
        })
        
        base_latency = path_config['base_latency']
        base_throughput = path_config['base_throughput']
        path_type = path_config['type']
        
        # Aplicar multiplicador de tráfico
        traffic_mult = context['traffic_multiplier']
        
        # Variación temporal sinusoidal
        time_variation = 1 + 0.3 * math.sin(simulation_time / 120)
        
        # Ruido aleatorio controlado
        noise = random.uniform(0.8, 1.2)
        
        # LATENCIA
        latency = base_latency * traffic_mult * time_variation * noise
        latency = max(1.0, min(100.0, latency))
        
        # JITTER (20-40% de la latencia)
        jitter_ratio = random.uniform(0.2, 0.4)
        jitter = latency * jitter_ratio * random.uniform(0.5, 1.5)
        jitter = max(0.1, min(25.0, jitter))
        
        # PACKET LOSS
        loss_base = {
            'fast': 0.05, 'slow': 0.15, 'limited': 0.8, 
            'intra': 0.01, 'unknown': 0.3
        }
        packet_loss = loss_base[path_type] * traffic_mult * noise
        packet_loss = max(0.0, min(5.0, packet_loss))
        
        # THROUGHPUT
        throughput = base_throughput / traffic_mult * random.uniform(0.9, 1.1)
        throughput = max(5.0, min(100.0, throughput))
        
        return {
            'latency': latency,
            'jitter': jitter,
            'packet_loss': packet_loss,
            'throughput': throughput,
            'path_type': path_type
        }
    
    def _calculate_binary_sla(self, metrics):
        """Calcular SLA binario: True si TODAS las métricas cumplen"""
        return (
            metrics['latency'] <= self.sla_thresholds['latency_ms'] and
            metrics['jitter'] <= self.sla_thresholds['jitter_ms'] and
            metrics['packet_loss'] <= self.sla_thresholds['packet_loss_percent'] and
            metrics['throughput'] >= self.sla_thresholds['throughput_mbps']
        )
    
    def _collect_sla_metrics(self):
        """Recopilar métricas SLA para todos los pares de hosts"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            simulation_time = time.time() % 3600
            context = self._get_current_context()
            
            for src in self.hosts:
                for dst in self.hosts:
                    if src != dst:
                        metrics = self._generate_realistic_metrics(src, dst, context, simulation_time)
                        sla_compliant = self._calculate_binary_sla(metrics)
                        
                        self._write_csv_record(
                            timestamp=timestamp,
                            src=src,
                            dst=dst,
                            latency=metrics['latency'],
                            jitter=metrics['jitter'],
                            packet_loss=metrics['packet_loss'],
                            throughput=metrics['throughput'],
                            sla_compliant=sla_compliant,
                            path_type=metrics['path_type'],
                            traffic_pattern=context['pattern_name'],
                            hour_of_day=context['hour'],
                            is_weekend=context['is_weekend']
                        )
                        
                        self.record_count += 1
                        if self.record_count >= self.target_records:
                            return
            
        except Exception as e:
            self.logger.error(f"Error recopilando métricas: {e}")
    
    def _write_csv_record(self, **kwargs):
        """Escribir registro al CSV"""
        try:
            with open(self.csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    kwargs['timestamp'],
                    kwargs['src'],
                    kwargs['dst'],
                    round(kwargs['latency'], 3),
                    round(kwargs['jitter'], 3),
                    round(kwargs['packet_loss'], 3),
                    round(kwargs['throughput'], 3),
                    kwargs['sla_compliant'],
                    kwargs['path_type'],
                    kwargs['traffic_pattern'],
                    kwargs['hour_of_day'],
                    kwargs['is_weekend']
                ])
        except Exception as e:
            self.logger.error(f"Error escribiendo CSV: {e}")

if __name__ == '__main__':
    pass