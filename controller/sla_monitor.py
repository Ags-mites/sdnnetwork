#!/usr/bin/env python3
"""
Controlador Ryu con monitoreo automático de SLA
Archivo: controller/sla_monitor.py

Funcionalidades:
- Forwarding básico L2
- Monitoreo de métricas de red
- Clasificación automática de SLA
- Exportación a CSV
"""

import time
import csv
import threading
from datetime import datetime
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp
from ryu.lib import hub

class SLAMonitorController(app_manager.RyuApp):
    """Controlador con monitoreo SLA automatizado"""
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(SLAMonitorController, self).__init__(*args, **kwargs)
        
        # Tabla MAC para forwarding L2
        self.mac_to_port = {}
        
        # Métricas de red
        self.flow_stats = {}
        self.port_stats = {}
        
        # Umbrales SLA
        self.sla_thresholds = {
            'latency_ms': 50,
            'jitter_ms': 10,
            'packet_loss_percent': 1.0,
            'throughput_mbps': 10.0
        }
        
        # CSV para datasets
        self.csv_file = '../data/metrics.csv'
        self.init_csv()
        
        # Hilo para monitoreo periódico
        self.monitor_thread = hub.spawn(self._monitor_loop)
        
        self.logger.info("� SLA Monitor Controller iniciado")
    
    def init_csv(self):
        """Inicializar archivo CSV con headers"""
        try:
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
                    'sla_status'
                ])
            self.logger.info(f"� CSV inicializado: {self.csv_file}")
        except Exception as e:
            self.logger.error(f"❌ Error inicializando CSV: {e}")
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Configurar switch al conectarse"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Regla por defecto: enviar al controlador
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        self.logger.info(f"� Switch {datapath.id} conectado")
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """Agregar regla de flujo"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                           actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                  priority=priority, match=match,
                                  instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                  match=match, instructions=inst)
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """Manejar paquetes entrantes"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        # Ignorar paquetes LLDP
        if eth.ethertype == 0x88cc:
            return
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        # Aprender MAC address
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        
        # Buscar puerto de destino
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        # Crear acciones
        actions = [parser.OFPActionOutput(out_port)]
        
        # Instalar regla de flujo si no es flood
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions, msg.buffer_id)
            
            # Registrar flujo para monitoreo
            flow_key = f"{dpid}_{src}_{dst}"
            self.flow_stats[flow_key] = {
                'timestamp': time.time(),
                'src_mac': src,
                'dst_mac': dst,
                'switch_id': dpid,
                'packets': 0,
                'bytes': 0
            }
        
        # Enviar paquete
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
    
    def _monitor_loop(self):
        """Hilo de monitoreo periódico"""
        while True:
            self._collect_metrics()
            hub.sleep(10)  # Monitorear cada 10 segundos
    
    def _collect_metrics(self):
        """Recopilar métricas de red"""
        try:
            # Simular métricas (en implementación real, usar OpenFlow stats)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Generar métricas simuladas para diferentes pares de hosts
            host_pairs = [
                ('h1', 'h3'), ('h1', 'h4'), ('h1', 'h5'), ('h1', 'h6'),
                ('h2', 'h3'), ('h2', 'h4'), ('h2', 'h5'), ('h2', 'h6'),
                ('h3', 'h5'), ('h3', 'h6'), ('h4', 'h5'), ('h4', 'h6')
            ]
            
            import random
            
            for src, dst in host_pairs:
                # Simular métricas con variaciones realistas
                latency = random.uniform(5, 80)  # 5-80ms
                jitter = random.uniform(0.5, 15)  # 0.5-15ms
                packet_loss = random.uniform(0, 3)  # 0-3%
                throughput = random.uniform(5, 95)  # 5-95 Mbps
                
                # Clasificar SLA
                sla_status = self._classify_sla(latency, jitter, packet_loss, throughput)
                
                # Escribir al CSV
                self._write_to_csv(timestamp, src, dst, latency, jitter, 
                                 packet_loss, throughput, sla_status)
            
            self.logger.info(f"� Métricas recopiladas: {timestamp}")
            
        except Exception as e:
            self.logger.error(f"❌ Error recopilando métricas: {e}")
    
    def _classify_sla(self, latency, jitter, packet_loss, throughput):
        """Clasificar estado SLA basado en umbrales"""
        violations = []
        
        if latency > self.sla_thresholds['latency_ms']:
            violations.append('latency')
        if jitter > self.sla_thresholds['jitter_ms']:
            violations.append('jitter')
        if packet_loss > self.sla_thresholds['packet_loss_percent']:
            violations.append('packet_loss')
        if throughput < self.sla_thresholds['throughput_mbps']:
            violations.append('throughput')
        
        if len(violations) == 0:
            return 'OK'
        elif len(violations) == 1:
            return 'WARN'
        else:
            return 'VIOLATED'
    
    def _write_to_csv(self, timestamp, src, dst, latency, jitter, 
                      packet_loss, throughput, sla_status):
        """Escribir métricas al archivo CSV"""
        try:
            with open(self.csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    timestamp, src, dst, 
                    round(latency, 3), round(jitter, 3), 
                    round(packet_loss, 3), round(throughput, 3),
                    sla_status
                ])
        except Exception as e:
            self.logger.error(f"❌ Error escribiendo CSV: {e}")

# Punto de entrada
if __name__ == '__main__':
    pass
