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
