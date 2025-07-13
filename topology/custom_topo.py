#!/usr/bin/env python3
"""
Topología SDN personalizada: 6 hosts / 3 subredes / 2 switches
Archivo: topology/custom_topo.py

Diseño de red:
- Switch s1: h1, h2 (subred 10.0.1.0/24)
- Switch s2: h3, h4 (subred 10.0.2.0/24)
- Switch s3: h5, h6 (subred 10.0.3.0/24)
- Enlaces inter-switch con diferentes capacidades para testing SLA
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import TCLink

class CustomSDNTopo(Topo):
    """Topología personalizada para testing SLA"""
    
    def __init__(self):
        # Inicializar topología
        Topo.__init__(self)
        
        # Crear switches
        s1 = self.addSwitch('s1', protocols='OpenFlow13')
        s2 = self.addSwitch('s2', protocols='OpenFlow13')
        s3 = self.addSwitch('s3', protocols='OpenFlow13')
        
        # Crear hosts con IPs específicas
        # Subred 1: 10.0.1.0/24
        h1 = self.addHost('h1', ip='10.0.1.10/24', defaultRoute='via 10.0.1.1')
        h2 = self.addHost('h2', ip='10.0.1.20/24', defaultRoute='via 10.0.1.1')
        
        # Subred 2: 10.0.2.0/24
        h3 = self.addHost('h3', ip='10.0.2.10/24', defaultRoute='via 10.0.2.1')
        h4 = self.addHost('h4', ip='10.0.2.20/24', defaultRoute='via 10.0.2.1')
        
        # Subred 3: 10.0.3.0/24
        h5 = self.addHost('h5', ip='10.0.3.10/24', defaultRoute='via 10.0.3.1')
        h6 = self.addHost('h6', ip='10.0.3.20/24', defaultRoute='via 10.0.3.1')
        
        # Enlaces host-switch (1 Gbps, latencia baja)
        self.addLink(h1, s1, bw=1000, delay='1ms', loss=0)
        self.addLink(h2, s1, bw=1000, delay='1ms', loss=0)
        self.addLink(h3, s2, bw=1000, delay='1ms', loss=0)
        self.addLink(h4, s2, bw=1000, delay='1ms', loss=0)
        self.addLink(h5, s3, bw=1000, delay='1ms', loss=0)
        self.addLink(h6, s3, bw=1000, delay='1ms', loss=0)
        
        # Enlaces inter-switch con diferentes características para testing SLA
        # s1-s2: Enlace rápido
        self.addLink(s1, s2, bw=100, delay='5ms', loss=0)
        
        # s1-s3: Enlace con mayor latencia
        self.addLink(s1, s3, bw=100, delay='20ms', loss=0)
        
        # s2-s3: Enlace con ancho de banda limitado
        self.addLink(s2, s3, bw=50, delay='10ms', loss=0.1)

def run_topology():
    """Ejecutar la topología con controlador remoto Ryu"""
    
    # Configurar nivel de log
    setLogLevel('info')
    
    # Crear topología
    topo = CustomSDNTopo()
    
    # Crear red con controlador remoto (Ryu)
    net = Mininet(
        topo=topo,
        controller=RemoteController,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )
    
    # Configurar controlador remoto
    net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)
    
    # Iniciar red
    net.start()
    
    print("� Topología SDN iniciada")
    print("� Configuración de hosts:")
    for host in net.hosts:
        print(f"  {host.name}: {host.IP()}")
    
    print("\n� Configuración de switches:")
    for switch in net.switches:
        print(f"  {switch.name}: {switch.dpid}")
    
    print("\n✅ Red lista para monitoreo SLA")
    print("� Usa 'pingall' para verificar conectividad")
    print("� Usa 'iperf h1 h4' para generar tráfico")
    
    # Abrir CLI para testing manual
    CLI(net)
    
    # Limpiar al salir
    net.stop()

if __name__ == '__main__':
    run_topology()
