#!/usr/bin/env python3
"""
Topología SDN Personalizada para Proyecto ML
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
    def __init__(self):
        Topo.__init__(self)
        
        s1 = self.addSwitch('s1', protocols='OpenFlow13')
        s2 = self.addSwitch('s2', protocols='OpenFlow13')
        s3 = self.addSwitch('s3', protocols='OpenFlow13')
        
        h1 = self.addHost('h1', ip='10.0.1.10/24', defaultRoute='via 10.0.1.1')
        h2 = self.addHost('h2', ip='10.0.1.20/24', defaultRoute='via 10.0.1.1')
        h3 = self.addHost('h3', ip='10.0.2.10/24', defaultRoute='via 10.0.2.1')
        h4 = self.addHost('h4', ip='10.0.2.20/24', defaultRoute='via 10.0.2.1')
        h5 = self.addHost('h5', ip='10.0.3.10/24', defaultRoute='via 10.0.3.1')
        h6 = self.addHost('h6', ip='10.0.3.20/24', defaultRoute='via 10.0.3.1')
        
        self.addLink(h1, s1, bw=1000, delay='1ms', loss=0)
        self.addLink(h2, s1, bw=1000, delay='1ms', loss=0)
        self.addLink(h3, s2, bw=1000, delay='1ms', loss=0)
        self.addLink(h4, s2, bw=1000, delay='1ms', loss=0)
        self.addLink(h5, s3, bw=1000, delay='1ms', loss=0)
        self.addLink(h6, s3, bw=1000, delay='1ms', loss=0)
        
        self.addLink(s1, s2, bw=100, delay='5ms', loss=0)     
        self.addLink(s1, s3, bw=100, delay='20ms', loss=0)    
        self.addLink(s2, s3, bw=50, delay='10ms', loss=0.1)   

def run_topology():
    """Ejecutar la topología para testing"""
    setLogLevel('info')
    topo = CustomSDNTopo()
    net = Mininet(
        topo=topo,
        controller=RemoteController,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )
    net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)
    
    net.start()
    print("----Topología SDN iniciada----")
    print("# Configuración de hosts:")
    for host in net.hosts:
        print(f"  {host.name}: {host.IP()}")
    
    print("\n# Configuración de switches:")
    for switch in net.switches:
        print(f"  {switch.name}: {switch.dpid}")
    
    print("\n----Red lista para monitoreo SLA----")
    CLI(net)
    net.stop()

if __name__ == '__main__':
    run_topology()