"""Message/parameter wiring checks with ROS stand-ins, not ROS acceptance."""
import importlib.util
import math
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import pytest

# Works from the source ROS package or installed package files. A detached
# test directory may explicitly provide TRAM_TEST_RUNTIME_DIR/TRAM_TEST_SHARE.
import os
STAGING=Path(os.environ.get('TRAM_TEST_RUNTIME_DIR',str(Path(__file__).resolve().parents[1])))
if not (STAGING/'tram_odometry/node.py').exists():
    spec=importlib.util.find_spec('tram_odometry')
    assert spec and spec.submodule_search_locations, 'Install runtime or set TRAM_TEST_RUNTIME_DIR'
    STAGING=Path(next(iter(spec.submodule_search_locations))).parent
SHARE=Path(os.environ.get('TRAM_TEST_SHARE',str(STAGING)))

def header():return NS(stamp=NS(sec=0,nanosec=0),frame_id='')

class Velocity:
    def __init__(self):self.header=header();self.velocity=0.

class Odom:
    def __init__(self):
        self.header=header();self.child_frame_id=''
        self.pose=NS(pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.)),covariance=[])
        self.twist=NS(twist=NS(linear=NS(x=0.,y=0.,z=0.)),covariance=[])

class Diagnostic:
    def __init__(self):self.header=header();self.status=[]

class Status:
    WARN=1;OK=0

class Publisher:
    def __init__(self):self.messages=[]
    def publish(self,value):self.messages.append(value)

class FakeNode:
    overrides={}
    def __init__(self,name):
        self.parameters={};self.created=[];self.destroyed=[];self.clock_ns=1_000_000_000
        self.clock=NS(now=lambda:NS(nanoseconds=self.clock_ns),create_jump_callback=lambda *a,**k:None)
    def declare_parameter(self,name,value):self.parameters[name]=self.overrides.get(name,value)
    def has_parameter(self,name):return name in self.parameters
    def get_parameter(self,name):return NS(value=self.parameters[name])
    def create_publisher(self,*a):return Publisher()
    def create_subscription(self,typ,topic,callback,qos):
        item=NS(type=typ,topic=topic,callback=callback);self.created.append(item);return item
    def destroy_subscription(self,item):self.destroyed.append(item)
    def create_timer(self,period,callback):return NS(period=period,callback=callback)
    def get_clock(self):return self.clock
    def get_logger(self):return NS(info=lambda x:None,warning=lambda x:None)

@pytest.fixture
def node_module(monkeypatch):
    modules={
        'ament_index_python.packages':{'get_package_share_directory':lambda name:str(SHARE)},
        'rclpy':{},'rclpy.node':{'Node':FakeNode},
            'rclpy.executors':{'ExternalShutdownException':type('ExternalShutdownException',(Exception,),{})},
        'rclpy.clock':{'JumpThreshold':lambda **k:NS(**k)},
        'rclpy.duration':{'Duration':lambda **k:NS(**k)},
        'rclpy.qos':{'QoSProfile':lambda **k:NS(**k),'ReliabilityPolicy':NS(BEST_EFFORT=1),'HistoryPolicy':NS(KEEP_LAST=1),'DurabilityPolicy':NS(VOLATILE=1)},
        'diagnostic_msgs.msg':{'DiagnosticArray':Diagnostic,'DiagnosticStatus':Status,'KeyValue':lambda **k:NS(**k)},
        'nav_msgs.msg':{'Odometry':Odom},'sensor_msgs.msg':{'NavSatFix':type('NavSatFix',(),{})},
        'geometry_msgs.msg':{'TwistStamped':type('TwistStamped',(),{})},
        'tram_vehicle_msgs.msg':{'DriverControllerCommand':type('DriverControllerCommand',(),{}),'VelocitySensor':Velocity},
    }
    for key,values in modules.items():
        if '.'in key:
            root=key.split('.')[0]
            if root not in sys.modules:monkeypatch.setitem(sys.modules,root,ModuleType(root))
        mod=ModuleType(key);mod.__dict__.update(values);monkeypatch.setitem(sys.modules,key,mod)
    monkeypatch.syspath_prepend(str(STAGING))
    spec=importlib.util.spec_from_file_location('contract_odometry',STAGING/'tram_odometry/__init__.py',submodule_search_locations=[str(STAGING/'tram_odometry')])
    package=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,package);spec.loader.exec_module(package)
    spec=importlib.util.spec_from_file_location('contract_odometry.node',STAGING/'tram_odometry/node.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(FakeNode,'overrides',{})
    return module

def test_topics_preserve_master_alias_and_two_receivers_have_distinct_sources(node_module):
    node=node_module.OdometryNode();calls=[]
    node.bridge.receive_fix=lambda *a,**k:calls.append((a,k))
    subscriptions={s.topic:s for s in node.created}
    assert len(subscriptions)==6
    fix=NS(header=header(),latitude=55.,longitude=37.,altitude=170.,status=NS(status=2),position_covariance=[0.]*9,position_covariance_type=0)
    fix.header.stamp.sec=1
    subscriptions['/sensing/gnss/master/fix'].callback(fix)
    subscriptions['/sensing/gnss/rover/fix'].callback(fix)
    assert [c[1]['source']for c in calls]==['master','rover']
    assert all(c[1]['clock_ns']==node.clock_ns and c[1]['covariance_type']==0 for c in calls)
    fix.position_covariance[4]=100.;fix.position_covariance_type=2
    node.on_fix(fix)
    assert calls[-1][1]['variance_m2']==100.
    node.clock_ns=0;node.on_fix(fix)
    assert len(calls)==3

def test_malformed_known_covariance_cannot_disappear_inside_max(node_module):
    node=node_module.OdometryNode();calls=[]
    node.bridge.receive_fix=lambda *a,**k:calls.append((a,k))
    fix=NS(header=header(),latitude=55.,longitude=37.,altitude=170.,status=NS(status=2),position_covariance=[0.]*9,position_covariance_type=2)
    fix.position_covariance[8]=float('nan')
    node.on_fix(fix)
    assert math.isnan(calls[-1][1]['variance_m2'])
    fix.position_covariance[8]=-1.
    node.on_fix(fix)
    assert math.isnan(calls[-1][1]['variance_m2'])

def test_enu_vector_is_passed_with_receipt_clock_without_velocity_covariance_invention(node_module):
    node=node_module.OdometryNode();calls=[]
    node.bridge.receive_velocity=lambda *a,**k:calls.append((a,k))
    message=NS(header=header(),twist=NS(linear=NS(x=3.,y=4.,z=.2)))
    message.header.stamp.sec=1
    node.on_gnss_velocity(message)
    assert calls==[((1_000_000_000,3.,4.,.2),{'source':'master','clock_ns':1_000_000_000})]

@pytest.mark.parametrize('index,value',[(i,float('nan'))for i in range(9)]+[(i,-1.)for i in (0,4,8)]+[(i,float('inf'))for i in (0,4,8)])
def test_invalid_covariance_does_not_consume_same_stamp(node_module,index,value):
    node=node_module.OdometryNode()
    fix=NS(header=header(),latitude=55.,longitude=37.,altitude=170.,status=NS(status=2),position_covariance=[0.]*9,position_covariance_type=2)
    fix.header.stamp.sec=1;fix.position_covariance[index]=value
    node.on_fix(fix)
    assert node.bridge.fusion.counts['received_fix']==0
    assert not node.bridge.fusion.seen and not node.bridge.fusion.receiver_pending
    fix.position_covariance=[0.]*9
    node.on_fix(fix)
    assert node.bridge.fusion.counts['received_fix']==1
    assert ('fix','master',1_000_000_000) in node.bridge.fusion.seen

def test_negative_off_diagonal_known_covariance_is_allowed(node_module):
    node=node_module.OdometryNode()
    fix=NS(header=header(),latitude=55.,longitude=37.,altitude=170.,status=NS(status=2),position_covariance=[9.,-1.,0.,-1.,9.,0.,0.,0.,9.],position_covariance_type=2)
    fix.header.stamp.sec=1
    node.on_fix(fix)
    assert node.bridge.fusion.counts['received_fix']==1

def test_relative_mode_has_no_gnss_and_publishes_mps_and_named_frame(node_module,monkeypatch):
    monkeypatch.setattr(FakeNode,'overrides',{'start_mode':'relative'})
    node=node_module.OdometryNode()
    assert len(node.created)==3 and not node.gnss_subscriptions
    wheel=Velocity();wheel.header.stamp.sec=1;wheel.velocity=18.
    node.on_wheel('front',wheel);node.on_wheel('rear',wheel)
    node.on_command(NS(header=wheel.header,position=0))
    assert node.velocity_pub.messages[-1].velocity==5.
    output=node.position_pub.messages[-1]
    assert output.header.frame_id=='odom_relative' and output.child_frame_id=='base_link'
    assert output.twist.twist.linear.x==5. and len(output.pose.covariance)==36
    diagnostics={v.key:v.value for v in node.diagnostic_pub.messages[-1].status[0].values}
    assert diagnostics['absolute_pose_valid']=='False'
    assert diagnostics['late_position_tracking_enabled']=='False'
    assert diagnostics['position_reference_point']=='master_gnss_projected_to_map'
    assert diagnostics['surveyed_body_transform_confirmed']=='False'

def test_clock_reset_replaces_state_without_duplicate_subscriptions(node_module):
    node=node_module.OdometryNode();old=node.bridge
    node.on_clock_jump(None);node._prepare_callback()
    assert node.bridge is not old and node.bridge.estimator.stamp_ns==0
    assert len(node.created)==6

def test_zero_window_disables_gnss_explicitly(node_module,monkeypatch):
    monkeypatch.setattr(FakeNode,'overrides',{'initialization_window_s':0.})
    node=node_module.OdometryNode()
    assert len(node.created)==3 and node.bridge.gnss_closed

def straight_bridge(node,module,*,start_mode='gnss',chainage=0.):
    from tram_position.route_localizer import DirectedRoute,RouteLocalizer
    from tram_estimator import Drive
    class Cartesian:
        yaw_map_to_enu_rad=0.
        def gnss_to_map(self,lat,lon,alt):return lat*1000,lon*1000,alt
    class Model:
        def predict(self,command,speed):return Drive(0.,.1)
    loc=RouteLocalizer(Cartesian(),(DirectedRoute('east',[(0,0,0),(100,0,0)]),))
    cfg=replace(node.bridge_config,wheel_input_unit='mps',start_mode=start_mode,route_name='east',start_chainage_m=chainage)
    node.bridge=module.Bridge(Model(),loc,cfg);node.bridge_config=cfg

def drive_tick(node,stamp,speed):
    node.clock_ns=stamp+100_000_000_000
    h=header();h.stamp.sec,h.stamp.nanosec=divmod(stamp,1_000_000_000)
    node.on_wheel('front',NS(header=h,velocity=speed));node.on_wheel('rear',NS(header=h,velocity=speed))
    node.on_command(NS(header=h,position=0))

def fix_at(node,stamp,x):
    h=header();h.stamp.sec,h.stamp.nanosec=divmod(stamp,1_000_000_000)
    node.clock_ns=stamp+100_000_000_000
    node.on_fix(NS(header=h,latitude=x/1000,longitude=0.,altitude=0.,status=NS(status=2),position_covariance=[0.]*9,position_covariance_type=0))

def test_actual_node_late_first_transition_and_later_freeze(node_module):
    node=node_module.OdometryNode();straight_bridge(node,node_module)
    for i in range(81):drive_tick(node,1_000_000_000+i*50_000_000,0.)
    old=node.position_pub.messages[-1]
    assert old.header.frame_id=='odom_relative'
    assert len(node.gnss_subscriptions)==3
    for i in range(81,161):
        stamp=1_000_000_000+i*50_000_000
        fix_at(node,stamp,5.)
        drive_tick(node,stamp,0.)
    assert node.position_pub.messages[-1].header.frame_id=='pathgraph'
    assert node.position_pub.messages[-1].pose.pose.position.x==5.
    fix_at(node,9_050_000_000,40.);drive_tick(node,9_050_000_000,0.)
    assert node.position_pub.messages[-1].pose.pose.position.x==5.
    assert old.header.frame_id=='odom_relative'
    assert len(node.gnss_subscriptions)==3 and len(node.created)==6

def test_actual_node_map_exit_frame_is_immediate(node_module):
    node=node_module.OdometryNode();straight_bridge(node,node_module,start_mode='route_chainage',chainage=95.)
    for i in range(61):
        drive_tick(node,1_000_000_000+i*50_000_000,5.)
        output=node.position_pub.messages[-1]
        if i==0:assert output.header.frame_id=='pathgraph'
        if not node.bridge.last_output.diagnostics['absolute_pose_valid']:
            assert output.header.frame_id=='odom_relative'
    assert output.header.frame_id=='odom_relative'
    assert node.bridge.last_output.estimate.distance_m>10.
