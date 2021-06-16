from typing import List, Dict, Callable, Any, Optional, Tuple
import math
import threading
import time
import struct

import numpy as np

import rclpy
import rclpy.node
import rclpy.qos
from rcl_interfaces.msg import (SetParametersResult, ParameterDescriptor, FloatingPointRange,
                                IntegerRange, ParameterType)
from std_msgs.msg import Empty, UInt8, Bool
from sensor_msgs.msg import Image, CompressedImage, Imu
# Publish camera info to rectify camera images
# from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Twist, TransformStamped, Point, Vector3
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros

from tello_msgs.msg import TelloStatus
from tellopy._internal import tello, event
from tellopy._internal.utils import byte, byte_to_hexstring
from tellopy._internal.protocol import (Packet, FlightData, STICK_CMD, VIDEO_START_CMD, EMERGENCY_CMD,
                                        FLIP_CMD)

# To load camera calibration from '.yaml' format
# TODO(Jerome): not avaliable in ROS2
# import camera_info_manager as cim


# Add 'EVENT_VIDEO_FRAME_H264' to collect h264 images


Event = Any
Sender = Any
Data = Any


class DynamicParam:

    def __init__(self, node: rclpy.node.Node, name: str, param_type: ParameterType, value: Any,
                 description: str = "", min_value: Any = None, max_value: Any = None,
                 cb: Callable[[Any], None] = lambda _: None) -> None:
        self.name = name
        self.description = description
        self.cb = cb
        node.dynamic_param[name] = self
        kwargs: Dict[str, Any] = {}
        if param_type == ParameterType.PARAMETER_INTEGER:
            if type(min_value) is int and type(max_value) is int:
                kwargs['integer_range'] = [
                    IntegerRange(from_value=min_value, to_value=max_value, step=1)]
        if param_type == ParameterType.PARAMETER_DOUBLE:
            if type(min_value) is float and type(max_value) is float:
                kwargs['floating_point_range'] = [
                    FloatingPointRange(from_value=min_value, to_value=max_value, step=0.0)]
        desc = ParameterDescriptor(name=self.name, type=param_type, read_only=False, **kwargs)
        self.value = node.declare_parameter(self.name, value, desc).value

    def set_value(self, value: Any, force: bool = False) -> None:
        if value != self.value or force:
            self.value = value
            self.cb(value)


class TelloNode(rclpy.node.Node, tello.Tello):  # type: ignore

    # Add event variable(s) to leave 'TelloPy' package untouched (Jordy)
    EVENT_VIDEO_FRAME_H264 = event.Event('video frame h264')

    def __init__(self) -> None:
        rclpy.node.Node.__init__(self, "tello_driver_node")
        self.flying = False
        self.dynamic_param: Dict[str, DynamicParam] = {}
        tello.log = self.get_logger()
        tello.log.log_level = 99
        tello.Tello.__init__(self, port=9000)

        state_qos = rclpy.qos.QoSProfile(
            depth=1,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.connection_pub = self.create_publisher(Bool, 'connected', state_qos)
        self.local_cmd_client_port = self.declare_parameter('local_cmd_client_port', 8890).value
        self.local_vid_server_port = self.declare_parameter('local_vid_server_port', 6038).value
        self.tello_ip: str = self.declare_parameter('tello_ip', '192.168.10.1').value
        self.tello_cmd_server_port: int = self.declare_parameter('tello_cmd_server_port', 8889).value
        self.connect_timeout_sec: float = self.declare_parameter('connect_timeout_sec', 10.0).value
        self.stream_h264_video: bool = self.declare_parameter('stream_h264_video', False).value
        self.bridge = CvBridge()
        self.frame_thread = None

        # Video rate options
        # 0: auto, 1: 1.0Mb/s, 2: 1.5Mb/s, 3:  2.0Mb/s, 4: 2.5Mb/s
        self.fixed_video_rate = DynamicParam(
            self, 'fixed_video_rate', ParameterType.PARAMETER_INTEGER, 0,
            min_value=0, max_value=4,
            description="video rate options",
            cb=self.set_video_encoder_rate)
        # Rate for regularly requesting SPS data from drone (0: disabled)
        self.video_req_sps_hz = DynamicParam(
            self, 'video_req_sps_hz', ParameterType.PARAMETER_DOUBLE, 0.5,
            min_value=0.0, max_value=4.0,
            description="Rate for regularly requesting SPS data from drone (0: disabled)",
            cb=self.set_video_req_sps_hz)
        self.altitude_limit = DynamicParam(
            self, 'altitude_limit', ParameterType.PARAMETER_INTEGER, 10,
            min_value=1, max_value=100,
            description="",
            cb=self.set_alt_limit)
        # Limit attitude of Tello
        self.attitude_limit = DynamicParam(
            self, 'attitude_limit', ParameterType.PARAMETER_INTEGER, 15,
            min_value=15, max_value=25,
            description="Limit attitude of Tello",
            cb=self.set_att_limit)
        # Set low battery threshold of Tello
        self.low_bat_threshold = DynamicParam(
            self, 'low_bat_threshold', ParameterType.PARAMETER_INTEGER, 7,
            min_value=1, max_value=100,
            description="Set low battery threshold of Tello",
            cb=self.set_low_bat_threshold)
        # Scale (down) vel_cmd value
        self.vel_cmd_scale = DynamicParam(
            self, 'vel_cmd_scale', ParameterType.PARAMETER_DOUBLE, 0.5,
            min_value=0.01, max_value=1.0,
            description="Scale (down) vel_cmd value")

        tf_prefix: str = self.declare_parameter('tf_prefix', '').value
        if tf_prefix and tf_prefix[-1] != '/':
            tf_prefix += '/'
        self.frame = tf_prefix + self.declare_parameter('frame', 'base_link').value
        self.odom_frame = tf_prefix + self.declare_parameter('odom_frame', 'odom').value
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(node=self)
        self.fast_mode = False
        self.connection_pub.publish(Bool(data=False))

        # TODO(Jerome): port to ROS2
        # rospy.on_shutdown(self.cb_shutdown)

        # Setup topics and services
        # NOTE: ROS interface deliberately made to resemble bebop_autonomy
        self.pub_status = self.create_publisher(TelloStatus, 'status', state_qos)
        if self.stream_h264_video:
            self.pub_image_h264 = self.create_publisher(
                CompressedImage, 'image_raw/h264', 1)
        else:
            self.pub_image_raw = self.create_publisher(
                Image, 'camera/image_raw', 1)

        self.sub_takeoff = self.create_subscription(Empty, 'takeoff', self.cb_takeoff, 1)
        self.sub_manual_takeoff = self.create_subscription(
            Empty, 'manual_takeoff', self.cb_manual_takeoff, 1)
        self.sub_throw_takeoff = self.create_subscription(
            Empty, 'throw_takeoff', self.cb_throw_takeoff, 1)
        self.sub_land = self.create_subscription(Empty, 'land', self.cb_land, 1)
        self.sub_palm_land = self.create_subscription(Empty, 'palm_land', self.cb_palm_land, 1)
        self.sub_flattrim = self.create_subscription(Empty, 'flattrim', self.cb_flattrim, 1)
        self.sub_flip = self.create_subscription(UInt8, 'flip', self.cb_flip, 1)
        self.sub_cmd_vel = self.create_subscription(Twist, 'cmd_vel', self.cb_cmd_vel, 1)
        self.sub_fast_mode = self.create_subscription(Empty, 'fast_mode', self.cb_fast_mode, 1)

        self.subscribe(self.EVENT_FLIGHT_DATA, self.cb_status_log)

        # Reconstruction H264 video frames
        self.prev_seq_id: Optional[int] = None
        self.seq_block_count = 0

        # Height from EVENT_FLIGHT_DATA more accurate than MVO (monocular visual odometry)
        self.height = 0.0

        # EVENT_LOG_DATA from 'TelloPy' package
        self.pub_odom = self.create_publisher(Odometry, 'odom', state_qos)
        self.pub_imu = self.create_publisher(Imu, 'imu', state_qos)
        self.position = (0.0, 0.0)
        self.delta_position: Optional[Tuple[float, float, float]] = None

        self.subscribe(self.EVENT_LOG_DATA, self.cb_data_log)

        self.sub_zoom = self.create_subscription(Empty, 'video_mode', self.cb_video_mode, 1)

        # TODO(Jerome): port to ROS2
        # calib_path = rospy.get_param('~camera_calibration', '')
        # self.caminfo = cim.loadCalibrationFile(calib_path, 'camera_front')
        # self.caminfo.header.frame_id = rospy.get_param('~camera_frame', rospy.get_namespace() + 'camera_front')
        # self.pub_caminfo = rospy.Publisher('camera/camera_info', CameraInfo, queue_size=1, latch=True)
        # self.pub_caminfo.publish(self.caminfo)

        self.sub_emergency = self.create_subscription(Empty, 'emergency', self.cb_emergency, 1)
        self.create_subscription(Point, 'set_position', self.has_updated_position, 1)

        self.add_on_set_parameters_callback(self.on_parameter_event)
        self.subscribe(self.EVENT_DISCONNECTED, self.cb_unconnected)
        self.subscribe(self.EVENT_CONNECTED, self.cb_connected)

        # Connect
        self.get_logger().info(f'Connecting to drone @ {self.tello_addr[0]}:{self.tello_addr[1]}')
        self.connect()

        if self.stream_h264_video:
            self.start_video()
            self.subscribe(self.EVENT_VIDEO_DATA, self.cb_video_data)
            self.subscribe(self.EVENT_VIDEO_FRAME_H264, self.cb_h264_frame)
        else:
            self.frame_thread = threading.Thread(target=self.framegrabber_loop)
            self.frame_thread.start()

        self.get_logger().info('Tello driver node ready')

    def notify_cmd_success(self, cmd: str, success: bool) -> None:
        if success:
            self.get_logger().info(f'{cmd} command executed')
        else:
            self.get_logger().warning(f'{cmd} command failed')

    def on_parameter_event(self, parameter_list: List[rclpy.Parameter]) -> SetParametersResult:
        req_sps_pps = False
        for param in parameter_list:
            dynamic_param = self.dynamic_param.get(param.name)
            if dynamic_param:
                dynamic_param.set_value(param.value, force=False)
            if param.name in ('video_req_sps_hz', 'fixed_video_rate'):
                req_sps_pps = True
        if req_sps_pps:
            self.send_req_video_sps_pps()
        return SetParametersResult(successful=True)

    def forward_param(self, param: rclpy.Parameter) -> None:
        (value, setter) = self.dynamic_param.get(param.name, (None, None))
        if value is None or setter is None:
            return
        if value != param.value:
            value = param.value
            self.dynamic_param[param.name] = (value, setter)
            setter(value)

    def has_updated_position(self, msg: Point) -> None:
        self.get_logger().info(f"Reset position to ({msg.x:.2f}, {msg.y:.2f}")
        self.delta_position = None
        self.position = (msg.x, msg.y)

    # Add 'Tello' compositions, leave 'TelloPy' package untouched (Jordy)

    def set_fast_mode(self, enabled: bool) -> None:
        self.fast_mode = enabled

    def reset_cmd_vel(self) -> None:
        self.left_x = 0.
        self.left_y = 0.
        self.right_x = 0.
        self.right_y = 0.
        self.fast_mode = False

    # scaling for velocity command
    def __scale_vel_cmd(self, cmd_val: float) -> float:
        return self.vel_cmd_scale.value * cmd_val

    # Override TelloPy '__send_stick_command' to add 'fast_mode' functionality
    #
    #    11 bits (-1024 ~ +1023) x 4 axis = 44 bits
    #    fast_mode takes 1 bit
    #    44+1 bits will be packed in to 6 bytes (48 bits)
    #     axis5      axis4      axis3      axis2      axis1
    #         |          |          |          |          |
    #             4         3         2         1         0
    #    98765432109876543210987654321098765432109876543210
    #     |       |       |       |       |       |       |
    #         byte5   byte4   byte3   byte2   byte1   byte0
    def _Tello__send_stick_command(self) -> bool:

        pkt = Packet(STICK_CMD, 0x60)
        axis1 = int(1024 + 660.0 * self.right_x) & 0x7ff
        axis2 = int(1024 + 660.0 * self.right_y) & 0x7ff
        axis3 = int(1024 + 660.0 * self.left_y) & 0x7ff
        axis4 = int(1024 + 660.0 * self.left_x) & 0x7ff
        axis5 = int(self.fast_mode) & 0x01
        self.log.debug("stick command: fast=%d yaw=%4d vrt=%4d pit=%4d rol=%4d" %
                       (axis5, axis4, axis3, axis2, axis1))

        packed = axis1 | (axis2 << 11) | (
            axis3 << 22) | (axis4 << 33) | (axis5 << 44)
        packed_bytes = struct.pack('<Q', packed)
        pkt.add_byte(byte(packed_bytes[0]))
        pkt.add_byte(byte(packed_bytes[1]))
        pkt.add_byte(byte(packed_bytes[2]))
        pkt.add_byte(byte(packed_bytes[3]))
        pkt.add_byte(byte(packed_bytes[4]))
        pkt.add_byte(byte(packed_bytes[5]))
        pkt.add_time()
        pkt.fixup()
        self.log.debug("stick command: %s" % byte_to_hexstring(pkt.get_buffer()))
        return self.send_packet(pkt)

    def manual_takeoff(self) -> bool:
        # Hold max 'yaw' and min 'pitch', 'roll', 'throttle' for several seconds
        self.set_pitch(-1)
        self.set_roll(-1)
        self.set_yaw(1)
        self.set_throttle(-1)
        self.fast_mode = False
        return self._Tello__send_stick_command()

    def cb_video_data(self, event: Event, sender: Sender, data: Data, **args: Any) -> None:
        now = time.time()
        # parse packet
        seq_id = byte(data[0])
        sub_id = byte(data[1])
        packet = data[2:]
        self.sub_last = False
        if sub_id >= 128:  # MSB asserted
            sub_id -= 128
            self.sub_last = True
        # associate packet to (new) frame
        if self.prev_seq_id is None or self.prev_seq_id != seq_id:
            # detect wrap-arounds
            if self.prev_seq_id is not None and self.prev_seq_id > seq_id:
                self.seq_block_count += 1
            self.frame_pkts = [None] * 128  # since sub_id uses 7 bits
            self.frame_t = now
            self.prev_seq_id = seq_id
        self.frame_pkts[sub_id] = packet

        # publish frame if completed
        if self.sub_last and all(self.frame_pkts[:sub_id + 1]):
            if isinstance(self.frame_pkts[sub_id], str):
                frame = ''.join(self.frame_pkts[:sub_id + 1])
            else:
                frame = b''.join(self.frame_pkts[:sub_id + 1])
            self._Tello__publish(
                event=self.EVENT_VIDEO_FRAME_H264,
                data=(frame, self.seq_block_count * 256 + seq_id, self.frame_t))

    def send_req_video_sps_pps(self) -> None:
        """Manually request drone to send an I-frame info (SPS/PPS) for video stream."""
        pkt = Packet(VIDEO_START_CMD, 0x60)
        pkt.fixup()
        return self.send_packet(pkt)

    def set_video_req_sps_hz(self, hz: float) -> None:
        """Internally sends a SPS/PPS request at desired rate; <0: disable."""
        if hz < 0:
            hz = 0.
        self.video_req_sps_hz.value = hz

    # emergency command
    def emergency(self) -> bool:
        """ Stop all motors """
        self.log.info('emergency (cmd=% seq=0x%04x)' % (EMERGENCY_CMD, self.pkt_seq_num))
        pkt = Packet(EMERGENCY_CMD)
        return self.send_packet(pkt)

    def flip(self, cmd: int) -> bool:
        """ tell drone to perform a flip in directions [0,8] """
        self.log.info('flip (cmd=0x%02x seq=0x%04x)' % (FLIP_CMD, self.pkt_seq_num))
        pkt = Packet(FLIP_CMD, 0x70)
        pkt.add_byte(cmd)
        pkt.fixup()
        return self.send_packet(pkt)

    # Additions to 'tello_driver_node' (Jordy) #####

    def cb_video_mode(self, msg: Empty) -> None:
        if not self.zoom:
            self.set_video_mode(True)
        else:
            self.set_video_mode(False)

    def cb_emergency(self, msg: Empty) -> None:
        success = self.emergency()
        self.notify_cmd_success('Emergency', success)

    # Modifications to 'tello_driver_node' (Jordy) #####

    def cb_status_log(self, event: Event, sender: Sender, data: FlightData, **args: Any) -> None:
        speed_horizontal_mps = math.sqrt(
            data.north_speed * data.north_speed + data.east_speed * data.east_speed) / 10.

        # TODO: verify outdoors: anecdotally, observed that:
        # data.east_speed points to South
        # data.north_speed points to East
        self.height = data.height / 10.
        self.flying = bool(data.em_sky)
        msg = TelloStatus(
            height_m=data.height / 10.,
            speed_northing_mps=-data.east_speed / 10.,
            speed_easting_mps=data.north_speed / 10.,
            speed_horizontal_mps=speed_horizontal_mps,
            speed_vertical_mps=-data.ground_speed / 10.,
            flight_time_sec=data.fly_time / 10.,
            imu_state=bool(data.imu_state),
            pressure_state=bool(data.pressure_state),
            down_visual_state=bool(data.down_visual_state),
            power_state=bool(data.power_state),
            battery_state=bool(data.battery_state),
            gravity_state=bool(data.gravity_state),
            wind_state=bool(data.wind_state),
            imu_calibration_state=data.imu_calibration_state,
            battery_percentage=data.battery_percentage,
            drone_fly_time_left_sec=data.drone_fly_time_left / 10.,
            drone_battery_left_sec=data.drone_battery_left / 10.,
            is_flying=bool(data.em_sky),
            is_on_ground=bool(data.em_ground),
            is_em_open=bool(data.em_open),
            is_drone_hover=bool(data.drone_hover),
            is_outage_recording=bool(data.outage_recording),
            is_battery_low=bool(data.battery_low),
            is_battery_lower=bool(data.battery_lower),
            is_factory_mode=bool(data.factory_mode),
            fly_mode=data.fly_mode,
            throw_takeoff_timer_sec=data.throw_fly_timer / 10.,
            camera_state=data.camera_state,
            electrical_machinery_state=data.electrical_machinery_state,
            front_in=bool(data.front_in),
            front_out=bool(data.front_out),
            front_lsc=bool(data.front_lsc),
            temperature_height_m=data.temperature_height / 10.,
            cmd_roll_ratio=self.right_x,
            cmd_pitch_ratio=self.right_y,
            cmd_yaw_ratio=self.left_x,
            cmd_vspeed_ratio=self.left_y,
            cmd_fast_mode=bool(self.fast_mode),
        )
        self.pub_status.publish(msg)

    def cb_data_log(self, event: Event, sender: Sender, data: Data, **args: Any) -> None:
        time_cb = self.get_clock().now().to_msg()
        # CHANGED Jerome: Check quaterion norm
        q = (data.imu.q0, data.imu.q1, data.imu.q2, data.imu.q3)
        q_norm = sum([v ** 2 for v in q])
        if abs(q_norm - 1.0) > 0.1:
            self.log.warning('Quaternion (%.2f, %.2f, %.2f, %.2f) is not valid' % q)
            return

        # CHANGED Jerome: Coordinate system comply now with ROS conventions (... almost, see below)
        odom_msg = Odometry()
        odom_msg.header.stamp = time_cb
        odom_msg.header.frame_id = self.odom_frame
        # CHANGED Jerome: twist linear is in odom frame.
        # It would be better to output it in body frame.
        odom_msg.child_frame_id = self.odom_frame
        # CHANGED Jerome: VO is garbage when the drone is on the ground,
        # which we can detect as the values are very small. In tha case,
        # we output the prev. valid position (and zero linear velocity)
        if abs(data.mvo.pos_z) < 1e-2:
            # self.log.info("z not valid %.3f %.3f %.3f (%d)" % (data.mvo.pos_x, data.mvo.pos_y, data.mvo.pos_z, self.flying))
            # Visual odometry cannot be trusted:
            x, y = self.position
            z = self.height  # or even = 0.0
            if self.flying:
                return
            self.delta_position = None
            valid = False
        else:
            if self.delta_position is None:
                self.delta_position = (data.mvo.pos_x - self.position[0],
                                       -data.mvo.pos_y - self.position[1],
                                       -data.mvo.pos_z - self.height - 0.3)
            x = data.mvo.pos_x - self.delta_position[0]
            y = -data.mvo.pos_y - self.delta_position[1]
            # Height from MVO received as negative distance to floor
            z = -data.mvo.pos_z - self.delta_position[2]
            self.position = (x, y)
            valid = True
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = z
        odom_msg.pose.pose.orientation.w = data.imu.q0
        odom_msg.pose.pose.orientation.x = data.imu.q1
        odom_msg.pose.pose.orientation.y = -data.imu.q2
        odom_msg.pose.pose.orientation.z = -data.imu.q3
        # Linear speeds from MVO received in dm/sec
        if valid:
            odom_msg.twist.twist.linear.x = data.mvo.vel_x / 10.0
            odom_msg.twist.twist.linear.y = -data.mvo.vel_y / 10.0
            odom_msg.twist.twist.linear.z = -data.mvo.vel_z / 10.0
        odom_msg.twist.twist.angular.x = data.imu.gyro_x
        odom_msg.twist.twist.angular.y = -data.imu.gyro_y
        odom_msg.twist.twist.angular.z = -data.imu.gyro_z

        self.pub_odom.publish(odom_msg)

        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation

        # CHANGED Jerome: broadcasted odom -> base_link tranform

        t = TransformStamped()
        t.header.stamp = time_cb
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.frame
        t.transform.translation = Vector3(x=p.x, y=p.y, z=p.z)
        t.transform.rotation = q
        self.tf_broadcaster.sendTransform(t)

        imu_msg = Imu()
        imu_msg.header.stamp = time_cb
        imu_msg.header.frame_id = self.frame

        imu_msg.orientation.w = data.imu.q0
        imu_msg.orientation.x = data.imu.q1
        imu_msg.orientation.y = -data.imu.q2
        imu_msg.orientation.z = -data.imu.q3
        imu_msg.angular_velocity.x = data.imu.gyro_x
        imu_msg.angular_velocity.y = -data.imu.gyro_y
        imu_msg.angular_velocity.z = -data.imu.gyro_z
        # CHANGED Jerome: now in m/s^2 with gravity. Should comply to ros conventions.
        # https://www.ros.org/reps/rep-0145.html
        G = 9.81
        imu_msg.linear_acceleration.x = data.imu.acc_x * G
        imu_msg.linear_acceleration.y = -data.imu.acc_y * G
        imu_msg.linear_acceleration.z = -data.imu.acc_z * G

        self.pub_imu.publish(imu_msg)

    def cb_cmd_vel(self, msg: Twist) -> None:
        self.set_pitch(self.__scale_vel_cmd(msg.linear.x))
        self.set_roll(self.__scale_vel_cmd(-msg.linear.y))
        self.set_yaw(self.__scale_vel_cmd(-msg.angular.z))
        self.set_throttle(self.__scale_vel_cmd(msg.linear.z))

    def cb_flip(self, msg: UInt8) -> None:
        if msg.data < 0 or msg.data > 7:  # flip integers between [0,7]
            rclpy.get_logger().warning(f'Invalid flip direction: {msg.data}')
            return
        success = self.flip(msg.data)
        self.notify_cmd_success(f'Flip {msg.data}', success)

    def cb_shutdown(self) -> None:
        self.connection_pub.publish(Bool(data=False))
        self.quit()
        if self.frame_thread is not None:
            self.frame_thread.join()
        rclpy.get_logger().info("cb_shutdown called")

    def cb_h264_frame(self, event: Event, sender: Sender, data: Data, **args: Any) -> None:
        frame, seq_id, frame_secs = data
        # self.get_logger().info(f"frame_secs {frame_secs}")
        stamp = rclpy.time.Time(nanoseconds=int(frame_secs * 1e9))
        pkt_msg = CompressedImage()
        # pkt_msg.header.frame_id = self.caminfo.header.frame_id
        pkt_msg.header.stamp = stamp.to_msg()
        pkt_msg.data = frame
        self.pub_image_h264.publish(pkt_msg)

        # TODO(Jerome): port to ROS2
        # self.caminfo.header.seq = seq_id
        # self.caminfo.header.stamp = rospy.Time.from_sec(frame_secs)
        # self.pub_caminfo.publish(self.caminfo)

    def framegrabber_loop(self) -> None:
        import av  # Import here as 'hack' to prevent troublesome install of PyAV when not used
        # Repeatedly try to connect
        vs = self.get_video_stream()
        while self.state != self.STATE_QUIT:
            try:
                container = av.open(vs)
                break
            except BaseException as err:
                self.get_logger().error('fgrab: pyav stream failed - %s' % str(err))
                time.sleep(1.0)

        # Once connected, process frames till drone/stream closes
        while self.state != self.STATE_QUIT:
            try:
                # vs blocks, dies on self.stop
                for frame in container.decode(video=0):
                    img = np.array(frame.to_image())
                    try:
                        img_msg = self.bridge.cv2_to_imgmsg(img, 'rgb8')
                        # img_msg.header.frame_id = self.caminfo.header.frame_id
                    except CvBridgeError as err:
                        self.get_logger().error(f'fgrab: cv bridge failed - {err}')
                        continue
                    self.pub_image_raw.publish(img_msg)
                    # self.pub_caminfo.publish(self.caminfo)
                break
            except BaseException as err:
                self.get_logger().error(f'fgrab: pyav decoder failed - {err}')

    def cb_takeoff(self, msg: Empty) -> None:
        success = self.takeoff()
        self.notify_cmd_success('Takeoff', success)

    def cb_manual_takeoff(self, msg: Empty) -> None:
        success = self.manual_takeoff()
        self.notify_cmd_success('Manual takeoff', success)

    def cb_throw_takeoff(self, msg: Empty) -> None:
        success = self.throw_and_go()
        if success:
            self.get_logger().info('Drone set to auto-takeoff when thrown')
        else:
            self.get_logger().warning('ThrowTakeoff command failed')

    def cb_land(self, msg: Empty) -> None:
        success = self.land()
        self.notify_cmd_success('Land', success)

    def cb_palm_land(self, msg: Empty) -> None:
        success = self.palm_land()
        self.notify_cmd_success('PalmLand', success)

    def cb_flattrim(self, msg: Empty) -> None:
        self.get_logger().warning('Flattrim not implemented')
        # success = self.flattrim()
        # notify_cmd_success('FlatTrim', success)

    def cb_fast_mode(self, msg: Empty) -> None:
        if self.fast_mode:
            self.set_fast_mode(False)
        elif not self.fast_mode:
            self.set_fast_mode(True)

    def cb_unconnected(self, event: Event, sender: Sender, data: Data, **args: Any) -> None:
        self.get_logger().info('Disconnected from drone')
        self.connection_pub.publish(Bool(data=False))

    def cb_connected(self, event: Event, sender: Sender, data: Data, **args: Any) -> None:
        self.get_logger().info('Connected to drone')
        for param in self.dynamic_param.values():
            param.set_value(param.value, force=True)
        self.connection_pub.publish(Bool(data=True))


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node = TelloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
