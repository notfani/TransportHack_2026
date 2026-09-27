"""Launch the integrated estimator and optional bag with explicit replay controls."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def optional_bag(context):
    bag = LaunchConfiguration("bag").perform(context)
    play = LaunchConfiguration("play_bag").perform(context).lower() in ("true", "1")
    if not bag or not play:
        sim = LaunchConfiguration("use_sim_time").perform(context).lower() in ("true", "1")
        if sim:
            return [LogInfo(msg="No bag player started. Waiting for external /clock and /vehicle/* inputs. "
                                "For replay pass bag:=/absolute/path/to/bag; a separate player needs --clock 100.")]
        return [LogInfo(msg="Live mode: waiting for external /vehicle/* inputs.")]
    return [ExecuteProcess(
        cmd=["ros2", "bag", "play", bag, "--clock", "100",
             "--delay", LaunchConfiguration("bag_delay_sec").perform(context),
             "--rate", LaunchConfiguration("bag_rate").perform(context),
             "--start-offset", LaunchConfiguration("bag_start_offset").perform(context),
             "--disable-keyboard-controls"], output="screen")]


def generate_launch_description():
    default_config = str(Path(get_package_share_directory("tram_odometry")) /
                         "config" / "params.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_config),
        DeclareLaunchArgument("params", default_value=LaunchConfiguration("params_file")),
        DeclareLaunchArgument("bag", default_value=""),
        DeclareLaunchArgument("play_bag", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("require_sim_time", default_value="true"),
        DeclareLaunchArgument("bag_delay_sec", default_value="3.0"),
        DeclareLaunchArgument("bag_rate", default_value="1.0"),
        DeclareLaunchArgument("bag_start_offset", default_value="0.0"),
        Node(package="tram_odometry", executable="odometry_node", name="tram_odometry",
             output="screen", parameters=[
                 LaunchConfiguration("params"),
                 {"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
                  "require_sim_time": ParameterValue(LaunchConfiguration("require_sim_time"), value_type=bool)}]),
        OpaqueFunction(function=optional_bag),
    ])
