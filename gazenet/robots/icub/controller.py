import time
import pexpect
from typing import Literal
import numpy as np

try:
    import yarp
    _yarp = yarp
except:
    _yarp = None


from gazenet.utils.registrar import *
from gazenet.utils.helpers import cartesian_to_spherical

if _yarp is not None:
    _yarp.Network.init()


@RobotControllerRegistrar.register
class ICubController(object):
    def __init__(self, simulation=False, real=True, ikingaze=False,
                 control_object="eyes", limiting_consts_xy=(0.3, 0.3)):
        """
        ICub motor controller
        :param simulation: boolean enabling simulation
        :param real: boolean enabling real robot actuation
        :param control_object: string "eyes"- move eyes; "head": move head; "head+eyes" or "eyes+head" move both
        :param limiting_consts_xy: float representing the limiting factor for looking within specified region

        """
        self.simulation = simulation
        self.real = real

        # prepare a property object
        if simulation:
            sim_props = yarp.Property()
            sim_props.put("device", "remote_controlboard")
            sim_props.put("local", "/client/head")
            sim_props.put("remote", "/icubSim/head")

            # create remote driver
            self._sim_head_driver = yarp.PolyDriver(sim_props)
            # query motor control interfaces
            self._sim_ipos = self._sim_head_driver.viewIPositionControl()
            self._sim_ivel = self._sim_head_driver.viewIVelocityControl()
            self._sim_ienc = self._sim_head_driver.viewIEncoders()

        if real:
            real_props = yarp.Property()
            real_props.put("device", "remote_controlboard")
            real_props.put("local", "/client/head")
            real_props.put("remote", "/icub/head")

            # create remote driver
            self._real_head_driver = yarp.PolyDriver(real_props)
            # query motor control interfaces
            self._real_ipos = self._real_head_driver.viewIPositionControl()
            self._real_ivel = self._real_head_driver.viewIVelocityControl()
            self._real_ienc = self._real_head_driver.viewIEncoders()

        if control_object == "head+eyes" or control_object == "eyes+head":
            assert ikingaze,  "Set ikingaze=True in order to move eyes and head simultaneously"
        self.ikingaze = ikingaze
        self.control_object = control_object
        self.limiting_consts_xy = limiting_consts_xy

        self.sim_init_pos = None
        self.real_init_pos = None

        # retrieve number of joints
        if simulation:
            self._sim_num_jnts = self._sim_ipos.getAxes()
            self._sim_encs = yarp.Vector(self._sim_num_jnts)
            # read encoders
            self._sim_ienc.getEncoders(self._sim_encs.data())
            print('Controlling', self._sim_num_jnts, 'joints')
        if real:
            self._real_num_jnts = self._real_ipos.getAxes()
            self._real_encs = yarp.Vector(self._real_num_jnts)
            # read encoders
            self._real_ienc.getEncoders(self._real_encs.data())
            print('Controlling', self._real_num_jnts, 'joints')

        ##################################GAZECONTROLLER#######################################
        if ikingaze:
            self._gaze_encs = yarp.Vector(3, 0.0)
            props_gaze = yarp.Property()
            props_gaze.clear()
            props_gaze.put("device", "gazecontrollerclient")
            props_gaze.put("remote", "/iKinGazeCtrl")
            props_gaze.put("local", "/client/gaze")
            #
            self._gaze_driver = yarp.PolyDriver(props_gaze)
            self._igaze = self._gaze_driver.viewIGazeControl()
            self._igaze.setStabilizationMode(True)
            self._igaze.setNeckTrajTime(0.8)
            self._igaze.setEyesTrajTime(0.5)

    def waitfor_gaze(self, reset=True):
        if self.ikingaze:
            # self._igaze.clearNeckPitch()
            # self._igaze.clearNeckRoll()
            # self._igaze.clearNeckYaw()
            # self._igaze.clearEyes()
            if reset:
                self._igaze.lookAtAbsAngles(self._gaze_encs)
            self._igaze.waitMotionDone(timeout=2.0)
        else:
            if self.simulation:
                if reset:
                    self._sim_ipos.positionMove(self._sim_encs.data())
                if not self.real:
                    while not self._sim_ipos.checkMotionDone():
                        pass
            if self.real:
                if reset:
                    self._real_ipos.positionMove(self._real_encs.data())
                if not self.simulation:
                    while not self._real_ipos.checkMotionDone():
                        pass
            if self.simulation and self.real:
                while not (self._sim_ipos.checkMotionDone() and self._real_ipos.checkMotionDone()):
                    pass

    def reset_gaze(self):
        self.waitfor_gaze(reset=True)

    def set_eyes_angle_degrees(self, pt):
        # eye gaze tilt
        if self.simulation:
            self.sim_init_pos = yarp.Vector(self._sim_num_jnts, self._sim_encs.data())
            self.sim_init_pos.set(3, self.sim_init_pos.get(3) + pt[1])  # eye gaze pan (always opposite)
            self.sim_init_pos.set(4, self.sim_init_pos.get(4) + pt[0])
            self.sim_init_pos.set(5, self.sim_init_pos.get(5) + 0)  # this seems to only influence the divergence between the eyes (always set to 0)
            self._sim_ipos.positionMove(self.sim_init_pos.data())
        if self.real:
            self.real_init_pos = yarp.Vector(self._real_num_jnts, self._real_encs.data())
            self.real_init_pos.set(3, self.real_init_pos.get(3) + pt[1])  # eye gaze pan (always opposite)
            self.real_init_pos.set(4, self.real_init_pos.get(4) + pt[0])
            self.real_init_pos.set(5, self.real_init_pos.get(5) + 0)  # this seems to only influence the divergence between the eyes (always set to 0)
            self._real_ipos.positionMove(self.real_init_pos.data())

    def set_head_angle_degrees(self, pt):
        if self.simulation:
            self.sim_init_pos = yarp.Vector(self._sim_num_jnts, self._sim_encs.data())
            self.sim_init_pos.set(0, self.sim_init_pos.get(0) + pt[1])  # tilt/pitch
            self.sim_init_pos.set(1, self.sim_init_pos.get(1) + 0)  # swing/roll
            self.sim_init_pos.set(2, self.sim_init_pos.get(2) + pt[0])  # pan/yaw
            self._sim_ipos.positionMove(self.sim_init_pos.data())
        if self.real:
            self.real_init_pos = yarp.Vector(self._real_num_jnts, self._real_encs.data())
            self.real_init_pos.set(0, self.real_init_pos.get(0) + pt[1])  # tilt/pitch
            self.real_init_pos.set(1, self.real_init_pos.get(1) + 0)  # swing/roll
            self.real_init_pos.set(2, self.real_init_pos.get(2) + pt[0])  # pan/yaw
            self._real_ipos.positionMove(self.real_init_pos.data())

    def set_head_eyes_angle_degrees(self, pt):
        if self.real:
            if self.real_init_pos is None:
                self.real_init_pos = yarp.Vector(3, self._gaze_encs.data())
            self.real_init_pos.set(0, pt[0])
            self.real_init_pos.set(1, pt[1])
            self.real_init_pos.set(2, 0.0)
            self._igaze.lookAtAbsAngles(self.real_init_pos)

    def gaze_at_screen(self, xy=(0, 0,), ):
        """
        Gaze at specific point in a normalized plane in front of the robot

        :param xy: tuple representing the x and y position limited to the range of -1 (bottom left) and 1 (top right)
        :return: None
        """
        # wait for the action to complete
        self.waitfor_gaze(reset=False)

        xy = np.array(xy) * np.array(self.limiting_consts_xy)  # limit viewing region
        ptr = cartesian_to_spherical((1, xy[0], -xy[1]))
        # initialize a new tmp vector identical to encs
        ptr_degrees = (np.rad2deg(ptr[0]), np.rad2deg(ptr[1]))

        if self.control_object == "head":
            self.set_head_angle_degrees(ptr_degrees)
        elif self.control_object == "eyes":
            self.set_eyes_angle_degrees(ptr_degrees)

        elif self.control_object == "head+eyes" or self.control_object == "eyes+head":
            self.set_head_eyes_angle_degrees(ptr_degrees)


@RobotControllerRegistrar.register
class ICubRpcController:
    """
    Class used to move the icub head and eyes. Can be used on real robot or in simulation. Also includes
    option of using the iKinGazeController. Utilizes the rpc command line interface.
    """
    def __init__(self, simulation=True, ikingaze=False, limiting_consts_xy=(0.3, 0.3)):
        self.ikingaze = ikingaze
        self.limiting_consts_xy = limiting_consts_xy
        client = "icubSim" if simulation else "icub"
        if ikingaze:
            self.client = pexpect.spawn(f'yarp rpc /iKinGazeCtrl/rpc')
            self.client.expect(">>")
        else:
            self.client = pexpect.spawn(f'yarp rpc /{client}/head/rpc:i')
            self.client.expect(">>")

    def move_joint(self, joint, angle):
        """
        Move specific joint to given absolute angle.
        @param joint: iCub head joint from 0-5 (see https://icub-tech-iit.github.io/documentation/icub_kinematics/icub-joints/icub-joints/)
        @param angle: Absolute angle in degrees
        """
        if not self.ikingaze:
            self.client.sendline(f"set pos {joint} {angle}")
            self.client.expect(">>")
        else:
            print("move_joint is only available when not using iKinGaze")

    def move_head(self, coords):
        if not self.ikingaze:
            self.client.sendline(f"set pos {0} {coords[0]}")
            self.client.expect(">>")
            self.client.sendline(f"set pos {1} {coords[1]}")
            self.client.expect(">>")
            self.client.sendline(f"set pos {2} {coords[2]}")
            self.client.expect(">>")
        else:
            print("move_head is only available when not using iKinGaze")

    def move_eyes(self, coords):
        if not self.ikingaze:
            self.client.sendline(f"set pos {3} {coords[0]}")
            self.client.expect(">>")
            self.client.sendline(f"set pos {4} {coords[1]}")
            self.client.expect(">>")
            self.client.sendline(f"set pos {5} {coords[2]}")
            self.client.expect(">>")
        else:
            print("move_eyes is only available when not using iKinGaze")

    def look_3d(self, coords):
        """
        Look to specific 3D coordinates using iKinGazeController.
        @param coords: coordinates in 3D space with robot head as reference frame.
        """
        if self.ikingaze:
            self.client.sendline(f"look 3D ({' '.join(coords)})")
            self.client.expect(">>")
        else:
            print("look_3D is only available when using iKinGaze")

    def look_angle(self, azi, ele, ver, mode: Literal["abs", "rel"] = "abs"):
        """
        Gaze at target specified in angular coordinates.
        @param azi: Aizmuth
        @param ele: Elevation
        @param ver: Eyes vergence
        @param mode: Angle mode. Can be one of "abs" or "rel"
        """
        if self.ikingaze:
            self.client.sendline(f"look ang ({' '.join([mode, azi, ele, ver])})")
            self.client.expect(">>")
        else:
            print("look_3D is only available when using iKinGaze")

    def gaze_at_screen(self, xy=(0, 0,), ):
        """
        Gaze at specific point in a normalized plane in front of the robot

        :param xy: tuple representing the x and y position limited to the range of -1 (bottom left) and 1 (top right)
        :return: None
        """
        xy = np.array(xy) * np.array(self.limiting_consts_xy)  # limit viewing region
        ptr = cartesian_to_spherical((1, xy[0], -xy[1]))
        ptr_degrees = (np.rad2deg(ptr[1]), np.rad2deg(ptr[0]), 0)
        self.move_eyes(ptr_degrees)

    def reset_gaze(self):
        self.move_eyes((0, 0, 0))
        self.move_head((0, 0, 0))


if __name__ == "__main__":
    # API tests
    control_object = "head"
    if control_object == "head+eyes" or control_object == "eyes+head":
        simulation = False
        real = True
        limiting_consts_xy = (0.35, 0.3)
        ikingaze = True
    else:
        simulation = True
        real = False
        limiting_consts_xy = (0.3, 0.3)
        ikingaze = False

    controller = ICubController(simulation=simulation, real=real,
                                ikingaze=ikingaze, control_object=control_object, limiting_consts_xy=limiting_consts_xy)
    time.sleep(0.1)
    controller.waitfor_gaze()
    for pos_x in np.linspace(-1, 1, 100):
        controller.gaze_at_screen((pos_x, 0))
        time.sleep(0.1)
    controller.waitfor_gaze()
    for pos_y in np.linspace(-1, 1, 100):
        controller.gaze_at_screen((0, pos_y))
        time.sleep(0.1)
    controller.waitfor_gaze()

    # RPC Controller tests
    simController = ICubRpcController(simulation=True)
    realController = ICubRpcController(simulation=False)

    for i in range(-10, 10, 1):
        simController.move_joint(3, i)
        realController.move_joint(3, i)
        time.sleep(1)
        print(i)