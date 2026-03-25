import os
from typing import List
import numpy as np
import xml.etree.ElementTree as ET

from urdf_types import Link, Joint, FixedJoint, RevoluteJoint
from config import RobotConfig
from rotation import rpy_to_mat,axis_angle_to_mat
from utils import str_to_np
from vis import Vis


class RobotModel:
    robot_cfg: RobotConfig
    links: List[Link]
    joints: List[Joint]

    def __init__(self, robot_cfg: RobotConfig):
        """
        Initialize the RobotModel with the given RobotConfig

        Parameters
        ----------
        robot_cfg : RobotConfig
            The configuration of the robot
        """
        self.robot_cfg = robot_cfg
        self.load_urdf(robot_cfg)

    def fk(self, qpos: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for all of the links.

        Here we assume the i-th joint has i-th link as parent
        and i+1-th link as child.

        In practice multiple joints can have a shared parent link.
        But dealing with those special cases is not required in the homework.

        The result is in robot frame, which means that the first link
        has (0, 0, 0) as translation and I as rotation matrix

        Here we assume each link's frame, except the first one,
        is as same as its parent joint's frame.

        See https://wiki.ros.org/urdf/XML/joint and urdf_types.py 
        for the definition of Link and Joint.

        You can import functions from rotation.py

        Parameters
        ----------
        qpos: np.ndarray
            The current joint angles with shape (J,)
            (which means its length is the number of revolute joints)

        Returns
        -------
        np.ndarray
            The poses of links with shape (L, 4, 4)
            (which means its length is the number of links)

        Note
        ----
        The 4*4 pose matrix combines translation and rotation.

        Its format is:
            R  t
            0  1
        where R is rotation matrix and t is translation vector
        """
        # ok , now we know that the structure of the robot is a chain 
        # the first link is fixed, and the i-th joint connects the i-th link and the (i+1)-th link 
        # we can compute the pose of each link iteratively

        poses = np.zeros((len(self.links), 4, 4))
        poses[0] = np.eye(4) # the first link is fixed at the origin 
        k = 0 # index for revolute joints
        for i in range(len(self.joints)):
            joint = self.joints[i]

            # this joint may be fixed or revolute 
            # for fixed joint, we can directly use the translation and rotation of its parent link 
            # for revolute joint, first find the axis and angle fo rotation , then compute the matrix 
            if isinstance(joint, FixedJoint):
                # the base pose of the child link is the same as the parent link 
                poses[i+1] = poses[i].copy()
                # though the fixed joint can't have any rotation, but compared with the parent link,
                # it has a fixed translation and rotation of the poses, so ...
                poses[i+1][:3, :3] = poses[i][:3, :3] @ joint.rot 
                poses[i+1][:3, 3] = poses[i][:3, 3] + poses[i][:3, :3] @ joint.trans
            elif isinstance(joint, RevoluteJoint):
                # the pose of the child link is the same as the parent link 
                poses[i+1] = poses[i].copy()
                # then apply the translation and rotation of the joint 
                # first compute the rotation matrix of this joint 
                axis = joint.axis 
                angle = qpos[k] 
                # we can use the function that we have already implemented to convert axis-angle to rotation matrix 
                rot = axis_angle_to_mat(axis * angle) # axis-angle to rotation matrix 
                poses[i+1][:3, :3] = poses[i][:3, :3] @ joint.rot @ rot
                poses[i+1][:3, 3] = poses[i][:3, 3] + poses[i][:3, :3] @ joint.trans    # here we need to notice that the joint.trans is the translation  related to the parent link, so 
                                                                                        # to get the translation of the chind link, we need to apply the rotation of the parent link to convert it to the world frame            
                k += 1
        return poses


    def load_urdf(self, robot_cfg: RobotConfig):
        """
        Load the URDF into this RobotModel

        Theoretically one can write a general code that load
        everything only from the URDF, but it will make the
        code too complex. Thus we read the joints' name and
        links' name from the RobotConfig instead.

        Parameters
        ----------
        robot_cfg : RobotConfig
            The configuration of the robot
        """
        self.links = [None for _ in robot_cfg.link_names]
        self.joints = [None for _ in robot_cfg.joint_names]
        tree = ET.parse(robot_cfg.urdf_path)
        root = tree.getroot()
        for child in root:
            if child.tag == "link":
                idx = robot_cfg.link_names.index(child.attrib["name"])
                self.links[idx] = Link(
                    name=child.attrib["name"],
                    visual_meshes=[
                        os.path.join(
                            os.path.dirname(robot_cfg.urdf_path), m.attrib["filename"]
                        )
                        for m in child.findall("./visual/geometry/mesh")
                    ],
                )
            elif child.tag == "joint":
                idx = robot_cfg.joint_names.index(child.attrib["name"])
                joint_type = child.attrib["type"]
                kwargs = dict(
                    name=child.attrib["name"],
                    trans=str_to_np(child.find("origin").attrib["xyz"]),
                    rot=rpy_to_mat(str_to_np(child.find("origin").attrib["rpy"])),
                )
                if joint_type == "fixed":
                    self.joints[idx] = FixedJoint(**kwargs)
                elif joint_type == "revolute":
                    self.joints[idx] = RevoluteJoint(
                        axis=str_to_np(child.find("axis").attrib["xyz"]),
                        lower_limit=float(child.find("limit").attrib["lower"]),
                        upper_limit=float(child.find("limit").attrib["upper"]),
                        **kwargs
                    )

    def vis(self, poses: np.ndarray, color: str) -> list:
        """
        A helper function to visualize the fk result with plotly.

        You can modify it for debugging

        Parameters
        ----------
        poses: np.ndarray
            The poses of each link with shape (L, 4, 4)

        color: str (or any other format supported by plotly)
            The color of the meshes shown in visualization

        Returns
        -------
        A list of plotly objects that can be shown in Vis.show
        """
        vis_list = []
        for l, p in zip(self.links, poses):
            vis_list += Vis.pose(p[:3, 3], p[:3, :3])
            for m in l.visual_meshes:
                vis_list += Vis.mesh(path=m, trans=p[:3, 3], rot=p[:3, :3], color=color)
        return vis_list


if __name__ == "__main__":
    # a simple test to check if the code is working
    # you can modify it to test your code
    from config import get_robot_config

    cfg = get_robot_config("galbot")
    robot_model = RobotModel(cfg)

    gt = np.load(os.path.join("data", "fk.npz"))
    idx = 0
    q = gt["q"][idx]
    gt_poses = gt["poses"][idx]

    # the correct answer is the green one
    gt_vis = robot_model.vis(gt_poses, color="lightgreen")

    my_poses = robot_model.fk(q)
    # your answer is the brown one
    my_vis = robot_model.vis(my_poses, color="brown")

    # it will be shown in the browser
    # if not, you can input a html path to this function and open it manually
    # if a new page is shown but nothing is displayed, you can try refreshing the page
    Vis.show(gt_vis + my_vis, path=None)
