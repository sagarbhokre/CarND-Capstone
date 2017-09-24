#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
SPEED = 40

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.waypoints = None
        self.tf_light_index = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(50) # 50Hz

        while not rospy.is_shutdown():
            lane = Lane()

            if self.waypoints is not None and self.pose is not None:
                #lane.header = self.waypoints.header
                closet_waypoint_index = self.get_closest_waypoint_index(self.waypoints.waypoints, self.pose)
                new_tf_light_index = None
                if self.tf_light_index >= closet_waypoint_index and self.tf_light_index <= closet_waypoint_index+LOOKAHEAD_WPS:
                    new_tf_light_index = self.tf_light_index - closet_waypoint_index
                lane.waypoints = self.waypoints.waypoints[closet_waypoint_index:closet_waypoint_index+LOOKAHEAD_WPS]

                for waypoint in lane.waypoints:
                    waypoint.twist.twist.linear.x = SPEED

                if new_tf_light_index is not None:
                    lane.waypoints[new_tf_light_index].twist.twist.linear.x = 0.0

                self.final_waypoints_pub.publish(lane)
            rate.sleep()


    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg.pose


    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.tf_light_index = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def get_closest_waypoint_index(self, waypoints, pose):
        dist = float('inf')
        index = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for waypoint in waypoints:
            temp_dist = dl(waypoint.pose.pose.position, pose.position)
            if temp_dist < dist:
                dist = temp_dist
                index = index + 1

        if waypoints[index].pose.pose.position.x < self.pose.position.x:
            index = index + 1

        return index

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
