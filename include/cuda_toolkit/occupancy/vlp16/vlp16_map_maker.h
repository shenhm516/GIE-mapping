
#ifndef SRC_VLP16_MAP_MAKER_H
#define SRC_VLP16_MAP_MAKER_H

#include "multiscan_param.h"
#include <cuda_toolkit/projection.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include "map_structure/local_batch.h"
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
// namespace velodyne_ros {
//   struct EIGEN_ALIGN16 Point {
//       PCL_ADD_POINT4D;
//       float intensity;
//       float time;
//       uint16_t ring;
//       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//   };
// }  // namespace velodyne_ros
// POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
//     (float, x, x)
//     (float, y, y)
//     (float, z, z)
//     (float, intensity, intensity)
//     (float, time, time)
//     (uint16_t, ring, ring)
// )

class Vlp16MapMaker
{
public:
    Vlp16MapMaker();
    ~Vlp16MapMaker();

    void initialize(const MulScanParam &p);
    bool is_initialized(){return _initialized;}

    void setLocMap(LocMap *lMap);
    void updateLocalOGM(const Projection& proj, const sensor_msgs::PointCloud2ConstPtr& pyntcld,
                        int3* VB_keys_loc_D, const int time,  bool for_motion_planner, int rbt_r2_grids,
                        ros::Publisher _pcl_in_pub, ros::Publisher _pcl_out_pub);
    void convertPyntCld(const sensor_msgs::PointCloud2ConstPtr &msg, ros::Publisher _pcl_in_pub, ros::Publisher _pcl_out_pub);
private:
    MulScanParam _mul_scan_param;
    int _range_byte_sz;
    SCAN_DEPTH_TPYE *_gpu_mulscan;
    bool _initialized = false;
    sensor_msgs::LaserScan  scanlines[16];
    const int rayid_toup[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    LocMap * _lMap;
};

#endif //SRC_VLP16_MAP_MAKER_H
