#include "cuda_toolkit/occupancy/point_cloud/pntcld_map_maker.h"
#include <sensor_msgs/point_cloud2_iterator.h>
#include "kernel/point_cloud/pntcld_interfaces.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>

PntcldMapMaker::PntcldMapMaker()
{

}

PntcldMapMaker::~PntcldMapMaker()
{
    if(_initialized)
    {
        GPU_FREE(_gpu_cld);
        delete [] _cpu_cld;
    }
}

void PntcldMapMaker::initialize(const PntcldParam &p)
{
    // Relase the old buffer if there is one
    if(_initialized)
    {
        GPU_FREE(_gpu_cld);
        delete [] _cpu_cld;
    }

    _pnt_param = p;
    // Create device memory
    _cld_byte_sz = (_pnt_param.cld_sz)*sizeof(PNT_TYPE);
    GPU_MALLOC(&_gpu_cld, _cld_byte_sz);
    _cpu_cld = new PNT_TYPE[_pnt_param.cld_sz];

    _initialized = true;
}

void PntcldMapMaker::initialize(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    PntcldParam p(msg->width*msg->height);
    initialize(p);
}


void PntcldMapMaker::setLocMap(LocMap *lMap)
{
    _lMap = lMap;
}

void PntcldMapMaker::pntcld_process(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    _pnt_param.valid_pnt_count = 0;
    for (sensor_msgs::PointCloud2ConstIterator<float> it(*msg, "x"); it != it.end(); ++it)
    {
        float4 pnt = make_float4(it[0],it[1],it[2],0); //shm: x,y,z,dyn_flag
        if (_pnt_param.valid_pnt_count < _pnt_param.cld_sz)
        {
            _cpu_cld[_pnt_param.valid_pnt_count] = pnt;
            _pnt_param.valid_pnt_count++;
        }
    }
}

void PntcldMapMaker::updateLocalOGM(const Projection& proj, const Projection& proj_prev, const sensor_msgs::PointCloud2::ConstPtr &msg,
                                    int3* VB_keys_loc_D, const int time, bool for_motion_planner, int rbt_r2_grids,
                                    ros::Publisher _pcl_in_pub)
{
    // Copy the data from point cloud to multiple scans
    pntcld_process(msg);

    // Copy the scan into gpu buffer
    GPU_MEMCPY_H2D(_gpu_cld,_cpu_cld,_cld_byte_sz);

    PNTCLD_RAYCAST::localOGMKernels(_lMap,_gpu_cld,proj, proj_prev,_pnt_param,VB_keys_loc_D,time, for_motion_planner, rbt_r2_grids);
    GPU_MEMCPY_D2H(_cpu_cld,_gpu_cld,_cld_byte_sz);
    
    pcl::PointCloud<pcl::PointXYZ> points_dyn;
    for (int i=0; i<_pnt_param.valid_pnt_count; i++) {
        if (_cpu_cld[i].w == 1) {
            pcl::PointXYZ point_dyn;
            point_dyn.x = _cpu_cld[i].x;
            point_dyn.y = _cpu_cld[i].y;
            point_dyn.z = _cpu_cld[i].z;
            points_dyn.points.push_back(point_dyn);
        }
    }
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(points_dyn, laserCloudmsg);
    laserCloudmsg.header = msg->header;
    // laserCloudmsg.header.frame_id = "body";
    _pcl_in_pub.publish(laserCloudmsg);
}
