#include "volumetric_mapper.h"
// #include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vol_node");
    VOLMAPNODE vol_mapper;
    ros::spin();
    // ros::MultiThreadedSpinner spinner(4);
    // std::cout << " !!!!" << std::endl;
    vol_mapper.save_glb_edt();
    return 0;
}
