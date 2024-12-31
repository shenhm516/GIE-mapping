#ifndef PNTCLD_INTERFACES_H
#define PNTCLD_INTERFACES_H

#include "cuda_toolkit/projection.h"
#include "cuda_toolkit/occupancy/point_cloud/pntcld_param.h"
#include "map_structure/local_batch.h"

namespace PNTCLD_RAYCAST
{
void localOGMKernels(LocMap* loc_map, float4 *pnt_cld, Projection proj, Projection proj_prev, PntcldParam param,
                     int3* VB_keys_loc_D, int time, bool for_motion_planner, int rbt_r2_grids);
}

#endif // PNTCLD_INTERFACES_H
