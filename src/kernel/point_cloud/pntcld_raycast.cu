#include "pntcld_interfaces.h"
#include <float.h>
#include "ray_cast.h"
#include "par_wave/voxmap_utils.cuh"

namespace PNTCLD_RAYCAST
{

__device__ __forceinline__
bool clearRayLoc(LocMap &loc_map,const int3 &crd, const float &val1, const float &val2, const int &time)
{
    if (loc_map.get_vox_type(crd) != VOXTYPE_OCCUPIED)
    {
        loc_map.atom_add_type_count(crd, -1);
        return true;
    }
    return false;
}

__global__
void FreeKNNCheck(LocMap loc_map, int3* VB_keys_loc_D, bool for_motion_planner, int rbt_r2_grids)
{
    // get the z and y coordinate of the grid we are about to scan
    int3 loc_crd;
    loc_crd.z = blockIdx.x;
    loc_crd.y = threadIdx.x;

    for (loc_crd.x = 0; loc_crd.x < loc_map._local_size.x; ++loc_crd.x)
    {
        int idx_1d = loc_map.coord2idx_local(loc_crd);
        char local_type = loc_map.get_vox_type(loc_crd);
        int cnt = 0;
        if (local_type==VOXTYPE_FREE && 
            loc_crd.x!=0 && (loc_crd.x<loc_map._local_size.x-1) &&
            loc_crd.y!=0 && (loc_crd.y<loc_map._local_size.y-1) &&
            loc_crd.z!=0 && (loc_crd.z<loc_map._local_size.z-1)) {
            for (int i=-1; i<2; i++) {
                for (int j=-1; j<2; j++) {
                    for (int k=-1; k<2; k++) {
                        int3 delta_crd = make_int3(i,j,k);
                        int3 near_crd = loc_crd + delta_crd;
                        char near_type = loc_map.get_vox_type(near_crd);
                        if (near_type==VOXTYPE_FREE || near_type ==VOXTYPE_OCCUPIED) {
                            cnt++;
                        }
                    }
                }
            }

        if (cnt<=26) VB_keys_loc_D[idx_1d] = make_int3(1e6,1e6,1e6); // vox type is unknown
        } else if (local_type==VOXTYPE_FREE) {
            // loc_map.set_vox_type(loc_crd,VOXTYPE_UNKNOWN);
            VB_keys_loc_D[idx_1d] = make_int3(1e6,1e6,1e6); // vox type is unknown
        }
    }
    for (loc_crd.x = 0; loc_crd.x < loc_map._local_size.x; ++loc_crd.x) {
        int idx_1d = loc_map.coord2idx_local(loc_crd);
        if (VB_keys_loc_D[idx_1d].x == 1e6) {
            loc_map.set_vox_type(loc_crd,VOXTYPE_UNKNOWN);
            VB_keys_loc_D[idx_1d] = EMPTY_KEY; // vox type is unknown
        }
    }
}


__global__
void getAllocKeys(LocMap loc_map, int3* VB_keys_loc_D, bool for_motion_planner, int rbt_r2_grids)
{
    // get the z and y coordinate of the grid we are about to scan
    int3 loc_crd;
    loc_crd.z = blockIdx.x;
    loc_crd.y = threadIdx.x;

    for (loc_crd.x = 0; loc_crd.x < loc_map._local_size.x; ++loc_crd.x)
    {
        // set grids around as known and free
        if (for_motion_planner)
        {
            int3 crd2center = loc_crd -loc_map._half_shift;
            if(crd2center.x*crd2center.x + crd2center.y*crd2center.y+crd2center.z*crd2center.z <= rbt_r2_grids)
            {
                loc_map.set_vox_count(loc_crd, -1);
            }

        }

        int idx_1d = loc_map.coord2idx_local(loc_crd);

        int count = loc_map.get_vox_count(loc_crd);

        if (count == 0) {
            // loc_map.set_vox_type(loc_crd,VOXTYPE_UNKNOWN);
            VB_keys_loc_D[idx_1d] = EMPTY_KEY; // vox type is unknown
        }
        else {
            if(count>0) {
                // if (loc_map.get_vox_type(loc_crd)!=VOXTYPE_FREE)
                loc_map.set_vox_type(loc_crd,VOXTYPE_OCCUPIED);
                // else loc_map.set_vox_type(loc_crd,VOXTYPE_FREE);
                // loc_map.set_vox_glb_type(loc_crd,VOXTYPE_OCCUPIED);
            }
            else {
                loc_map.set_vox_type(loc_crd,VOXTYPE_FREE);
                // loc_map.set_vox_glb_type(loc_crd,VOXTYPE_FREE);
            }
            int3 glb_crd = loc_map.loc2glb(loc_crd);
            VB_keys_loc_D[idx_1d] = get_VB_key(glb_crd);
        }
    }
}



__global__
void freeLocObs(LocMap loc_map, float4 *pnt_cld, Projection proj, int pnt_sz, int time)
{
    int ring_id = blockIdx.x;
    int scan_id = threadIdx.x;
    int id = threadIdx.x + blockIdx.x *blockDim.x;

    if(id >= pnt_sz) return;

    float3 glb_pos = proj.L2G*make_float3(pnt_cld[id].x, pnt_cld[id].y, pnt_cld[id].z);

    RAY::rayCastLoc(loc_map, proj.origin,  glb_pos, time, 0.707f*loc_map._local_size.x*loc_map._voxel_width, &clearRayLoc);
}


__global__
void registerLocObs(LocMap loc_map, float4 *pnt_cld, Projection proj,  int pnt_sz, int time)
{
    int ring_id = blockIdx.x;
    int scan_id = threadIdx.x;
    int id = threadIdx.x + blockIdx.x *blockDim.x;

    if(id >= pnt_sz) return;

    float3 glb_pos = proj.L2G*make_float3(pnt_cld[id].x, pnt_cld[id].y, pnt_cld[id].z);
    // if (glb_pos.z >= loc_map._update_min_h && glb_pos.z <= loc_map._update_max_h)
    {
        int3 glb_crd = loc_map.pos2coord(glb_pos);
        int3 loc_crd = loc_map.glb2loc(glb_crd);

        // if (loc_map.get_vox_glb_type(loc_crd)==VOXTYPE_FREE) {
        //     loc_map.set_vox_type(loc_crd,VOXTYPE_DYN);
        //     pnt_cld[id].w = 1;
        // }
        // else {
        loc_map.set_vox_type(loc_crd,VOXTYPE_OCCUPIED);
        loc_map.atom_add_type_count(loc_crd,1);
        // }
    }
}

__global__
void registerLocDyn(LocMap loc_map, float4 *pnt_cld, Projection proj, Projection proj_prev, int pnt_sz, int time)
{
    int ring_id = blockIdx.x;
    int scan_id = threadIdx.x;
    int id = threadIdx.x + blockIdx.x *blockDim.x;

    if(id >= pnt_sz) return;
    float3 glb_pos = proj.L2G*make_float3(pnt_cld[id].x, pnt_cld[id].y, pnt_cld[id].z);
    int3 glb_crd = loc_map.pos2coord(glb_pos);
    // int3 loc_crd = loc_map.glb2loc(glb_crd);
    int3 pivot = loc_map.pos2coord(proj_prev.origin);
    pivot.x -= loc_map._local_size.x/2;
    pivot.y -= loc_map._local_size.y/2;
    pivot.z -= loc_map._local_size.z/2;
    int3 loc_crd = glb_crd-pivot;

    if (loc_map.get_vox_glb_type(loc_crd)==VOXTYPE_FREE) {
        pnt_cld[id].w = 1;
    }
}


void localOGMKernels(LocMap* loc_map, float4 *pnt_cld, Projection proj, Projection proj_prev, PntcldParam param,
                     int3* VB_keys_loc_D, int time, bool for_motion_planner, int rbt_r2_grids)
{
    // Register the point clouds
    registerLocObs<<<param.valid_pnt_count/256+1, 256>>>(*loc_map,pnt_cld,proj,param.valid_pnt_count,time);

    // Free the empty areas
    freeLocObs<<<param.valid_pnt_count/256+1, 256>>>(*loc_map,pnt_cld,proj,param.valid_pnt_count,time);

    const int gridSize = loc_map->_local_size.z;
    const int blkSize = loc_map->_local_size.y;
    getAllocKeys<<<gridSize,blkSize>>>(*loc_map, VB_keys_loc_D, for_motion_planner, rbt_r2_grids);
    FreeKNNCheck<<<gridSize,blkSize>>>(*loc_map, VB_keys_loc_D, for_motion_planner, rbt_r2_grids);
    registerLocDyn<<<param.valid_pnt_count/256+1, 256>>>(*loc_map,pnt_cld,proj,proj_prev,param.valid_pnt_count,time);
}
}
