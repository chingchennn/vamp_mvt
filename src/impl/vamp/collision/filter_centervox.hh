#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>

#include <vamp/collision/math.hh>
#include <vamp/vector.hh>

namespace vamp::collision
{
    struct Voxel {
        Point stored_point;
        Point voxel_center;
        float stored_point_dist_sq = 0.0f;
        bool occupied = false;
        
        void set_voxel_center(uint8_t vx, uint8_t vy, uint8_t vz, float voxel_size, Point workspace_min) {
            voxel_center[0] = workspace_min[0] + (vx + 0.5f) * voxel_size;
            voxel_center[1] = workspace_min[1] + (vy + 0.5f) * voxel_size;
            voxel_center[2] = workspace_min[2] + (vz + 0.5f) * voxel_size;
        }
        
        bool try_insert(const Point& point) {
            // Calculate new point's distance to voxel center
            const float dx = point[0] - voxel_center[0];
            const float dy = point[1] - voxel_center[1];
            const float dz = point[2] - voxel_center[2];
            const float new_dist_sq = dx*dx + dy*dy + dz*dz;
            
            if (!occupied) {
                // First point in this voxel
                stored_point = point;
                stored_point_dist_sq = new_dist_sq;
                occupied = true;
                return true;
            }
            
            // Compare with stored distance
            if (new_dist_sq < stored_point_dist_sq) {
                stored_point = point;
                stored_point_dist_sq = new_dist_sq;
                return true;
            }
            
            return false;
        }
    };

    struct CenterSelectiveVoxelFilter {
        static constexpr uint8_t INVALID_INDEX = 255;
        static constexpr uint8_t MAX_GRID_SIZE = 255;
        static constexpr uint8_t Y_TABLE_CAPACITY = 32;
        static constexpr uint8_t Z_TABLE_CAPACITY_PER_Y = 32;
        static constexpr uint8_t POINT_CAPACITY_PER_Z = 32;
        static constexpr uint8_t INDEX_ARRAY_LEN = 128;

        struct ZLevelTable {
            std::vector<Voxel> voxels;
            uint8_t z_coord_to_voxel_idx[INDEX_ARRAY_LEN];
            
            ZLevelTable() {
                std::fill(z_coord_to_voxel_idx, z_coord_to_voxel_idx + INDEX_ARRAY_LEN, INVALID_INDEX);
                voxels.reserve(POINT_CAPACITY_PER_Z);
            }
        };
        
        struct YLevelTable {
            std::vector<ZLevelTable> z_tables;
            uint8_t y_coord_to_z_table_idx[INDEX_ARRAY_LEN];
            
            YLevelTable() {
                std::fill(y_coord_to_z_table_idx, y_coord_to_z_table_idx + INDEX_ARRAY_LEN, INVALID_INDEX);
                z_tables.reserve(Z_TABLE_CAPACITY_PER_Y);
            }
        };

        struct XLevelTable {
            std::vector<YLevelTable> y_tables;
            uint8_t x_coord_to_y_table_idx[INDEX_ARRAY_LEN];
            
            XLevelTable() {
                std::fill(x_coord_to_y_table_idx, x_coord_to_y_table_idx + INDEX_ARRAY_LEN, INVALID_INDEX);
                y_tables.reserve(Y_TABLE_CAPACITY);
            }
        };

        XLevelTable x_level_table;
        Point workspace_aabb_min;
        Point workspace_aabb_max;
        float inverse_scale_factor;
        float voxel_size;
        float max_range_sq;
        Point origin_point;
        bool enable_culling;

        CenterSelectiveVoxelFilter(float voxel_sz, float max_range, Point origin, 
                                  Point workspace_min, Point workspace_max, bool cull)
            : workspace_aabb_min(workspace_min), workspace_aabb_max(workspace_max),
              voxel_size(voxel_sz), max_range_sq(max_range * max_range), 
              origin_point(origin), enable_culling(cull)
        {
            const float workspace_width = std::max({
                workspace_aabb_max[0] - workspace_aabb_min[0],
                workspace_aabb_max[1] - workspace_aabb_min[1], 
                workspace_aabb_max[2] - workspace_aabb_min[2]
            });
            
            const int grid_width = std::min(static_cast<int>(MAX_GRID_SIZE), 
                                          static_cast<int>(std::ceil(workspace_width / voxel_size)));
            
            inverse_scale_factor = grid_width / workspace_width;
        }

        bool try_insert_point(const Point& point) {
            // Range and bounds culling
            if (enable_culling) {
                const float dx = point[0] - origin_point[0];
                const float dy = point[1] - origin_point[1];
                const float dz = point[2] - origin_point[2];
                if (dx*dx + dy*dy + dz*dz >= max_range_sq) return false;

                if (point[0] < workspace_aabb_min[0] || point[0] > workspace_aabb_max[0] ||
                    point[1] < workspace_aabb_min[1] || point[1] > workspace_aabb_max[1] ||
                    point[2] < workspace_aabb_min[2] || point[2] > workspace_aabb_max[2]) {
                    return false;
                }
            }

            // Convert to uint8_t voxel coordinates
            const uint8_t vx = static_cast<uint8_t>(std::clamp(
                static_cast<int>((point[0] - workspace_aabb_min[0]) * inverse_scale_factor), 
                0, MAX_GRID_SIZE - 1));
            const uint8_t vy = static_cast<uint8_t>(std::clamp(
                static_cast<int>((point[1] - workspace_aabb_min[1]) * inverse_scale_factor), 
                0, MAX_GRID_SIZE - 1));
            const uint8_t vz = static_cast<uint8_t>(std::clamp(
                static_cast<int>((point[2] - workspace_aabb_min[2]) * inverse_scale_factor), 
                0, MAX_GRID_SIZE - 1));

            return insert_to_voxel(vx, vy, vz, point);
        }

        std::vector<Point> extract_points() const {
            std::vector<Point> result;
            
            for (const auto& y_table : x_level_table.y_tables) {
                for (const auto& z_table : y_table.z_tables) {
                    for (const auto& voxel : z_table.voxels) {
                        if (voxel.occupied) {
                            result.push_back(voxel.stored_point);
                        }
                    }
                }
            }
            
            return result;
        }

    private:
        bool insert_to_voxel(uint8_t voxel_x, uint8_t voxel_y, uint8_t voxel_z, const Point& point) {
            // Level 1: Map X coordinate to Y-level table
            uint8_t y_level_index = x_level_table.x_coord_to_y_table_idx[voxel_x];
            if (y_level_index == INVALID_INDEX) {
                y_level_index = static_cast<uint8_t>(x_level_table.y_tables.size());
                if (y_level_index >= Y_TABLE_CAPACITY) return false;
                
                x_level_table.x_coord_to_y_table_idx[voxel_x] = y_level_index;
                x_level_table.y_tables.emplace_back();
            }
            
            auto& y_level_table = x_level_table.y_tables[y_level_index];
            
            // Level 2: Map Y coordinate to Z-level table
            uint8_t z_level_index = y_level_table.y_coord_to_z_table_idx[voxel_y];
            if (z_level_index == INVALID_INDEX) {
                z_level_index = static_cast<uint8_t>(y_level_table.z_tables.size());
                if (z_level_index >= Z_TABLE_CAPACITY_PER_Y) return false;
                
                y_level_table.y_coord_to_z_table_idx[voxel_y] = z_level_index;
                y_level_table.z_tables.emplace_back();
            }
            
            auto& z_level_table = y_level_table.z_tables[z_level_index];
            
            // Level 3: Map Z coordinate to voxel
            uint8_t voxel_index = z_level_table.z_coord_to_voxel_idx[voxel_z];
            if (voxel_index == INVALID_INDEX) {
                voxel_index = static_cast<uint8_t>(z_level_table.voxels.size());
                if (voxel_index >= POINT_CAPACITY_PER_Z) return false;
                
                z_level_table.z_coord_to_voxel_idx[voxel_z] = voxel_index;
                z_level_table.voxels.emplace_back();
                
                // Set voxel center when creating new voxel (uint8_t coordinates)
                z_level_table.voxels.back().set_voxel_center(
                    voxel_x, voxel_y, voxel_z, voxel_size, workspace_aabb_min);
            }
            
            auto& voxel = z_level_table.voxels[voxel_index];
            return voxel.try_insert(point);
        }
    };
    
    // CenterVox: MVT-inspired voxel filter for efficient point cloud downsampling
    //
    // Inputs
    // - `pc`: the initial pointcloud
    // - `min_dist`: The minimum distance between two points to be considered distinct.
    // - `max_range`: The maximum distance for a point from the origin to be retained from `pc`.
    // - `origin`: The location of the origin.
    // - `workspace_min`: The minimum vertex of the AABB describing the workspace.
    // - `workspace_max`: The maximum vertex of the AABB describing the workspace.
    // - `cull`: Enable range and bounds filtering
    //
    // Returns a subset of `pc`, subject to the following conditions:
    // - Points are spatially distributed with minimum distance >= min_dist
    // - At most one point per voxel
    // - Within each voxel, selects point closest to voxel center
    // - Points beyond max_range from origin are removed (if cull=true)  
    // - Points outside workspace bounds are removed (if cull=true)
    template <typename PointCloud>
    auto filter_pointcloud_centervox(
        const PointCloud &pc,
        float min_dist,
        float max_range,
        Point origin,
        Point workspace_min,
        Point workspace_max,
        bool cull) -> std::vector<Point>
    {
        if (pc.shape(0) == 0) {
            return std::vector<Point>();
        }

        // Create voxel filter with min_dist as voxel size
        CenterSelectiveVoxelFilter filter(min_dist * 1.4, max_range, origin, workspace_min, workspace_max, cull);
        
        // Insert all points (point closer to voxel center wins for each voxel)
        for (uint32_t i = 0; i < pc.shape(0); ++i) {
            Point point{pc(i, 0), pc(i, 1), pc(i, 2)};
            filter.try_insert_point(point);
        }
        
        // Extract filtered points
        auto voxel_result = filter.extract_points();

        return voxel_result;
    }

    template <>
    inline auto filter_pointcloud_centervox(
        const std::vector<Point> &pc,
        float min_dist,
        float max_range,
        Point origin,
        Point workspace_min,
        Point workspace_max,
        bool cull) -> std::vector<Point>
    {
        struct PointcloudWrapper
        {
            inline auto shape(std::size_t dim) const noexcept -> std::size_t
            {
                if (dim == 0)
                {
                    return pc.size();
                }

                if (dim == 1)
                {
                    return 3;
                }

                return -1;
            }

            inline auto operator()(std::size_t i, std::size_t j) const noexcept -> typename Point::value_type
            {
                return pc[i][j];
            }

            const std::vector<Point> &pc;
        };

        return filter_pointcloud_centervox(
            PointcloudWrapper{pc},
            min_dist,
            max_range,
            std::move(origin),
            std::move(workspace_min),
            std::move(workspace_max),
            cull);
    }

}  // namespace vamp::collision
