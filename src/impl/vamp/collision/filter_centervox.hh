#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <memory>

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
            const float dx = point[0] - voxel_center[0];
            const float dy = point[1] - voxel_center[1];
            const float dz = point[2] - voxel_center[2];
            const float new_dist_sq = dx*dx + dy*dy + dz*dz;

            if (!occupied || new_dist_sq < stored_point_dist_sq) {
                stored_point = point;
                stored_point_dist_sq = new_dist_sq;
                occupied = true;
                return true;
            }
            
            return false;
        }
    };

    struct CenterSelectiveVoxelFilter {
        static constexpr uint8_t INVALID_INDEX = 255;
        static constexpr uint8_t MAX_GRID_SIZE = 255;
        static constexpr uint8_t MAX_VOXEL_PER_Z_TABLE = 255;
        static constexpr uint8_t INDEX_ARRAY_LEN = 255;

        struct ZLevelTable {
            Voxel* voxels[MAX_VOXEL_PER_Z_TABLE];
            uint8_t voxel_count = 0;     
            uint8_t z_coord_to_voxel_idx[INDEX_ARRAY_LEN];
            
            ZLevelTable() {
                std::fill(z_coord_to_voxel_idx, z_coord_to_voxel_idx + INDEX_ARRAY_LEN, INVALID_INDEX);
            }
        };
        
        struct YLevelTable {
            std::vector<ZLevelTable> z_tables;
            uint8_t y_coord_to_z_table_idx[INDEX_ARRAY_LEN];
            
            YLevelTable() {
                std::fill(y_coord_to_z_table_idx, y_coord_to_z_table_idx + INDEX_ARRAY_LEN, INVALID_INDEX);
                z_tables.reserve(32);
            }
        };

        struct XLevelTable {
            std::vector<YLevelTable> y_tables;
            uint8_t x_coord_to_y_table_idx[INDEX_ARRAY_LEN];
            
            XLevelTable() {
                std::fill(x_coord_to_y_table_idx, x_coord_to_y_table_idx + INDEX_ARRAY_LEN, INVALID_INDEX);
                y_tables.reserve(32);
            }
        };

        XLevelTable x_level_table;
        Point workspace_aabb_min;
        Point workspace_aabb_max;
        float inverse_scale_factor;
        float voxel_size;
        float max_range_sq;
        Point origin_point;
        
        // Voxel memory management
        std::unique_ptr<Voxel[], decltype(&std::free)> voxel_pool{nullptr, &std::free};
        size_t voxel_pool_size;
        size_t allocated_voxel_count = 0;

        CenterSelectiveVoxelFilter(float voxel_sz, float max_range, Point origin, 
                                  Point workspace_min, Point workspace_max)
            : workspace_aabb_min(workspace_min), workspace_aabb_max(workspace_max),
              voxel_size(voxel_sz), max_range_sq(max_range * max_range), 
              origin_point(origin)
        {
            const float workspace_width = std::max({
                workspace_aabb_max[0] - workspace_aabb_min[0],
                workspace_aabb_max[1] - workspace_aabb_min[1], 
                workspace_aabb_max[2] - workspace_aabb_min[2]
            });
            
            const int grid_width = std::min(static_cast<int>(MAX_GRID_SIZE), 
                                          static_cast<int>(std::ceil(workspace_width / voxel_size)));
            
            inverse_scale_factor = grid_width / workspace_width;
            
            // Estimated voxel pool size: (workspace_width/voxel_size)^3 * 0.05
            const float voxels_per_dimension = workspace_width / voxel_size;
            const size_t estimated_voxels = static_cast<size_t>(
                std::pow(voxels_per_dimension, 3.0f) * 0.05f
            );
            
            voxel_pool_size = std::min(estimated_voxels, size_t(32768));
            
            // Allocate aligned memory for voxels
            void* raw_memory = nullptr;
            int result = posix_memalign(&raw_memory, 32, sizeof(Voxel) * voxel_pool_size); // sizeof(Voxel): 32 bytes
            if (result != 0 || raw_memory == nullptr) {
                throw std::bad_alloc();
            }

            voxel_pool.reset(static_cast<Voxel*>(raw_memory));
        }
        
        Voxel* allocate_voxel() {
            if (allocated_voxel_count >= voxel_pool_size) {
                throw std::runtime_error("Voxel pool exhausted");
            }
            
            // Append-only allocation
            return new(&voxel_pool[allocated_voxel_count++]) Voxel();
        }

        bool try_insert_point(const Point& point) {
            // Range and bounds culling
            const float dx = point[0] - origin_point[0];
            const float dy = point[1] - origin_point[1];
            const float dz = point[2] - origin_point[2];
            if (dx*dx + dy*dy + dz*dz >= max_range_sq) return false;

            if (point[0] < workspace_aabb_min[0] || point[0] > workspace_aabb_max[0] ||
                point[1] < workspace_aabb_min[1] || point[1] > workspace_aabb_max[1] ||
                point[2] < workspace_aabb_min[2] || point[2] > workspace_aabb_max[2]) {
                return false;
            }

            // Convert to voxel coordinates
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
            result.reserve(allocated_voxel_count);
        
            for (const auto& y_table : x_level_table.y_tables) {
                for (const auto& z_table : y_table.z_tables) {
                    for (uint8_t i = 0; i < z_table.voxel_count; ++i) {
                        const Voxel* voxel = z_table.voxels[i];
                        if (voxel->occupied) {
                            result.push_back(voxel->stored_point);
                        }
                    }
                }
            }
            
            return result;
        }

        // Visualization and debugging
        void visualize_summary() const {
            uint32_t total_y_tables = x_level_table.y_tables.size();
            uint32_t total_z_tables = 0;
            uint32_t total_voxels = 0;
            uint32_t occupied_voxels = 0;
            uint32_t max_z_tables_per_y = 0;
            uint32_t max_voxels_per_z = 0;
            
            for (const auto& y_table : x_level_table.y_tables) {
                total_z_tables += y_table.z_tables.size();
                max_z_tables_per_y = std::max(max_z_tables_per_y, 
                                            static_cast<uint32_t>(y_table.z_tables.size()));
                
                for (const auto& z_table : y_table.z_tables) {
                    total_voxels += z_table.voxel_count;
                    max_voxels_per_z = std::max(max_voxels_per_z,
                                              static_cast<uint32_t>(z_table.voxel_count));
                    
                    for (uint8_t i = 0; i < z_table.voxel_count; ++i) {
                        if (z_table.voxels[i]->occupied) {
                            occupied_voxels++;
                        }
                    }
                }
            }
            
            printf("=== CenterSelectiveVoxelFilter Summary ===\n");
            printf("Structure: %u Y-tables, %u Z-tables, %u voxels (%u occupied)\n",
                total_y_tables, total_z_tables, total_voxels, occupied_voxels);
            printf("Pool: %zu/%zu allocated (%.1f%%)\n", 
                allocated_voxel_count, voxel_pool_size, (allocated_voxel_count * 100.0f) / voxel_pool_size);
            printf("Max utilization: %u Z-tables/Y, %u voxels/Z\n", 
                max_z_tables_per_y, max_voxels_per_z);
            printf("Workspace: (%.3f,%.3f,%.3f) to (%.3f,%.3f,%.3f), voxel_size=%.6f\n",
                workspace_aabb_min[0], workspace_aabb_min[1], workspace_aabb_min[2],
                workspace_aabb_max[0], workspace_aabb_max[1], workspace_aabb_max[2], voxel_size);
        }

        void test_point_roundtrip(size_t input_pc_size) const {
            auto extracted = extract_points();
            printf("=== Point Roundtrip Test ===\n");
            printf("Input: %zu points, Output: %zu points (%.2f%% compression)\n", 
                input_pc_size, extracted.size(),
                (extracted.size() * 100.0f) / input_pc_size);
        }

        size_t get_allocated_voxel_count() const { return allocated_voxel_count; }
        size_t get_max_capacity() const { return voxel_pool_size; }

    private:
        bool insert_to_voxel(uint8_t voxel_x, uint8_t voxel_y, uint8_t voxel_z, const Point& point) {
            // Level 1: Get or create Y-level table for X coordinate
            uint8_t& y_level_index = x_level_table.x_coord_to_y_table_idx[voxel_x];
            if (y_level_index == INVALID_INDEX) {
                y_level_index = static_cast<uint8_t>(x_level_table.y_tables.size());
                x_level_table.y_tables.emplace_back();
            }
            
            auto& y_level_table = x_level_table.y_tables[y_level_index];
            
            // Level 2: Get or create Z-level table for Y coordinate
            uint8_t& z_level_index = y_level_table.y_coord_to_z_table_idx[voxel_y];
            if (z_level_index == INVALID_INDEX) {
                z_level_index = static_cast<uint8_t>(y_level_table.z_tables.size());
                y_level_table.z_tables.emplace_back();
            }
            
            auto& z_level_table = y_level_table.z_tables[z_level_index];
            
            // Level 3: Get or create voxel for Z coordinate
            uint8_t& voxel_index = z_level_table.z_coord_to_voxel_idx[voxel_z];
            if (voxel_index == INVALID_INDEX) {
                if (z_level_table.voxel_count >= MAX_VOXEL_PER_Z_TABLE) {
                    throw std::runtime_error("Z-table capacity exceeded");
                }

                Voxel* new_voxel = allocate_voxel();

                voxel_index = z_level_table.voxel_count++;
                z_level_table.voxels[voxel_index] = new_voxel;
                
                // Set voxel center coordinates
                new_voxel->set_voxel_center(voxel_x, voxel_y, voxel_z, voxel_size, workspace_aabb_min);
            }
            
            return z_level_table.voxels[voxel_index]->try_insert(point);
        }
    };
    
    // CenterVox: Voxel-based point cloud downsampling filter
    //
    // Subdivides workspace into 3D grid of cubic voxels, keeping at most one point per voxel.
    // The representative point is chosen as the one closest to the voxel's geometric center.
    //
    // Parameters:
    // - voxel_size: Edge length of cubic voxels (controls downsampling resolution)
    // - max_range: Maximum distance from origin for point retention  
    // - origin: Origin point for range filtering
    // - workspace_min/max: Workspace bounding box
    //
    // Features:
    // - Adaptive memory allocation based on workspace volume
    // - Append-only voxel pool for optimal performance
    // - Three-level hierarchical lookup (X->Y->Z) for sparse storage

    template <typename PointCloud>
    auto filter_pointcloud_centervox(
        const PointCloud &pc,
        float voxel_size,
        float max_range,
        Point origin,
        Point workspace_min,
        Point workspace_max) -> std::vector<Point>
    {
        if (pc.shape(0) == 0) {
            return std::vector<Point>();
        }

        CenterSelectiveVoxelFilter filter(voxel_size, max_range, origin, workspace_min, workspace_max);
        
        for (uint32_t i = 0; i < pc.shape(0); ++i) {
            Point point{pc(i, 0), pc(i, 1), pc(i, 2)};
            filter.try_insert_point(point);
        }

        // filter.test_point_roundtrip(pc.shape(0));
        // filter.visualize_summary();

        return filter.extract_points();
    }

    template <>
    inline auto filter_pointcloud_centervox(
        const std::vector<Point> &pc,
        float voxel_size,
        float max_range,
        Point origin,
        Point workspace_min,
        Point workspace_max) -> std::vector<Point>
    {
        struct PointcloudWrapper {
            const std::vector<Point> &pc;
            
            auto shape(std::size_t dim) const noexcept -> std::size_t {
                return (dim == 0) ? pc.size() : (dim == 1) ? 3 : 0;
            }

            auto operator()(std::size_t i, std::size_t j) const noexcept -> typename Point::value_type {
                return pc[i][j];
            }
        };

        return filter_pointcloud_centervox(
            PointcloudWrapper{pc}, voxel_size, max_range,
            std::move(origin), std::move(workspace_min), std::move(workspace_max));
    }

}  // namespace vamp::collision