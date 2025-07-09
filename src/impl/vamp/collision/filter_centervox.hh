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
    struct alignas(64) Voxel {
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
        static constexpr uint8_t INDEX_ARRAY_LEN = 255;

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


        // Visualization helper
        struct VizStats {
            uint32_t total_y_tables = 0;
            uint32_t total_z_tables = 0; 
            uint32_t total_voxels = 0;
            uint32_t occupied_voxels = 0;
            uint32_t max_y_tables_per_x = 0;
            uint32_t max_z_tables_per_y = 0;
            uint32_t max_voxels_per_z = 0;
            float memory_utilization_percent = 0.0f;
        };

        // Summary visualization - shows structure statistics
        VizStats visualize_summary() const {
            VizStats stats;
            
            stats.total_y_tables = x_level_table.y_tables.size();
            
            for (const auto& y_table : x_level_table.y_tables) {
                stats.total_z_tables += y_table.z_tables.size();
                stats.max_z_tables_per_y = std::max(stats.max_z_tables_per_y, 
                                                static_cast<uint32_t>(y_table.z_tables.size()));
                
                for (const auto& z_table : y_table.z_tables) {
                    stats.total_voxels += z_table.voxels.size();
                    stats.max_voxels_per_z = std::max(stats.max_voxels_per_z,
                                                    static_cast<uint32_t>(z_table.voxels.size()));
                    
                    for (const auto& voxel : z_table.voxels) {
                        if (voxel.occupied) {
                            stats.occupied_voxels++;
                        }
                    }
                }
            }
            
            stats.max_y_tables_per_x = stats.total_y_tables;
            
            // Calculate memory utilization vs max possible
            uint32_t max_possible_voxels = Y_TABLE_CAPACITY * Z_TABLE_CAPACITY_PER_Y * POINT_CAPACITY_PER_Z;
            stats.memory_utilization_percent = (float(stats.total_voxels) / max_possible_voxels) * 100.0f;
            
            printf("=== CenterSelectiveVoxelFilter Summary ===\n");
            printf("Structure Counts:\n");
            printf("  Y-tables:      %u\n", stats.total_y_tables);
            printf("  Z-tables:      %u\n", stats.total_z_tables);
            printf("  Total voxels:  %u\n", stats.total_voxels);
            printf("  Occupied:      %u (%.1f%%)\n", stats.occupied_voxels, 
                stats.total_voxels > 0 ? (float(stats.occupied_voxels) / stats.total_voxels) * 100.0f : 0.0f);
            
            printf("\nUtilization:\n");
            printf("  Max Y/X:       %u\n", stats.max_y_tables_per_x);
            printf("  Max Z/Y:       %u\n", stats.max_z_tables_per_y);
            printf("  Max Voxels/Z:  %u\n", stats.max_voxels_per_z);
            printf("  Memory util:   %.2f%%\n", stats.memory_utilization_percent);
            
            printf("\nWorkspace Info:\n");
            printf("  Min: (%.3f, %.3f, %.3f)\n", workspace_aabb_min[0], workspace_aabb_min[1], workspace_aabb_min[2]);
            printf("  Max: (%.3f, %.3f, %.3f)\n", workspace_aabb_max[0], workspace_aabb_max[1], workspace_aabb_max[2]);
            printf("  Voxel size: %.6f\n", voxel_size);
            printf("  Scale factor: %.6f\n", inverse_scale_factor);
            printf("  Max range: %.3f\n", std::sqrt(max_range_sq));
            printf("  Origin: (%.3f, %.3f, %.3f)\n", origin_point[0], origin_point[1], origin_point[2]);
            
            return stats;
        }

        // Detailed visualization - shows all mappings and voxel contents
        void visualize_detailed(bool show_empty_voxels = false) const {
            printf("=== Detailed Structure Dump ===\n");
            
            // Show X-level mapping
            printf("\nX-coordinate mappings:\n");
            for (int x = 0; x < INDEX_ARRAY_LEN; ++x) {
                uint8_t y_idx = x_level_table.x_coord_to_y_table_idx[x];
                if (y_idx != INVALID_INDEX) {
                    printf("  X[%d] -> Y-table[%u] (size: %zu)\n", 
                        x, y_idx, x_level_table.y_tables[y_idx].z_tables.size());
                }
            }
            
            // Iterate through structure
            for (size_t y_table_idx = 0; y_table_idx < x_level_table.y_tables.size(); ++y_table_idx) {
                const auto& y_table = x_level_table.y_tables[y_table_idx];
                
                printf("\n--- Y-table[%zu] ---\n", y_table_idx);
                
                // Show Y-level mapping
                printf("Y-coordinate mappings:\n");
                for (int y = 0; y < INDEX_ARRAY_LEN; ++y) {
                    uint8_t z_idx = y_table.y_coord_to_z_table_idx[y];
                    if (z_idx != INVALID_INDEX) {
                        printf("  Y[%d] -> Z-table[%u] (size: %zu)\n", 
                            y, z_idx, y_table.z_tables[z_idx].voxels.size());
                    }
                }
                
                for (size_t z_table_idx = 0; z_table_idx < y_table.z_tables.size(); ++z_table_idx) {
                    const auto& z_table = y_table.z_tables[z_table_idx];
                    
                    printf("\n  --- Z-table[%zu] ---\n", z_table_idx);
                    
                    // Show Z-level mapping  
                    printf("  Z-coordinate mappings:\n");
                    for (int z = 0; z < INDEX_ARRAY_LEN; ++z) {
                        uint8_t voxel_idx = z_table.z_coord_to_voxel_idx[z];
                        if (voxel_idx != INVALID_INDEX) {
                            const auto& voxel = z_table.voxels[voxel_idx];
                            printf("    Z[%d] -> Voxel[%u] %s\n", 
                                z, voxel_idx, voxel.occupied ? "OCCUPIED" : "empty");
                        }
                    }
                    
                    // Show voxel details
                    printf("  Voxels:\n");
                    for (size_t voxel_idx = 0; voxel_idx < z_table.voxels.size(); ++voxel_idx) {
                        const auto& voxel = z_table.voxels[voxel_idx];
                        
                        if (!voxel.occupied && !show_empty_voxels) continue;
                        
                        printf("    [%zu] Center:(%.3f,%.3f,%.3f)", 
                            voxel_idx, 
                            voxel.voxel_center[0], voxel.voxel_center[1], voxel.voxel_center[2]);
                        
                        if (voxel.occupied) {
                            printf(" Point:(%.3f,%.3f,%.3f) Dist²:%.6f", 
                                voxel.stored_point[0], voxel.stored_point[1], voxel.stored_point[2],
                                voxel.stored_point_dist_sq);
                        }
                        printf("\n");
                    }
                }
            }
        }

        // Verification function - checks internal consistency
        bool verify_structure_integrity() const {
            bool is_valid = true;
            
            printf("=== Structure Integrity Check ===\n");
            
            // Check X-level mappings
            for (int x = 0; x < INDEX_ARRAY_LEN; ++x) {
                uint8_t y_idx = x_level_table.x_coord_to_y_table_idx[x];
                if (y_idx != INVALID_INDEX) {
                    if (y_idx >= x_level_table.y_tables.size()) {
                        printf("ERROR: X[%d] maps to invalid Y-table[%u] (size: %zu)\n", 
                            x, y_idx, x_level_table.y_tables.size());
                        is_valid = false;
                    }
                }
            }
            
            // Check Y and Z level mappings
            for (size_t y_table_idx = 0; y_table_idx < x_level_table.y_tables.size(); ++y_table_idx) {
                const auto& y_table = x_level_table.y_tables[y_table_idx];
                
                for (int y = 0; y < INDEX_ARRAY_LEN; ++y) {
                    uint8_t z_idx = y_table.y_coord_to_z_table_idx[y];
                    if (z_idx != INVALID_INDEX) {
                        if (z_idx >= y_table.z_tables.size()) {
                            printf("ERROR: Y-table[%zu] Y[%d] maps to invalid Z-table[%u] (size: %zu)\n", 
                                y_table_idx, y, z_idx, y_table.z_tables.size());
                            is_valid = false;
                        }
                    }
                }
                
                for (size_t z_table_idx = 0; z_table_idx < y_table.z_tables.size(); ++z_table_idx) {
                    const auto& z_table = y_table.z_tables[z_table_idx];
                    
                    for (int z = 0; z < INDEX_ARRAY_LEN; ++z) {
                        uint8_t voxel_idx = z_table.z_coord_to_voxel_idx[z];
                        if (voxel_idx != INVALID_INDEX) {
                            if (voxel_idx >= z_table.voxels.size()) {
                                printf("ERROR: Z-table[%zu] Z[%d] maps to invalid voxel[%u] (size: %zu)\n", 
                                    z_table_idx, z, voxel_idx, z_table.voxels.size());
                                is_valid = false;
                            }
                        }
                    }
                }
            }
            
            printf("Structure integrity: %s\n", is_valid ? "VALID" : "INVALID");
            return is_valid;
        }

        // Test function to verify point insertion and retrieval
        void test_point_roundtrip(const std::vector<Point>& test_points) const {
            printf("=== Point Roundtrip Test ===\n");
            printf("Testing %zu points...\n", test_points.size());
            
            auto extracted = extract_points();
            printf("Extracted %zu points from filter\n", extracted.size());
            
            // Check if all extracted points were in original set
            size_t found_count = 0;
            for (const auto& extracted_pt : extracted) {
                bool found = false;
                for (const auto& orig_pt : test_points) {
                    float dx = extracted_pt[0] - orig_pt[0];
                    float dy = extracted_pt[1] - orig_pt[1]; 
                    float dz = extracted_pt[2] - orig_pt[2];
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    if (dist_sq < 1e-10f) { // Very small tolerance for float precision
                        found = true;
                        found_count++;
                        break;
                    }
                }
                if (!found) {
                    printf("ERROR: Extracted point (%.6f,%.6f,%.6f) not found in original set!\n",
                        extracted_pt[0], extracted_pt[1], extracted_pt[2]);
                }
            }
            
            printf("Found %zu/%zu extracted points in original set\n", found_count, extracted.size());
            printf("Compression ratio: %.2f%% (%zu -> %zu points)\n", 
                test_points.empty() ? 0.0f : (float(extracted.size()) / test_points.size()) * 100.0f,
                test_points.size(), extracted.size());
        }

    private:
        bool insert_to_voxel(uint8_t voxel_x, uint8_t voxel_y, uint8_t voxel_z, const Point& point) {
            // Level 1: Map X coordinate to Y-level table
            uint8_t y_level_index = x_level_table.x_coord_to_y_table_idx[voxel_x];
            if (y_level_index == INVALID_INDEX) {
                y_level_index = static_cast<uint8_t>(x_level_table.y_tables.size());
                
                x_level_table.x_coord_to_y_table_idx[voxel_x] = y_level_index;
                x_level_table.y_tables.emplace_back();
            }
            
            auto& y_level_table = x_level_table.y_tables[y_level_index];
            
            // Level 2: Map Y coordinate to Z-level table
            uint8_t z_level_index = y_level_table.y_coord_to_z_table_idx[voxel_y];
            if (z_level_index == INVALID_INDEX) {
                z_level_index = static_cast<uint8_t>(y_level_table.z_tables.size());
                
                y_level_table.y_coord_to_z_table_idx[voxel_y] = z_level_index;
                y_level_table.z_tables.emplace_back();
            }
            
            auto& z_level_table = y_level_table.z_tables[z_level_index];
            
            // Level 3: Map Z coordinate to voxel
            uint8_t voxel_index = z_level_table.z_coord_to_voxel_idx[voxel_z];
            if (voxel_index == INVALID_INDEX) {
                voxel_index = static_cast<uint8_t>(z_level_table.voxels.size());
               
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
        
        // std::vector<Point> tmp_pc;
        
        // Insert all points (point closer to voxel center wins for each voxel)
        for (uint32_t i = 0; i < pc.shape(0); ++i) {
            Point point{pc(i, 0), pc(i, 1), pc(i, 2)};
            filter.try_insert_point(point);
            // tmp_pc.push_back(point);
        }
    
        
        // Verify correctness
        // auto stats = filter.visualize_summary();
        // filter.verify_structure_integrity();
        // filter.test_point_roundtrip(tmp_pc);
        // filter.visualize_detailed(false);

        return filter.extract_points();
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
