#pragma once

#include <unistd.h>
#include <iostream>
#include <fstream> 
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <new>
#include <numeric>
#include <limits>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>

#include <vamp/collision/math.hh>
#include <vamp/vector.hh>

#define ALIGN_TO_SIMD_WIDTH(value) ((((value) + FVectorT::num_scalars - 1) / FVectorT::num_scalars) * FVectorT::num_scalars)

namespace vamp::collision
{
    struct MVT
    {
        using FVectorT = FloatVector<>;
        using IVectorT = IntVector<>;
        
        // Constants
        static constexpr uint8_t INVALID_INDEX = 255;
        static constexpr uint8_t MAX_GRID_WIDTH = 255;

        struct alignas(32) Voxel {
            // Structure-of-Arrays for SIMD efficiency
            float* x_coords = nullptr;
            float* y_coords = nullptr;
            float* z_coords = nullptr;
            size_t point_count = 0;
            size_t capacity = 0;
            
            // Axis-Aligned Bounding Box bounds
            Point bbox_min = {0.0f, 0.0f, 0.0f};
            Point bbox_max = {0.0f, 0.0f, 0.0f};
            
            Voxel() = default;

            void initialize_with_pool(float* x_ptr, float* y_ptr, float* z_ptr, size_t cap) {
                x_coords = x_ptr;
                y_coords = y_ptr;
                z_coords = z_ptr;
                capacity = cap;
                point_count = 0;
            }

            void add_point(const Point& point) {
                if (point_count >= capacity) {
                    std::cout << "Try to add " << point_count + 1 << "th point to a voxel" << std::endl;
                    throw std::runtime_error("Voxel capacity exceeded");
                }

                x_coords[point_count] = point[0];
                y_coords[point_count] = point[1];
                z_coords[point_count] = point[2];
                
                // Update bounding box
                if (point_count == 0) {
                    bbox_min = bbox_max = point;
                } else {
                    bbox_min[0] = std::min(bbox_min[0], point[0]);
                    bbox_min[1] = std::min(bbox_min[1], point[1]);
                    bbox_min[2] = std::min(bbox_min[2], point[2]);
                    bbox_max[0] = std::max(bbox_max[0], point[0]);
                    bbox_max[1] = std::max(bbox_max[1], point[1]);
                    bbox_max[2] = std::max(bbox_max[2], point[2]);
                }
                ++point_count;
            }
        };

        // Three-level voxel table
        struct ZLevelTable {
            std::vector<Voxel> voxels;
            uint8_t* z_coord_to_voxel_idx = nullptr;
            
            void initialize_with_pool(uint8_t* idx_ptr, uint8_t array_len) {
                z_coord_to_voxel_idx = idx_ptr;
                std::fill(z_coord_to_voxel_idx, z_coord_to_voxel_idx + array_len, INVALID_INDEX);
            }
        };
        
        struct YLevelTable {
            std::vector<ZLevelTable> z_tables;
            uint8_t* y_coord_to_z_table_idx = nullptr;
            
            void initialize_with_pool(uint8_t* idx_ptr, uint8_t array_len) {
                y_coord_to_z_table_idx = idx_ptr;
                std::fill(y_coord_to_z_table_idx, y_coord_to_z_table_idx + array_len, INVALID_INDEX);
            }
        };

        struct XLevelTable {
            std::vector<YLevelTable> y_tables;
            uint8_t* x_coord_to_y_table_idx = nullptr;
            
            void initialize_with_pool(uint8_t* idx_ptr,uint8_t array_len) {
                x_coord_to_y_table_idx = idx_ptr;
                std::fill(x_coord_to_y_table_idx, x_coord_to_y_table_idx + array_len, INVALID_INDEX);
            }
        };

        // Member variables
        float min_query_radius;
        float max_query_radius;
        float point_radius;
        
        Point workspace_aabb_min;
        Point workspace_aabb_max;
        Point global_aabb_min;
        Point global_aabb_max;
        
        float inverse_scale_factor;
        uint8_t grid_width;
        uint8_t index_array_len;
        
        std::unique_ptr<float[], decltype(&std::free)> point_coord_pool{nullptr, &std::free};
        size_t point_coord_pool_size = 0;
        size_t point_coord_pool_used = 0;
        size_t estimated_max_point_per_voxel = 0;

        std::unique_ptr<uint8_t[], decltype(&std::free)> index_pool{nullptr, &std::free};
        size_t index_pool_size = 0;
        size_t index_pool_used = 0;

        XLevelTable x_level_table;
        
        FVectorT simd_global_min_x, simd_global_min_y, simd_global_min_z;
        FVectorT simd_global_max_x, simd_global_max_y, simd_global_max_z;
        FVectorT simd_workspace_min_x, simd_workspace_min_y, simd_workspace_min_z;

        // Construct a multilevel voxel grid for spatial queries.
        //
        // Parameters:
        // - points: Collection of 3D points to include in the grid
        // - min_radius: Minimum radius for collision queries (inclusive)
        // - max_radius: Maximum radius for collision queries (inclusive)
        // - workspace_aabb_min: Lower corner of the robot's workspace AABB
        // - workspace_aabb_max: Upper corner of the robot's workspace AABB
        // - point_radius: Radius associated with each point in the grid
        //
        // Assumption:
        // - Workspace is cubic
        MVT(
            const std::vector<Point>& points,
            const float min_radius,
            const float max_radius,
            const Point &workspace_aabb_min, 
            const Point &workspace_aabb_max, 
            const float point_radius) noexcept
            : min_query_radius{min_radius},
              max_query_radius{max_radius},
              workspace_aabb_min{workspace_aabb_min},
              workspace_aabb_max{workspace_aabb_max},
              point_radius{point_radius}
        {
            if (points.empty()) {
                global_aabb_min = Point{std::numeric_limits<float>::max(), 
                                        std::numeric_limits<float>::max(), 
                                        std::numeric_limits<float>::max()};
                global_aabb_max = Point{std::numeric_limits<float>::lowest(), 
                                        std::numeric_limits<float>::lowest(), 
                                        std::numeric_limits<float>::lowest()};
                return;
            }
            
            initialize_memory_pool();       
            build_spatial_grid(points);
            compute_global_bounds();
            setup_simd_vectors();

            // std::ofstream log_file("scripts/log/tmp.txt");
            // print_grid_structure(log_file);
            // print_voxel_details(0, 255, 0, 255, 0, 255, log_file);
        }

        // Copy constructor
        MVT(const MVT& other) 
            : min_query_radius(other.min_query_radius),
              max_query_radius(other.max_query_radius),
              point_radius(other.point_radius),
              workspace_aabb_min(other.workspace_aabb_min),
              workspace_aabb_max(other.workspace_aabb_max),
              global_aabb_min(other.global_aabb_min),
              global_aabb_max(other.global_aabb_max),
              inverse_scale_factor(other.inverse_scale_factor),
              grid_width(other.grid_width),
              index_array_len(other.index_array_len),
              point_coord_pool_size(other.point_coord_pool_size),
              point_coord_pool_used(other.point_coord_pool_used),
              estimated_max_point_per_voxel(other.estimated_max_point_per_voxel),
              index_pool_size(other.index_pool_size),
              index_pool_used(other.index_pool_used) 
        {   
            copy_memory_pool(other);
 
            x_level_table = other.x_level_table;
            
            // Update pointers in voxels to point to new pool
            update_voxel_pointers_after_copy(other);
            setup_simd_vectors();
        }

        // Copy assignment
        MVT& operator=(const MVT& other) {
            if (this != &other) {
                MVT temp(other);  // copy constructor
                *this = std::move(temp);  // move assignment
            }
            return *this;
        }

        // Test whether a sphere centered at 'center' with radius 'radius' collides with any
        // point in this grid.
        // Returns true if in collision, false otherwise.
        [[nodiscard]] auto collides(const Point& center, float radius) const noexcept -> bool
        {
            // Calculate the expanded query radius (sphere radius + point radius)
            const float query_radius = radius + point_radius;

            // Check if query sphere overlaps with global AABB
            if (center[0] + query_radius < global_aabb_min[0] ||
                center[0] - query_radius > global_aabb_max[0] ||
                center[1] + query_radius < global_aabb_min[1] ||
                center[1] - query_radius > global_aabb_max[1] ||
                center[2] + query_radius < global_aabb_min[2] ||
                center[2] - query_radius > global_aabb_max[2]) {
                return false; // No collision possible
            }

            const float query_radius_squared = query_radius * query_radius;
                
            const float grid_query_radius = query_radius * inverse_scale_factor;
            const float grid_center_x_float = (center[0] - workspace_aabb_min[0]) * inverse_scale_factor;
            const float grid_center_y_float = (center[1] - workspace_aabb_min[1]) * inverse_scale_factor;
            const float grid_center_z_float = (center[2] - workspace_aabb_min[2]) * inverse_scale_factor;
            
            // Calculate actual sphere bounds in grid space
            const int min_x_touched = static_cast<int>(grid_center_x_float - grid_query_radius);
            const int max_x_touched = static_cast<int>(grid_center_x_float + grid_query_radius);
            const int min_y_touched = static_cast<int>(grid_center_y_float - grid_query_radius);
            const int max_y_touched = static_cast<int>(grid_center_y_float + grid_query_radius);
            const int min_z_touched = static_cast<int>(grid_center_z_float - grid_query_radius);
            const int max_z_touched = static_cast<int>(grid_center_z_float + grid_query_radius);

            // Determine voxel bounds to check (at most 26-neighborhood)
            const int min_x = std::max({0, min_x_touched, static_cast<int>(grid_center_x_float) - 1});
            const int max_x = std::min({254, max_x_touched, static_cast<int>(grid_center_x_float) + 1});
            const int min_y = std::max({0, min_y_touched, static_cast<int>(grid_center_y_float) - 1});
            const int max_y = std::min({254, max_y_touched, static_cast<int>(grid_center_y_float) + 1});
            const int min_z = std::max({0, min_z_touched, static_cast<int>(grid_center_z_float) - 1});
            const int max_z = std::min({254, max_z_touched, static_cast<int>(grid_center_z_float) + 1});


            // Iterate through potentially overlapping voxels
            for (int voxel_x = min_x; voxel_x <= max_x; ++voxel_x) {
                const uint8_t y_level_index = x_level_table.x_coord_to_y_table_idx[voxel_x];
                if (y_level_index == INVALID_INDEX) continue;
                
                const auto& y_level_table = x_level_table.y_tables[y_level_index];
                
                for (int voxel_y = min_y; voxel_y <= max_y; ++voxel_y) {
                    const uint8_t z_level_index = y_level_table.y_coord_to_z_table_idx[voxel_y];
                    if (z_level_index == INVALID_INDEX) continue;
                    
                    const auto& z_level_table = y_level_table.z_tables[z_level_index];
                    
                    for (int voxel_z = min_z; voxel_z <= max_z; ++voxel_z) {
                        const uint8_t voxel_index = z_level_table.z_coord_to_voxel_idx[voxel_z];
                        if (voxel_index == INVALID_INDEX) continue;
                        
                        const auto& voxel = z_level_table.voxels[voxel_index];
                        
                        // Bounding box check first
                        if (center[0] + query_radius < voxel.bbox_min[0] ||
                            center[0] - query_radius > voxel.bbox_max[0] ||
                            center[1] + query_radius < voxel.bbox_min[1] ||
                            center[1] - query_radius > voxel.bbox_max[1] ||
                            center[2] + query_radius < voxel.bbox_min[2] ||
                            center[2] - query_radius > voxel.bbox_max[2]) {
                            continue;
                        }
                        
                        // Check each point in the voxel
                        const size_t num_points = voxel.point_count;
                        for (size_t i = 0; i < num_points; ++i) {
                            const float dx = center[0] - voxel.x_coords[i];
                            const float dy = center[1] - voxel.y_coords[i];
                            const float dz = center[2] - voxel.z_coords[i];
                            const float distance_squared = dx * dx + dy * dy + dz * dz;
                            
                            if (distance_squared <= query_radius_squared) {
                                return true; // Collision detected
                            }
                        }
                    }
                }
            }
            return false; // No collision
        }

        // Determine whether any of a set of spheres collides with a point in this grid.
        //
        // Template parameters:
        // - FVectorT: Type of a SIMD vector of floats
        // - IVectorT: Type of a SIMD vector of integer indexes
        //
        // Parameters:
        // - centers: (x, y, z) struct-of-arrays of the centers of each sphere
        // - radii: SIMD vector of the radii of each sphere
        auto inline collides_simd(const std::array<FVectorT, 3> &centers, 
                          FVectorT radii) const noexcept -> bool
        {
            constexpr size_t SIMD_WIDTH = FVectorT::num_scalars;
    
            // Broadcast constants to SIMD vectors
            const FVectorT point_radius_vec = FVectorT::fill(point_radius);
            const FVectorT query_radii = radii + point_radius_vec;
  
            // Global AABB check - if all spheres are outside, no collision possible
            const auto outside_x_low = (centers[0] + query_radii) < simd_global_min_x;
            const auto outside_x_high = simd_global_max_x < (centers[0] - query_radii);
            const auto outside_y_low = (centers[1] + query_radii) < simd_global_min_y;
            const auto outside_y_high = simd_global_max_y < (centers[1] - query_radii);
            const auto outside_z_low = (centers[2] + query_radii) < simd_global_min_z;
            const auto outside_z_high = simd_global_max_z < (centers[2] - query_radii);
            
            const auto outside_mask = outside_x_low | outside_x_high | 
                                      outside_y_low | outside_y_high | 
                                      outside_z_low | outside_z_high;
            
            if (outside_mask.all()) {
                return false;
            }
            
            const FVectorT inv_scale = FVectorT::fill(inverse_scale_factor);
            
            // Convert centers to grid coordinates (vectorized)
            const FVectorT grid_center_x = ((centers[0] - simd_workspace_min_x) * inv_scale);
            const FVectorT grid_center_y = ((centers[1] - simd_workspace_min_y) * inv_scale);
            const FVectorT grid_center_z = ((centers[2] - simd_workspace_min_z) * inv_scale);

            const auto query_radii_squared = query_radii * query_radii;

            // For each sphere that's not completely outside, do detailed collision check
            // Extract individual spheres for detailed checking
            const auto centers_x_array = centers[0].to_array();
            const auto centers_y_array = centers[1].to_array();
            const auto centers_z_array = centers[2].to_array();
            const auto query_radii_array = query_radii.to_array();
            const auto query_radii_squared_array = query_radii_squared.to_array();
            const auto outside_array = outside_mask.to_array();
            const auto grid_x_array = grid_center_x.to_array();
            const auto grid_y_array = grid_center_y.to_array();
            const auto grid_z_array = grid_center_z.to_array();
            
            for (size_t i = 0; i < SIMD_WIDTH; ++i) {
                // Skip spheres that are completely outside global AABB
                if (outside_array[i] != 0) {
                    continue;
                }
                
                const Point center = {centers_x_array[i], centers_y_array[i], centers_z_array[i]};
                const float query_radius = query_radii_array[i];
                const float query_radius_squared = query_radii_squared_array[i];
                const float grid_query_radius = query_radii_array[i] * inverse_scale_factor;

                const float grid_center_x_float = grid_x_array[i];
                const float grid_center_y_float = grid_y_array[i];
                const float grid_center_z_float = grid_z_array[i];
                
                // Calculate actual sphere bounds in grid space
                const int min_x_touched = static_cast<int>(grid_center_x_float - grid_query_radius);
                const int max_x_touched = static_cast<int>(grid_center_x_float + grid_query_radius);
                const int min_y_touched = static_cast<int>(grid_center_y_float - grid_query_radius);
                const int max_y_touched = static_cast<int>(grid_center_y_float + grid_query_radius);
                const int min_z_touched = static_cast<int>(grid_center_z_float - grid_query_radius);
                const int max_z_touched = static_cast<int>(grid_center_z_float + grid_query_radius);

                // Determine voxel bounds to check
                const int min_x = std::max({0, min_x_touched, static_cast<int>(grid_center_x_float) - 1});
                const int max_x = std::min({254, max_x_touched, static_cast<int>(grid_center_x_float) + 1});
                const int min_y = std::max({0, min_y_touched, static_cast<int>(grid_center_y_float) - 1});
                const int max_y = std::min({254, max_y_touched, static_cast<int>(grid_center_y_float) + 1});
                const int min_z = std::max({0, min_z_touched, static_cast<int>(grid_center_z_float) - 1});
                const int max_z = std::min({254, max_z_touched, static_cast<int>(grid_center_z_float) + 1});

                // Check voxels in the neighborhood
                for (int voxel_x = min_x; voxel_x <= max_x; ++voxel_x) {
                    const uint8_t y_level_index = x_level_table.x_coord_to_y_table_idx[voxel_x];
                    if (y_level_index == INVALID_INDEX) continue;
                    
                    const auto& y_level_table = x_level_table.y_tables[y_level_index];
                    
                    for (int voxel_y = min_y; voxel_y <= max_y; ++voxel_y) {
                        const uint8_t z_level_index = y_level_table.y_coord_to_z_table_idx[voxel_y];
                        if (z_level_index == INVALID_INDEX) continue;
                        
                        const auto& z_level_table = y_level_table.z_tables[z_level_index];
                        
                        for (int voxel_z = min_z; voxel_z <= max_z; ++voxel_z) {
                            const uint8_t voxel_index = z_level_table.z_coord_to_voxel_idx[voxel_z];
                            if (voxel_index == INVALID_INDEX) continue;
                            
                            const auto& voxel = z_level_table.voxels[voxel_index];
                            
                            // Voxel bounding box check
                            if (center[0] + query_radius < voxel.bbox_min[0] ||
                                center[0] - query_radius > voxel.bbox_max[0] ||
                                center[1] + query_radius < voxel.bbox_min[1] ||
                                center[1] - query_radius > voxel.bbox_max[1] ||
                                center[2] + query_radius < voxel.bbox_min[2] ||
                                center[2] - query_radius > voxel.bbox_max[2]) {
                                continue;
                            }
                            
                            const size_t num_points = voxel.point_count;
                            const auto* x_coords = voxel.x_coords;
                            const auto* y_coords = voxel.y_coords;
                            const auto* z_coords = voxel.z_coords;

                            // Option1 : Always do SIMD point-by-point collision check within this voxel
                            // Broadcast sphere center and radius for SIMD comparison
                            const FVectorT sphere_x = FVectorT::fill(center[0]);
                            const FVectorT sphere_y = FVectorT::fill(center[1]);
                            const FVectorT sphere_z = FVectorT::fill(center[2]);
                            const FVectorT sphere_radius_sq = FVectorT::fill(query_radius_squared);

                            // Process points in SIMD chunks
                            size_t point_idx = 0;
                            const size_t num_points_aligned = ALIGN_TO_SIMD_WIDTH(num_points);

                            // Process all points including zero padding with SIMD
                            for (; point_idx < num_points_aligned; point_idx += SIMD_WIDTH) {
                                // Load point coordinates
                                const FVectorT point_x(x_coords + point_idx); // aligned load
                                const FVectorT point_y(y_coords + point_idx);
                                const FVectorT point_z(z_coords + point_idx);
                                
                                // Calculate squared distances
                                const FVectorT dx = sphere_x - point_x;
                                const FVectorT dy = sphere_y - point_y;
                                const FVectorT dz = sphere_z - point_z;
                                const FVectorT dist_sq = dx * dx + dy * dy + dz * dz;
                                
                                // Check for collision
                                const auto collision_mask = dist_sq <= sphere_radius_sq;
                                if (collision_mask.any()) {
                                    return true; // Collision detected
                                }
                            }

                            // // Option 2: Hybrid
                            // // Hot path: optimize for voxels with few points (median is 3)
                            // if (num_points <= 4) {
                            //     // Scalar code with manual unrolling
                            //     const float cx = center[0];
                            //     const float cy = center[1];
                            //     const float cz = center[2];
                                
                            //     // Fully unrolled for common cases
                            //     switch (num_points) {
                            //         case 1: {
                            //             const float dx = cx - x_coords[0];
                            //             const float dy = cy - y_coords[0];
                            //             const float dz = cz - z_coords[0];
                            //             if (dx*dx + dy*dy + dz*dz <= query_radius_squared) return true;
                            //             break;
                            //         }
                            //         case 2: {
                            //             const float dx0 = cx - x_coords[0];
                            //             const float dy0 = cy - y_coords[0];
                            //             const float dz0 = cz - z_coords[0];
                            //             if (dx0*dx0 + dy0*dy0 + dz0*dz0 <= query_radius_squared) return true;
                                        
                            //             const float dx1 = cx - x_coords[1];
                            //             const float dy1 = cy - y_coords[1];
                            //             const float dz1 = cz - z_coords[1];
                            //             if (dx1*dx1 + dy1*dy1 + dz1*dz1 <= query_radius_squared) return true;
                            //             break;
                            //         }
                            //         case 3: {
                            //             const float dx0 = cx - x_coords[0];
                            //             const float dy0 = cy - y_coords[0];
                            //             const float dz0 = cz - z_coords[0];
                            //             if (dx0*dx0 + dy0*dy0 + dz0*dz0 <= query_radius_squared) return true;
                                        
                            //             const float dx1 = cx - x_coords[1];
                            //             const float dy1 = cy - y_coords[1];
                            //             const float dz1 = cz - z_coords[1];
                            //             if (dx1*dx1 + dy1*dy1 + dz1*dz1 <= query_radius_squared) return true;
                                        
                            //             const float dx2 = cx - x_coords[2];
                            //             const float dy2 = cy - y_coords[2];
                            //             const float dz2 = cz - z_coords[2];
                            //             if (dx2*dx2 + dy2*dy2 + dz2*dz2 <= query_radius_squared) return true;
                            //             break;
                            //         }
                            //         case 4: {
                            //             // Process 4 points
                            //             for (size_t i = 0; i < 4; ++i) {
                            //                 const float dx = cx - x_coords[i];
                            //                 const float dy = cy - y_coords[i];
                            //                 const float dz = cz - z_coords[i];
                            //                 if (dx*dx + dy*dy + dz*dz <= query_radius_squared) return true;
                            //             }
                            //             break;
                            //         }
                            //     }
                            // } else {
                            //     // Use SIMD for larger voxels (5+ points)
                            //     const FVectorT sphere_x = FVectorT::fill(center[0]);
                            //     const FVectorT sphere_y = FVectorT::fill(center[1]);
                            //     const FVectorT sphere_z = FVectorT::fill(center[2]);
                            //     const FVectorT sphere_radius_sq = FVectorT::fill(query_radius_squared);

                            //     size_t point_idx = 0;
                            //     constexpr size_t SIMD_MASK = ~(SIMD_WIDTH - 1);
                            //     const size_t num_points_aligned = (num_points + SIMD_WIDTH - 1) & SIMD_MASK;

                            //     for (; point_idx < num_points_aligned; point_idx += SIMD_WIDTH) {
                            //         const FVectorT point_x(x_coords + point_idx);
                            //         const FVectorT point_y(y_coords + point_idx);
                            //         const FVectorT point_z(z_coords + point_idx);
                                    
                            //         const FVectorT dx = sphere_x - point_x;
                            //         const FVectorT dy = sphere_y - point_y;
                            //         const FVectorT dz = sphere_z - point_z;
                            //         const FVectorT dist_sq = dx * dx + dy * dy + dz * dz;
                                    
                            //         const auto collision_mask = dist_sq <= sphere_radius_sq;
                            //         if (collision_mask.any()) {
                            //             return true;
                            //         }
                            //     }
                            // }

                            // // Option 3: Do serial collision checking
                            // size_t point_idx = 0; 
                            // for (; point_idx < num_points; ++point_idx) {
                            //     const float dx = center[0] - x_coords[point_idx];
                            //     const float dy = center[1] - y_coords[point_idx];
                            //     const float dz = center[2] - z_coords[point_idx];

                            //     if (dx * dx + dy * dy + dz * dz <= query_radius_squared) {
                            //         return true; // Collision detected
                            //     }
                            // }
                        }
                    }
                }
            }

            // // Call serial collision checking
            // constexpr size_t SIMD_WIDTH = FVectorT::num_scalars;
            // const auto centers_x = centers[0].to_array();
            // const auto centers_y = centers[1].to_array();
            // const auto centers_z = centers[2].to_array();
            // const auto radiivec = radii.to_array();
            
            // for (size_t i = 0; i < SIMD_WIDTH; ++i) {
            //     Point center = {centers_x[i], centers_y[i], centers_z[i]};
            //     float radius = radiivec[i];
                
            //     if (collides(center, radius)) {
            //         return true; // Early exit if any sphere collides
            //     }
            // }
            
            return false; // No collisions detected
        }
        
        // Print structure of the level hashing grid
        void print_grid_structure(std::ostream& os = std::cout) const {
            os << "=== Level Hashing Grid Structure ===" << std::endl;
            os << "Workspace bounds: [" 
               << workspace_aabb_min[0] << ", " 
               << workspace_aabb_min[1] << ", " 
               << workspace_aabb_min[2] << "] to ["
               << workspace_aabb_max[0] << ", " 
               << workspace_aabb_max[1] << ", " 
               << workspace_aabb_max[2] << "]" << std::endl;
            os << "Query radius range: [" << min_query_radius 
               << ", " << max_query_radius << "]" << std::endl;
            os << "Point radius: " << point_radius << std::endl;
            os << "Inverse scale factor: " << inverse_scale_factor << std::endl;
            os << std::endl;

            // Count statistics
            size_t total_voxels = 0;
            size_t total_points = 0;
            size_t non_empty_x_entries = 0;
            size_t non_empty_y_entries = 0;
            size_t non_empty_z_entries = 0;
            
            // Collect voxel statistics
            std::vector<size_t> points_per_voxel;
            
            // Traverse the three-level structure
            for (size_t x = 0; x < index_array_len; ++x) {
                if (x_level_table.x_coord_to_y_table_idx[x] != INVALID_INDEX) {
                    non_empty_x_entries++;
                    uint8_t y_level_idx = x_level_table.x_coord_to_y_table_idx[x];
                    const auto& y_level = x_level_table.y_tables[y_level_idx];
                    
                    for (size_t y = 0; y < index_array_len; ++y) {
                        if (y_level.y_coord_to_z_table_idx[y] != INVALID_INDEX) {
                            non_empty_y_entries++;
                            uint8_t z_level_idx = y_level.y_coord_to_z_table_idx[y];
                            const auto& z_level = y_level.z_tables[z_level_idx];
                            
                            for (size_t z = 0; z < index_array_len; ++z) {
                                if (z_level.z_coord_to_voxel_idx[z] != INVALID_INDEX) {
                                    non_empty_z_entries++;
                                    uint8_t voxel_idx = z_level.z_coord_to_voxel_idx[z];
                                    const auto& voxel = z_level.voxels[voxel_idx];
                                    size_t voxel_point_count = voxel.point_count;
                                    
                                    total_voxels++;
                                    total_points += voxel_point_count;
                                    points_per_voxel.push_back(voxel_point_count);
                                }
                            }
                        }
                    }
                }
            }
            
            os << "=== Grid Statistics ===" << std::endl;
            os << "Total points stored: " << total_points << std::endl;
            os << "Total voxels: " << total_voxels << std::endl;
            os << "Non-empty X entries: " << non_empty_x_entries << " / " << (size_t)index_array_len << std::endl;
            os << "Non-empty Y entries: " << non_empty_y_entries << " (total across all X)" << std::endl;
            os << "Non-empty Z entries: " << non_empty_z_entries << " (total across all Y)" << std::endl;
            
            if (!points_per_voxel.empty()) {
                // Calculate statistics
                std::sort(points_per_voxel.begin(), points_per_voxel.end());
                size_t min_points = points_per_voxel.front();
                size_t max_points = points_per_voxel.back();
                double avg_points = static_cast<double>(total_points) / total_voxels;
                size_t median_points = points_per_voxel[points_per_voxel.size() / 2];
                
                os << "\n=== Points per Voxel Statistics ===" << std::endl;
                os << "Minimum: " << min_points << std::endl;
                os << "Maximum: " << max_points << std::endl;
                os << "Average: " << std::fixed << std::setprecision(2) << avg_points << std::endl;
                os << "Median: " << median_points << std::endl;
            }
            os << "\n=== Memory Usage Estimate ===" << std::endl;

            // Calculate actual memory usage including dynamic allocations
            size_t x_level_table_memory = sizeof(XLevelTable);
            size_t y_tables_memory = 0;
            size_t z_tables_memory = 0;
            size_t voxels_memory = 0;
            size_t point_data_memory = 0;

            os << "Calculation breakdown:" << std::endl;
            os << "  Root table struct: " << sizeof(XLevelTable) << " bytes" << std::endl;

            // Memory for root hash table's lookup array (always index_array_len entries)
            size_t root_lookup_array = index_array_len * sizeof(uint8_t);
            x_level_table_memory += root_lookup_array;
            os << "  Root lookup array (" << (size_t)index_array_len << " * " << sizeof(uint8_t) << "): " << root_lookup_array << " bytes" << std::endl;

            // Memory for Y-level tables vector in root
            size_t root_y_vector = x_level_table.y_tables.capacity() * sizeof(YLevelTable);
            x_level_table_memory += root_y_vector;
            os << "  Root Y-tables vector (" << x_level_table.y_tables.capacity() 
            << " * " << sizeof(YLevelTable) << "): " << root_y_vector << " bytes" << std::endl;

            // Counters for calculation breakdown
            size_t y_table_count = 0;
            size_t z_table_count = 0;
            size_t voxel_count = 0;
            size_t total_coordinate_capacity = 0;

            // Traverse the three-level structure to calculate actual memory usage
            for (size_t x = 0; x < index_array_len; ++x) {
                if (x_level_table.x_coord_to_y_table_idx[x] != INVALID_INDEX) {
                    uint8_t y_level_idx = x_level_table.x_coord_to_y_table_idx[x];
                    const auto& y_level = x_level_table.y_tables[y_level_idx];
                    
                    // Memory for this Y-level table
                    y_table_count++;
                    y_tables_memory += sizeof(YLevelTable);
                    y_tables_memory += index_array_len * sizeof(uint8_t); // y_coord_to_z_table_idx array
                    y_tables_memory += y_level.z_tables.capacity() * sizeof(ZLevelTable);
                    
                    for (size_t y = 0; y < index_array_len; ++y) {
                        if (y_level.y_coord_to_z_table_idx[y] != INVALID_INDEX) {
                            uint8_t z_level_idx = y_level.y_coord_to_z_table_idx[y];
                            const auto& z_level = y_level.z_tables[z_level_idx];
                            
                            // Memory for this Z-level table
                            z_table_count++;
                            z_tables_memory += sizeof(ZLevelTable);
                            z_tables_memory += index_array_len * sizeof(uint8_t); // z_coord_to_voxel_idx array
                            z_tables_memory += z_level.voxels.capacity() * sizeof(Voxel);
                            
                            for (size_t z = 0; z < index_array_len; ++z) {
                                if (z_level.z_coord_to_voxel_idx[z] != INVALID_INDEX) {
                                    uint8_t voxel_idx = z_level.z_coord_to_voxel_idx[z];
                                    const auto& voxel = z_level.voxels[voxel_idx];
                                    
                                    // Memory for this voxel's data
                                    voxel_count++;
                                    voxels_memory += sizeof(Voxel);
                                    voxels_memory += 2 * sizeof(Point); // bounding_box_min and max
                                    
                                    // Memory for the coordinate vectors
                                    size_t voxel_coord_capacity = voxel.capacity * 3;
                                    total_coordinate_capacity += voxel_coord_capacity;
                                    point_data_memory += voxel_coord_capacity * sizeof(float);
                                }
                            }
                        }
                    }
                }
            }

            os << "\nDetailed calculations:" << std::endl;
            os << "  Y-level tables (" << y_table_count << " tables):" << std::endl;
            os << "    - Structs: " << y_table_count << " * " << sizeof(YLevelTable) 
            << " = " << y_table_count * sizeof(YLevelTable) << " bytes" << std::endl;
            os << "    - Lookup arrays: " << y_table_count << " * " << (size_t)index_array_len << " * " << sizeof(uint8_t)
            << " = " << y_table_count * index_array_len * sizeof(uint8_t) << " bytes" << std::endl;

            os << "  Z-level tables (" << z_table_count << " tables):" << std::endl;
            os << "    - Structs: " << z_table_count << " * " << sizeof(ZLevelTable)
            << " = " << z_table_count * sizeof(ZLevelTable) << " bytes" << std::endl;
            os << "    - Lookup arrays: " << z_table_count << " * " << (size_t)index_array_len << " * " << sizeof(uint8_t)
            << " = " << z_table_count * index_array_len * sizeof(uint8_t) << " bytes" << std::endl;

            os << "  Voxels (" << voxel_count << " voxels):" << std::endl;
            os << "    - Structs + bounding boxes: " << voxel_count << " * (" 
            << sizeof(Voxel) << " + " << (2 * sizeof(Point)) << ") = " 
            << voxel_count * (sizeof(Voxel) + 2 * sizeof(Point)) << " bytes" << std::endl;

            os << "  Point coordinates:" << std::endl;
            os << "    - Total capacity: " << total_coordinate_capacity << " floats" << std::endl;
            os << "    - Memory: " << total_coordinate_capacity << " * " << sizeof(float)
            << " = " << point_data_memory << " bytes" << std::endl;

            size_t total_hash_table_overhead = x_level_table_memory + y_tables_memory + z_tables_memory + voxels_memory;
            size_t total_memory = total_hash_table_overhead + point_data_memory;

            os << "Root hash table: " << x_level_table_memory << " bytes" << std::endl;
            os << "Y-level tables: " << y_tables_memory << " bytes" << std::endl;
            os << "Z-level tables: " << z_tables_memory << " bytes" << std::endl;
            os << "Voxel structures: " << voxels_memory << " bytes" << std::endl;
            os << "Point coordinate data: " << point_data_memory << " bytes" << std::endl;
            os << "Total hash table overhead: " << total_hash_table_overhead << " bytes" << std::endl;
            os << "Total memory usage: " << total_memory << " bytes ("
            << std::fixed << std::setprecision(2) 
            << total_memory / (1024.0 * 1024.0) << " MB)" << std::endl;

            // Additional memory efficiency statistics
            if (total_points > 0) {
                double bytes_per_point = static_cast<double>(total_memory) / total_points;
                double overhead_ratio = static_cast<double>(total_hash_table_overhead) / point_data_memory;
                
                os << "\n=== Memory Efficiency ===" << std::endl;
                os << "Bytes per point: " << std::fixed << std::setprecision(2) << bytes_per_point << std::endl;
                os << "Overhead ratio: " << std::fixed << std::setprecision(2) << overhead_ratio 
                << " (hash tables / point data)" << std::endl;
            }
        }

        // Print detailed voxel contents for a specific region
        void print_voxel_details(int x_start, int x_end, 
                                int y_start, int y_end,
                                int z_start, int z_end,
                                std::ostream& os = std::cout) const {
            os << "\n=== Voxel Details for Region [" 
               << x_start << "-" << x_end << ", "
               << y_start << "-" << y_end << ", "
               << z_start << "-" << z_end << "] ===" << std::endl;
            
            for (int x = x_start; x <= x_end && x < index_array_len; ++x) {
                if (x_level_table.x_coord_to_y_table_idx[x] == INVALID_INDEX) continue;
                
                uint8_t y_level_idx = x_level_table.x_coord_to_y_table_idx[x];
                const auto& y_level = x_level_table.y_tables[y_level_idx];
                
                for (int y = y_start; y <= y_end && y < index_array_len; ++y) {
                    if (y_level.y_coord_to_z_table_idx[y] == INVALID_INDEX) continue;
                    
                    uint8_t z_level_idx = y_level.y_coord_to_z_table_idx[y];
                    const auto& z_level = y_level.z_tables[z_level_idx];
                    
                    for (int z = z_start; z <= z_end && z < index_array_len; ++z) {
                        if (z_level.z_coord_to_voxel_idx[z] == INVALID_INDEX) continue;
                        
                        uint8_t voxel_idx = z_level.z_coord_to_voxel_idx[z];
                        const auto& voxel = z_level.voxels[voxel_idx];
                        
                        os << "\nVoxel [" << x << ", " << y << ", " << z << "]:" << std::endl;
                        os << "  Points: " << voxel.point_count << std::endl;
                        os << "  AABB: [" 
                           << voxel.bbox_min[0] << ", "
                           << voxel.bbox_min[1] << ", "
                           << voxel.bbox_min[2] << "] to ["
                           << voxel.bbox_max[0] << ", "
                           << voxel.bbox_max[1] << ", "
                           << voxel.bbox_max[2] << "]" << std::endl;
                        
                        // Print first few points
                        os << "  First " << std::min(size_t(3), voxel.point_count) 
                           << " points:" << std::endl;
                        for (size_t i = 0; i < std::min(size_t(3), voxel.point_count); ++i) {
                            os << "    [" << voxel.x_coords[i] << ", "
                               << voxel.y_coords[i] << ", "
                               << voxel.z_coords[i] << "]" << std::endl;
                        }
                        if (voxel.point_count > 3) {
                            os << "    ... (" << voxel.point_count - 3 
                               << " more points)" << std::endl;
                        }
                    }
                }
            }
        }

        ~MVT() = default;

    private:
        void initialize_memory_pool() {        
            // 1. Estimate max # points per voxel
            //    Here we assume distance between points is 1 cm 
            constexpr size_t SIMD_WIDTH = FVectorT::num_scalars;
            const float estimated_max_point_per_voxel_dim = (max_query_radius) / (0.01 * 2);
            estimated_max_point_per_voxel = static_cast<unsigned int>(std::pow(estimated_max_point_per_voxel_dim, 3.0f));
            estimated_max_point_per_voxel = ALIGN_TO_SIMD_WIDTH(estimated_max_point_per_voxel);
            // std::cout << "Num point per voxel: " << estimated_max_point_per_voxel << std::endl;
           
            // 2. Calculate grid_width
            const float workspace_width = workspace_aabb_max[0] - workspace_aabb_min[0];
            grid_width = std::min(static_cast<uint8_t>(MAX_GRID_WIDTH), static_cast<uint8_t>(std::ceil(workspace_width / (max_query_radius))));
            
            // 3. Estimate pool size of point coord values
            //    Point coord pool will allocate "estimated_max_point_per_voxel * 3" floats to a newly initialized voxel
            //    Here we assume grid occupancy is 20%
            point_coord_pool_size = static_cast<size_t>(grid_width) * grid_width * grid_width * 0.2 * (estimated_max_point_per_voxel * 3);
            
            void* raw_ptr = nullptr;
            constexpr size_t COORD_ALIGNMENT = SIMD_WIDTH * sizeof(float); // AVX2: 8*4, NEON: 4*4
            if (posix_memalign(&raw_ptr, COORD_ALIGNMENT, point_coord_pool_size * sizeof(float)) != 0) {
                throw std::runtime_error("Failed to allocate aligned memory pool");
            }
            // std::cout << "Num float in pool: " << point_coord_pool_size << std::endl;
            point_coord_pool.reset(static_cast<float*>(raw_ptr));

            // 4. Estimate pool size of table indices
            const int estimated_tables = 1 + grid_width * 0.5 + grid_width * 0.5 * grid_width;
            index_array_len = next_power_of_two(grid_width);
            index_pool_size = estimated_tables * index_array_len;  // There are index_array_len entries per table
            
            void* idx_raw_ptr = nullptr;
            if (posix_memalign(&idx_raw_ptr, 64, index_pool_size * sizeof(uint8_t)) != 0) {
                throw std::runtime_error("Failed to allocate aligned index pool");
            }
            index_pool.reset(static_cast<uint8_t*>(idx_raw_ptr));
        }

        void build_spatial_grid(const std::vector<Point>& points) {
            const float workspace_width = workspace_aabb_max[0] - workspace_aabb_min[0];
            inverse_scale_factor = grid_width / workspace_width;
            
            x_level_table.initialize_with_pool(allocate_index_array(), index_array_len);
            x_level_table.y_tables.reserve(grid_width);
            
            for (const auto& point : points) {
                const int gx = static_cast<int>((point[0] - workspace_aabb_min[0]) * inverse_scale_factor);
                const int gy = static_cast<int>((point[1] - workspace_aabb_min[1]) * inverse_scale_factor);
                const int gz = static_cast<int>((point[2] - workspace_aabb_min[2]) * inverse_scale_factor);
                
                uint8_t vx = static_cast<uint8_t>(std::clamp(gx, 0, MAX_GRID_WIDTH - 1));
                uint8_t vy = static_cast<uint8_t>(std::clamp(gy, 0, MAX_GRID_WIDTH - 1));
                uint8_t vz = static_cast<uint8_t>(std::clamp(gz, 0, MAX_GRID_WIDTH - 1));

                insert_point(vx, vy, vz, point);
            }
        }

        void compute_global_bounds() {
            global_aabb_min = Point{std::numeric_limits<float>::max(), 
                             std::numeric_limits<float>::max(), 
                             std::numeric_limits<float>::max()};
            global_aabb_max = Point{std::numeric_limits<float>::lowest(), 
                             std::numeric_limits<float>::lowest(), 
                             std::numeric_limits<float>::lowest()};
            
            for (const auto& y_table : x_level_table.y_tables) {
                for (const auto& z_table : y_table.z_tables) {
                    for (const auto& voxel : z_table.voxels) {
                        global_aabb_min[0] = std::min(global_aabb_min[0], voxel.bbox_min[0]);
                        global_aabb_min[1] = std::min(global_aabb_min[1], voxel.bbox_min[1]);
                        global_aabb_min[2] = std::min(global_aabb_min[2], voxel.bbox_min[2]);
                        global_aabb_max[0] = std::max(global_aabb_max[0], voxel.bbox_max[0]);
                        global_aabb_max[1] = std::max(global_aabb_max[1], voxel.bbox_max[1]);
                        global_aabb_max[2] = std::max(global_aabb_max[2], voxel.bbox_max[2]);
                    }
                }
            }
        }

        void setup_simd_vectors() {
            simd_global_min_x = FVectorT::fill(global_aabb_min[0]);
            simd_global_min_y = FVectorT::fill(global_aabb_min[1]);
            simd_global_min_z = FVectorT::fill(global_aabb_min[2]);
            simd_global_max_x = FVectorT::fill(global_aabb_max[0]);
            simd_global_max_y = FVectorT::fill(global_aabb_max[1]);
            simd_global_max_z = FVectorT::fill(global_aabb_max[2]);
            
            simd_workspace_min_x = FVectorT::fill(workspace_aabb_min[0]);
            simd_workspace_min_y = FVectorT::fill(workspace_aabb_min[1]);
            simd_workspace_min_z = FVectorT::fill(workspace_aabb_min[2]);
        }

        void copy_memory_pool(const MVT& other) {
            // Copy point coordinate pool
            if (other.point_coord_pool && other.point_coord_pool_size > 0) {
                void* raw_ptr = nullptr;
                if (posix_memalign(&raw_ptr, 32, point_coord_pool_size * sizeof(float)) != 0) {
                    throw std::runtime_error("Failed to allocate aligned memory pool");
                }
                // std::cout << "(Copy mem pool) Num float in pool: " << point_coord_pool_size << std::endl;
                point_coord_pool.reset(static_cast<float*>(raw_ptr));
                std::memcpy(point_coord_pool.get(), other.point_coord_pool.get(), 
                point_coord_pool_used * sizeof(float));
            }

            // Copy index pool
            if (other.index_pool && other.index_pool_size > 0) {
                void* idx_raw_ptr = nullptr;
                if (posix_memalign(&idx_raw_ptr, 64, index_pool_size * sizeof(uint8_t)) != 0) {
                    throw std::runtime_error("Failed to allocate aligned index pool");
                }
                index_pool.reset(static_cast<uint8_t*>(idx_raw_ptr));
                std::memcpy(index_pool.get(), other.index_pool.get(), 
                            index_pool_used * sizeof(uint8_t));
            }
        }

        void update_voxel_pointers_after_copy(const MVT& other) {
            // Calculate offset between old and new pools
            const ptrdiff_t pool_offset = point_coord_pool.get() - other.point_coord_pool.get();
            const ptrdiff_t index_offset = index_pool.get() - other.index_pool.get();

            // Update root table index pointer
            if (x_level_table.x_coord_to_y_table_idx) {
                x_level_table.x_coord_to_y_table_idx += index_offset;
            }
            
            // Update all voxel pointers
            for (auto& y_table : x_level_table.y_tables) {
                if (y_table.y_coord_to_z_table_idx) {
                    y_table.y_coord_to_z_table_idx += index_offset;
                }

                for (auto& z_table : y_table.z_tables) {
                    if (z_table.z_coord_to_voxel_idx) {
                        z_table.z_coord_to_voxel_idx += index_offset;
                    }

                    for (auto& voxel : z_table.voxels) {
                        if (voxel.x_coords) {
                            voxel.x_coords += pool_offset;
                            voxel.y_coords += pool_offset;
                            voxel.z_coords += pool_offset;
                        }
                    }
                }
            }
        }
        
        void insert_point(uint8_t voxel_x, uint8_t voxel_y, uint8_t voxel_z, const Point& point) {
            // Level 1: Map X coordinate to Y-level table
            uint8_t y_level_index = x_level_table.x_coord_to_y_table_idx[voxel_x];
            if (y_level_index == INVALID_INDEX) {
                y_level_index = static_cast<uint8_t>(x_level_table.y_tables.size());
                x_level_table.x_coord_to_y_table_idx[voxel_x] = y_level_index;
                x_level_table.y_tables.emplace_back();

                // Initialize Y-level table with pool
                x_level_table.y_tables.back().initialize_with_pool(allocate_index_array(), index_array_len);
                x_level_table.y_tables.back().z_tables.reserve(grid_width);
            }
            
            auto& y_level_table = x_level_table.y_tables[y_level_index];
            
            // Level 2: Map Y coordinate to Z-level table
            uint8_t z_level_index = y_level_table.y_coord_to_z_table_idx[voxel_y];
            if (z_level_index == INVALID_INDEX) {
                z_level_index = static_cast<uint8_t>(y_level_table.z_tables.size());
                y_level_table.y_coord_to_z_table_idx[voxel_y] = z_level_index;
                y_level_table.z_tables.emplace_back();

                // Initialize Z-level table with pool
                y_level_table.z_tables.back().initialize_with_pool(allocate_index_array(), index_array_len);
                y_level_table.z_tables.back().voxels.reserve(grid_width);
            }
            
            auto& z_level_table = y_level_table.z_tables[z_level_index];
            
            // Level 3: Map Z coordinate to voxel
            uint8_t voxel_index = z_level_table.z_coord_to_voxel_idx[voxel_z];
            if (voxel_index == INVALID_INDEX) {
                voxel_index = static_cast<uint8_t>(z_level_table.voxels.size());
                z_level_table.z_coord_to_voxel_idx[voxel_z] = voxel_index;
                z_level_table.voxels.emplace_back();

                // Allocate space from pool
                float* x_ptr = allocate_coords(estimated_max_point_per_voxel);
                float* y_ptr = allocate_coords(estimated_max_point_per_voxel);
                float* z_ptr = allocate_coords(estimated_max_point_per_voxel);
                z_level_table.voxels.back().initialize_with_pool(x_ptr, y_ptr, z_ptr, estimated_max_point_per_voxel);
            }
            
            z_level_table.voxels[voxel_index].add_point(point);
        }

        float* allocate_coords(size_t count) {
            // Align count to SIMD boundary
            const size_t aligned_count = ALIGN_TO_SIMD_WIDTH(count);
            if (point_coord_pool_used + aligned_count > point_coord_pool_size) {
                throw std::runtime_error("Point coordinate pool exhausted");
            }
            
            float* result = point_coord_pool.get() + point_coord_pool_used;
            std::fill(result, result + aligned_count, 0.0f);

            point_coord_pool_used += aligned_count;
            return result;
        }

        uint8_t* allocate_index_array() {
            if (index_pool_used + index_array_len > index_pool_size) {
                throw std::runtime_error("Index pool exhausted");
            }
            
            uint8_t* result = index_pool.get() + index_pool_used;
            index_pool_used += index_array_len;

            return result;
        }

        unsigned int next_power_of_two(uint8_t n) {
            if (n == 0) return 1;
            n--;
        
            // Set all bits to the right of the MSB to 1
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
    
            n++;
            
            return n;
        }
    };

    
}  // namespace vamp::collision