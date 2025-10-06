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

namespace vamp::collision
{
    /**
     * Multi-level Voxel Table (MVT) - A hierarchical spatial data structure
     * for efficient collision detection using a three-level sparse table.
     * 
     * Assumption: all points to be inserted to MVT is in the given workspace bounds 
     */
    struct MVT
    {
        // ====================================================================
        // TYPE DEFINITIONS
        // ====================================================================
        
        using FVectorT = FloatVector<>;
        using IVectorT = IntVector<>;
        using VoxelIndex = uint32_t;
        
        static constexpr VoxelIndex INVALID_VOXEL_INDEX = std::numeric_limits<VoxelIndex>::max();
        static constexpr uint16_t MAX_GRID_WIDTH = std::numeric_limits<uint16_t>::max();
        
        // Three-level table types
        using ZLevelTable = uint32_t*;      // Z table: voxel indices
        using YLevelTable = uint32_t**;     // Y table: pointers to Z tables
        using XLevelTable = uint32_t***;    // X table: pointers to Y tables

        // ====================================================================
        // VOXEL STRUCTURE
        // ====================================================================
        
        struct alignas(32) Voxel {
            float* x_coords = nullptr;
            float* y_coords = nullptr;
            float* z_coords = nullptr;
            size_t point_count = 0;
            size_t capacity = 0;
            
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
                
                update_bounding_box(point);
                ++point_count;
            }

        private:
            void update_bounding_box(const Point& point) {
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
            }
        };

        // ====================================================================
        // MEMBER VARIABLES
        // ====================================================================
        
        // Query parameters
        float min_query_radius;
        float max_query_radius;
        float point_radius;
        
        // Spatial bounds
        Point workspace_aabb_min;
        Point workspace_aabb_max;
        Point global_aabb_min;
        Point global_aabb_max;
        
        // Grid configuration
        float inverse_scale_factor;
        uint16_t grid_width;
        uint16_t table_array_len;
        uint16_t z_table_array_len;
        
        // Memory pools
        std::unique_ptr<float[], decltype(&std::free)> point_coord_pool{nullptr, &std::free};
        size_t point_coord_pool_size = 0;
        size_t point_coord_pool_used = 0;
        size_t estimated_max_point_per_voxel = 0;
        
        std::unique_ptr<void*[], decltype(&std::free)> pointer_array_pool{nullptr, &std::free};
        size_t pointer_array_pool_size = 0;
        size_t pointer_array_pool_used = 0;
        
        std::unique_ptr<VoxelIndex[], decltype(&std::free)> voxel_index_pool{nullptr, &std::free};
        size_t voxel_index_pool_size = 0;
        size_t voxel_index_pool_used = 0;

        // Voxel storage and hierarchy entry
        std::vector<Voxel> voxel_storage;
        XLevelTable x_level_table = nullptr;
        
        // SIMD-optimized bounds
        FVectorT simd_global_min_x, simd_global_min_y, simd_global_min_z;
        FVectorT simd_global_max_x, simd_global_max_y, simd_global_max_z;
        FVectorT simd_workspace_min_x, simd_workspace_min_y, simd_workspace_min_z;

        // ====================================================================
        // CONSTRUCTOR & DESTRUCTOR
        // ====================================================================
        
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
                initialize_empty_bounds();
                return;
            }
            
            configure_grid();
            initialize_memory_pools();
            build_spatial_grid(points);
            compute_global_bounds();
            setup_simd_vectors();
        }

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
              table_array_len(other.table_array_len),
              z_table_array_len(other.z_table_array_len),
              point_coord_pool_size(other.point_coord_pool_size),
              point_coord_pool_used(other.point_coord_pool_used),
              estimated_max_point_per_voxel(other.estimated_max_point_per_voxel),
              pointer_array_pool_size(other.pointer_array_pool_size),
              pointer_array_pool_used(other.pointer_array_pool_used),
              voxel_index_pool_size(other.voxel_index_pool_size),
              voxel_index_pool_used(other.voxel_index_pool_used),
              voxel_storage(other.voxel_storage)
        {
            copy_memory_pools(other);
            update_pointers_after_copy(other);
            setup_simd_vectors();
        }

        ~MVT() = default;

        // ====================================================================
        // COLLISION DETECTION
        // ====================================================================
        
        // Scalar collision detection
        [[nodiscard]] auto collides(const Point& center, float radius) const noexcept -> bool
        {
            const float query_radius = radius + point_radius;
            const float query_radius_squared = query_radius * query_radius;

            // Early exit: Global AABB check
            if (center[0] + query_radius < global_aabb_min[0] ||
                center[0] - query_radius > global_aabb_max[0] ||
                center[1] + query_radius < global_aabb_min[1] ||
                center[1] - query_radius > global_aabb_max[1] ||
                center[2] + query_radius < global_aabb_min[2] ||
                center[2] - query_radius > global_aabb_max[2]) {
                return false;
            }

            // Compute grid space coordinates and query bounds
            const float grid_query_radius = std::min(1.0f, query_radius * inverse_scale_factor);
            const float grid_center_x_float = (center[0] - workspace_aabb_min[0]) * inverse_scale_factor;
            const float grid_center_y_float = (center[1] - workspace_aabb_min[1]) * inverse_scale_factor;
            const float grid_center_z_float = (center[2] - workspace_aabb_min[2]) * inverse_scale_factor;
            
            //Calculate voxel iteration bounds
            const uint16_t min_x = static_cast<uint16_t>(std::max(0.0f, (grid_center_x_float - grid_query_radius)));
            const uint16_t max_x = static_cast<uint16_t>(std::min(static_cast<float>(grid_width - 1), (grid_center_x_float + grid_query_radius)));
            const uint16_t min_y = static_cast<uint16_t>(std::max(0.0f, (grid_center_y_float - grid_query_radius)));
            const uint16_t max_y = static_cast<uint16_t>(std::min(static_cast<float>(grid_width - 1), (grid_center_y_float + grid_query_radius)));
            const uint16_t min_z = static_cast<uint16_t>(std::max(0.0f, (grid_center_z_float - grid_query_radius)));
            const uint16_t max_z = static_cast<uint16_t>(std::min(static_cast<float>(grid_width - 1), (grid_center_z_float + grid_query_radius)));

            // Traverse three-level spatial hierarchy
            for (uint16_t voxel_x = min_x; voxel_x <= max_x; ++voxel_x) {
                YLevelTable y_level_table = x_level_table[voxel_x];
                if (y_level_table == nullptr) continue;
                
                for (uint16_t voxel_y = min_y; voxel_y <= max_y; ++voxel_y) {
                    ZLevelTable z_level_table = y_level_table[voxel_y];
                    if (z_level_table == nullptr) continue;
                    
                    for (uint16_t voxel_z = min_z; voxel_z <= max_z; ++voxel_z) {
                        VoxelIndex voxel_index = z_level_table[voxel_z];
                        if (voxel_index == INVALID_VOXEL_INDEX) continue;
                        
                        const Voxel& voxel = voxel_storage[voxel_index];
                        
                        // Voxel-level AABB culling
                        if (center[0] + query_radius < voxel.bbox_min[0] ||
                            center[0] - query_radius > voxel.bbox_max[0] ||
                            center[1] + query_radius < voxel.bbox_min[1] ||
                            center[1] - query_radius > voxel.bbox_max[1] ||
                            center[2] + query_radius < voxel.bbox_min[2] ||
                            center[2] - query_radius > voxel.bbox_max[2]) {
                            continue;
                        }
                        
                        // Point-level collision detection
                        const size_t num_points = voxel.point_count;
                        for (size_t i = 0; i < num_points; ++i) {
                            const float dx = center[0] - voxel.x_coords[i];
                            const float dy = center[1] - voxel.y_coords[i];
                            const float dz = center[2] - voxel.z_coords[i];
                            const float distance_squared = dx * dx + dy * dy + dz * dz;
                            
                            if (distance_squared <= query_radius_squared) {
                                return true;
                            }
                        }
                    }
                }
            }
            
            return false;
        }

        // SIMD vectorized collision detection for multiple spheres
        auto inline collides_simd(const std::array<FVectorT, 3> &centers, 
                                FVectorT radii) const noexcept -> bool
        {
            constexpr size_t SIMD_WIDTH = FVectorT::num_scalars;

            // Compute query radii for all spheres
            const FVectorT point_radius_vec = FVectorT::fill(point_radius);
            const FVectorT query_radii = radii + point_radius_vec;

            // SIMD global AABB check - cull entire lanes that are completely outside
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
                return false;  // All spheres are outside global bounds
            }
            
            // Transform centers to grid space
            const FVectorT inv_scale = FVectorT::fill(inverse_scale_factor);
            const FVectorT grid_center_x = (centers[0] - simd_workspace_min_x) * inv_scale;
            const FVectorT grid_center_y = (centers[1] - simd_workspace_min_y) * inv_scale;
            const FVectorT grid_center_z = (centers[2] - simd_workspace_min_z) * inv_scale;
            const auto query_radii_squared = query_radii * query_radii;

            // Extract scalar arrays for per-sphere processing
            const auto centers_x_array = centers[0].to_array();
            const auto centers_y_array = centers[1].to_array();
            const auto centers_z_array = centers[2].to_array();
            const auto query_radii_array = query_radii.to_array();
            const auto query_radii_squared_array = query_radii_squared.to_array();
            const auto outside_array = outside_mask.to_array();
            const auto grid_x_array = grid_center_x.to_array();
            const auto grid_y_array = grid_center_y.to_array();
            const auto grid_z_array = grid_center_z.to_array();
            
            // Process each sphere individually
            for (size_t sphere_idx = 0; sphere_idx < SIMD_WIDTH; ++sphere_idx) {
                // Skip spheres that failed global AABB test
                if (outside_array[sphere_idx] != 0) {
                    continue;
                }
                
                // Extract sphere parameters
                const Point center = {centers_x_array[sphere_idx], centers_y_array[sphere_idx], centers_z_array[sphere_idx]};
                const float query_radius = query_radii_array[sphere_idx];
                const float query_radius_squared = query_radii_squared_array[sphere_idx];
                const float grid_query_radius = std::min(1.0f, query_radius * inverse_scale_factor);
                const float grid_center_x_float = grid_x_array[sphere_idx];
                const float grid_center_y_float = grid_y_array[sphere_idx];
                const float grid_center_z_float = grid_z_array[sphere_idx];
                
                // Calculate voxel iteration bounds for this sphere
                const uint16_t min_x = static_cast<uint16_t>(std::max(0.0f, (grid_center_x_float - grid_query_radius)));
                const uint16_t max_x = static_cast<uint16_t>(std::min(static_cast<float>(grid_width - 1), (grid_center_x_float + grid_query_radius)));
                const uint16_t min_y = static_cast<uint16_t>(std::max(0.0f, (grid_center_y_float - grid_query_radius)));
                const uint16_t max_y = static_cast<uint16_t>(std::min(static_cast<float>(grid_width - 1), (grid_center_y_float + grid_query_radius)));
                const uint16_t min_z = static_cast<uint16_t>(std::max(0.0f, (grid_center_z_float - grid_query_radius)));
                const uint16_t max_z = static_cast<uint16_t>(std::min(static_cast<float>(grid_width - 1), (grid_center_z_float + grid_query_radius)));

                // Traverse spatial hierarchy for this sphere
                for (uint16_t voxel_x = min_x; voxel_x <= max_x; ++voxel_x) {
                    YLevelTable y_level_table = x_level_table[voxel_x];
                    if (y_level_table == nullptr) continue;
                    
                    for (uint16_t voxel_y = min_y; voxel_y <= max_y; ++voxel_y) {
                        ZLevelTable z_level_table = y_level_table[voxel_y];
                        if (z_level_table == nullptr) continue;
                        
                        for (uint16_t voxel_z = min_z; voxel_z <= max_z; ++voxel_z) {
                            VoxelIndex voxel_index = z_level_table[voxel_z];
                            if (voxel_index == INVALID_VOXEL_INDEX) continue;
                            
                            const Voxel& voxel = voxel_storage[voxel_index];
                            
                            // Voxel-level AABB culling
                            if (center[0] + query_radius < voxel.bbox_min[0] ||
                                center[0] - query_radius > voxel.bbox_max[0] ||
                                center[1] + query_radius < voxel.bbox_min[1] ||
                                center[1] - query_radius > voxel.bbox_max[1] ||
                                center[2] + query_radius < voxel.bbox_min[2] ||
                                center[2] - query_radius > voxel.bbox_max[2]) {
                                continue;
                            }
                            
                            // SIMD point-level collision detection
                            const size_t num_points = voxel.point_count;
                            const auto* x_coords = voxel.x_coords;
                            const auto* y_coords = voxel.y_coords;
                            const auto* z_coords = voxel.z_coords;

                            const FVectorT sphere_x = FVectorT::fill(center[0]);
                            const FVectorT sphere_y = FVectorT::fill(center[1]);
                            const FVectorT sphere_z = FVectorT::fill(center[2]);
                            const FVectorT sphere_radius_sq = FVectorT::fill(query_radius_squared);

                            // Process points in SIMD chunks
                            for (size_t point_idx = 0; point_idx < num_points; point_idx += SIMD_WIDTH) {
                                const FVectorT point_x(x_coords + point_idx);
                                const FVectorT point_y(y_coords + point_idx);
                                const FVectorT point_z(z_coords + point_idx);
                                
                                const FVectorT dx = sphere_x - point_x;
                                const FVectorT dy = sphere_y - point_y;
                                const FVectorT dz = sphere_z - point_z;
                                const FVectorT dist_sq = dx * dx + dy * dy + dz * dz;
                                
                                const auto collision_mask = dist_sq <= sphere_radius_sq;
                                if (collision_mask.any()) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            
            return false;

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

            // return false; // No collisions detected
        }

    private:
        // ====================================================================
        // INITIALIZATION HELPERS
        // ====================================================================
        
        void initialize_empty_bounds() {
            constexpr float max_val = std::numeric_limits<float>::max();
            constexpr float min_val = std::numeric_limits<float>::lowest();
            
            global_aabb_min = Point{max_val, max_val, max_val};
            global_aabb_max = Point{min_val, min_val, min_val};
        }

        void configure_grid() {
            const float workspace_width = workspace_aabb_max[0] - workspace_aabb_min[0];
            
            grid_width = static_cast<uint16_t>(std::min(
                static_cast<uint32_t>(std::floor(workspace_width / max_query_radius)), // Empirically < 100 for manipulator robots
                static_cast<uint32_t>(MAX_GRID_WIDTH) // upper bound
            ));

            inverse_scale_factor = grid_width / workspace_width;
        }

        void initialize_memory_pools() {    
            initialize_point_coord_pool();
            initialize_pointer_array_pool();
            initialize_voxel_index_pool();
            initialize_voxel_storage();
        }

        void initialize_point_coord_pool() {
            estimated_max_point_per_voxel = std::pow(max_query_radius / 0.02, 3.0f);
            
            const unsigned int point_coord_array_in_bytes = std::max(
                next_power_of_two(static_cast<unsigned int>(estimated_max_point_per_voxel) * sizeof(float)),
                static_cast<unsigned int>(FVectorT::num_scalars * sizeof(float)) // lower bound
            );
        
            size_t point_coord_pool_size_in_bytes = 
                static_cast<size_t>(grid_width) * grid_width * grid_width * 0.1 * 
                static_cast<size_t>(point_coord_array_in_bytes) * 3;
            
            point_coord_pool_size = point_coord_pool_size_in_bytes / sizeof(float);
            estimated_max_point_per_voxel = static_cast<size_t>(point_coord_array_in_bytes) / sizeof(float);

            void* raw_ptr = nullptr;
            
            if (posix_memalign(&raw_ptr, 64, point_coord_pool_size_in_bytes) != 0) {
                throw std::runtime_error("Failed to allocate aligned memory pool");
            }
            
            point_coord_pool.reset(static_cast<float*>(raw_ptr));
        }

        void initialize_pointer_array_pool() {
            const size_t estimated_tables = 1 + grid_width;
            const size_t table_array_len_in_bytes = next_power_of_two(static_cast<unsigned int>(grid_width * sizeof(void*)));
            const size_t pointer_array_pool_size_in_bytes = estimated_tables * table_array_len_in_bytes;
            
            void* raw_ptr = nullptr;
            if (posix_memalign(&raw_ptr, 64, pointer_array_pool_size_in_bytes) != 0) {
                throw std::runtime_error("Failed to allocate aligned pointer array pool");
            }
            
            pointer_array_pool.reset(static_cast<void**>(raw_ptr));
            pointer_array_pool_size = pointer_array_pool_size_in_bytes / sizeof(void*);
            table_array_len = static_cast<uint16_t>(table_array_len_in_bytes / sizeof(void*));
        }

        void initialize_voxel_index_pool() {
            const size_t estimated_z_tables = static_cast<size_t>(grid_width) * grid_width * 0.5;
            const size_t z_table_size_in_bytes = next_power_of_two(static_cast<unsigned int>(grid_width * sizeof(VoxelIndex)));
            const size_t voxel_index_pool_size_in_bytes = estimated_z_tables * z_table_size_in_bytes;
            
            void* raw_ptr = nullptr;
            if (posix_memalign(&raw_ptr, 64, voxel_index_pool_size_in_bytes) != 0) {
                throw std::runtime_error("Failed to allocate voxel index pool");
            }
            
            voxel_index_pool.reset(static_cast<VoxelIndex*>(raw_ptr));
            voxel_index_pool_size = voxel_index_pool_size_in_bytes / sizeof(VoxelIndex);
            z_table_array_len = static_cast<uint16_t>(z_table_size_in_bytes / sizeof(VoxelIndex));
        }

        void initialize_voxel_storage() {
            const size_t estimated_voxel_count = 
                static_cast<size_t>(grid_width) * grid_width * grid_width * 0.1;
            voxel_storage.reserve(estimated_voxel_count);
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

        // ====================================================================
        // SPATIAL GRID CONSTRUCTION
        // ====================================================================
        
        void build_spatial_grid(const std::vector<Point>& points) {
            // Initialize root of three-level table hierarchy
            x_level_table = allocate_pointer_table<XLevelTable>();
            
            // Insert each point into the corresponding voxel
            for (const auto& point : points) {
                // Transform point coordinates to grid space
                const float voxel_x_float = (point[0] - workspace_aabb_min[0]) * inverse_scale_factor;
                const float voxel_y_float = (point[1] - workspace_aabb_min[1]) * inverse_scale_factor;
                const float voxel_z_float = (point[2] - workspace_aabb_min[2]) * inverse_scale_factor;
                if ((voxel_x_float >= grid_width) || (voxel_y_float >= grid_width) || (voxel_z_float >= grid_width)) {
                    std::cout << "warning: voxel coordinate (" << voxel_x_float << ", " << voxel_y_float  << ", " << voxel_z_float
                              << ") will be clamped within [0, " << grid_width - 1 << "]" << std::endl;
                }

                // Clamp to valid grid indices just in case
                const uint16_t voxel_x = static_cast<uint16_t>(std::clamp(voxel_x_float, 0.0f, static_cast<float>(grid_width - 1)));
                const uint16_t voxel_y = static_cast<uint16_t>(std::clamp(voxel_y_float, 0.0f, static_cast<float>(grid_width - 1)));
                const uint16_t voxel_z = static_cast<uint16_t>(std::clamp(voxel_z_float, 0.0f, static_cast<float>(grid_width - 1)));
                // Intervals are half-open [lower, upper) except for the last voxel

                // Level 1: Get or create Y-level table
                YLevelTable y_level_table = x_level_table[voxel_x];
                if (y_level_table == nullptr) {
                    y_level_table = allocate_pointer_table<YLevelTable>();
                    x_level_table[voxel_x] = y_level_table;
                }
                
                // Level 2: Get or create Z-level table (voxel index table)
                ZLevelTable z_level_table = y_level_table[voxel_y];
                if (z_level_table == nullptr) {
                    z_level_table = allocate_index_table();
                    y_level_table[voxel_y] = z_level_table;
                }
                
                // Level 3: Get or create voxel
                VoxelIndex voxel_index = z_level_table[voxel_z];
                if (voxel_index == INVALID_VOXEL_INDEX) {
                    // Check capacity before adding to prevent reallocation during insertion
                    if (voxel_storage.size() >= voxel_storage.capacity()) {
                        std::cout << "Voxel storage capacity exceeded. Please consider reserving larger space." << std::endl;
                    }
                    
                    // Create new voxel and assign index
                    voxel_index = static_cast<VoxelIndex>(voxel_storage.size());
                    voxel_storage.emplace_back();
                    z_level_table[voxel_z] = voxel_index;
                    
                    // Allocate coordinate storage from memory pool
                    float* x_ptr = allocate_coords(estimated_max_point_per_voxel);
                    float* y_ptr = allocate_coords(estimated_max_point_per_voxel);
                    float* z_ptr = allocate_coords(estimated_max_point_per_voxel);
                    voxel_storage[voxel_index].initialize_with_pool(x_ptr, y_ptr, z_ptr, estimated_max_point_per_voxel);
                }
                
                // Add point to voxel
                voxel_storage[voxel_index].add_point(point);
            }
        }

        void compute_global_bounds() {
            initialize_empty_bounds();
            
            for (const auto& voxel : voxel_storage) {
                if (voxel.point_count > 0) {
                    global_aabb_min[0] = std::min(global_aabb_min[0], voxel.bbox_min[0]);
                    global_aabb_min[1] = std::min(global_aabb_min[1], voxel.bbox_min[1]);
                    global_aabb_min[2] = std::min(global_aabb_min[2], voxel.bbox_min[2]);
                    global_aabb_max[0] = std::max(global_aabb_max[0], voxel.bbox_max[0]);
                    global_aabb_max[1] = std::max(global_aabb_max[1], voxel.bbox_max[1]);
                    global_aabb_max[2] = std::max(global_aabb_max[2], voxel.bbox_max[2]);
                }
            }
        }

        // ====================================================================
        // MEMORY POOL ALLOCATION
        // ====================================================================
        
        template<typename T>
        T allocate_pointer_table() {
            if (pointer_array_pool_used + table_array_len > pointer_array_pool_size) {
                throw std::runtime_error("Pointer array pool exhausted");
            }
            
            T result = reinterpret_cast<T>(pointer_array_pool.get() + pointer_array_pool_used);
            std::fill(result, result + table_array_len, nullptr);
            pointer_array_pool_used += table_array_len;
            return result;
        }

        ZLevelTable allocate_index_table() {
            if (voxel_index_pool_used + z_table_array_len > voxel_index_pool_size) {
                throw std::runtime_error("Voxel index pool exhausted");
            }
            
            ZLevelTable result = voxel_index_pool.get() + voxel_index_pool_used;
            std::fill(result, result + z_table_array_len, INVALID_VOXEL_INDEX);
            voxel_index_pool_used += z_table_array_len;
            return result;
        }

        float* allocate_coords(size_t count) {
            if (point_coord_pool_used + count > point_coord_pool_size) {
                throw std::runtime_error("Point coordinate pool exhausted");
            }
            
            float* result = point_coord_pool.get() + point_coord_pool_used;
            std::fill(result, result + count, std::numeric_limits<float>::infinity());
            point_coord_pool_used += count;
            return result;
        }

        // ====================================================================
        // COPY CONSTRUCTOR HELPERS
        // ====================================================================
        
        void copy_memory_pools(const MVT& other) {
            copy_point_coord_pool(other);
            copy_pointer_array_pool(other);
            copy_voxel_index_pool(other);
        }

        void copy_point_coord_pool(const MVT& other) {
            if (!other.point_coord_pool || other.point_coord_pool_size == 0) return;
            
            void* raw_ptr = nullptr;

            if (posix_memalign(&raw_ptr, 64, point_coord_pool_size * sizeof(float)) != 0) {
                throw std::runtime_error("Failed to allocate aligned memory pool");
            }
            
            point_coord_pool.reset(static_cast<float*>(raw_ptr));
            std::memcpy(point_coord_pool.get(), other.point_coord_pool.get(), 
                       point_coord_pool_used * sizeof(float));
        }

        void copy_pointer_array_pool(const MVT& other) {
            if (!other.pointer_array_pool || other.pointer_array_pool_size == 0) return;
            
            void* raw_ptr = nullptr;
            if (posix_memalign(&raw_ptr, 64, pointer_array_pool_size * sizeof(void*)) != 0) {
                throw std::runtime_error("Failed to allocate aligned pointer array pool");
            }
            
            pointer_array_pool.reset(static_cast<void**>(raw_ptr));
            std::memcpy(pointer_array_pool.get(), other.pointer_array_pool.get(), 
                       pointer_array_pool_used * sizeof(void*));
        }

        void copy_voxel_index_pool(const MVT& other) {
            if (!other.voxel_index_pool || other.voxel_index_pool_size == 0) return;
            
            void* raw_ptr = nullptr;
            if (posix_memalign(&raw_ptr, 64, voxel_index_pool_size * sizeof(VoxelIndex)) != 0) {
                throw std::runtime_error("Failed to allocate voxel index pool");
            }
            
            voxel_index_pool.reset(static_cast<VoxelIndex*>(raw_ptr));
            std::memcpy(voxel_index_pool.get(), other.voxel_index_pool.get(), 
                       voxel_index_pool_used * sizeof(VoxelIndex));
        }

        void update_pointers_after_copy(const MVT& other) {
            relocate_table_hierarchy(other);
            relocate_voxel_coordinates(other);
        }

        void relocate_table_hierarchy(const MVT& other) {
            const ptrdiff_t idx_pool_offset = reinterpret_cast<char*>(voxel_index_pool.get()) - 
                                              reinterpret_cast<char*>(other.voxel_index_pool.get());
            
            // Relocate root X-level table
            ptrdiff_t x_table_offset = reinterpret_cast<void**>(other.x_level_table) - 
                                       other.pointer_array_pool.get();
            x_level_table = reinterpret_cast<XLevelTable>(pointer_array_pool.get() + x_table_offset);
            
            // Fix pointers in X and Y tables
            for (size_t i = 0; i < table_array_len; ++i) {
                if (x_level_table[i] != nullptr) {
                    relocate_y_table(i, other, idx_pool_offset);
                }
            }
        }

        void relocate_y_table(size_t x_idx, const MVT& other, ptrdiff_t idx_pool_offset) {
            YLevelTable old_y_ptr = x_level_table[x_idx];
            ptrdiff_t y_offset = reinterpret_cast<void**>(old_y_ptr) - other.pointer_array_pool.get();
            x_level_table[x_idx] = reinterpret_cast<YLevelTable>(pointer_array_pool.get() + y_offset);
            
            for (size_t j = 0; j < z_table_array_len; ++j) {
                if (x_level_table[x_idx][j] != nullptr) {
                    relocate_z_table(x_idx, j, other, idx_pool_offset);
                }
            }
        }

        void relocate_z_table(size_t x_idx, size_t y_idx, const MVT& other, ptrdiff_t idx_pool_offset) {
            ZLevelTable old_z_ptr = x_level_table[x_idx][y_idx];
            ptrdiff_t z_offset = reinterpret_cast<const char*>(old_z_ptr) - 
                                reinterpret_cast<const char*>(other.voxel_index_pool.get());
            x_level_table[x_idx][y_idx] = reinterpret_cast<ZLevelTable>(
                reinterpret_cast<char*>(voxel_index_pool.get()) + z_offset);
        }

        void relocate_voxel_coordinates(const MVT& other) {
            for (size_t idx = 0; idx < voxel_storage.size(); ++idx) {
                auto& voxel = voxel_storage[idx];
                const auto& other_voxel = other.voxel_storage[idx];
                
                if (other_voxel.x_coords != nullptr) {
                    voxel.x_coords = point_coord_pool.get() + (other_voxel.x_coords - other.point_coord_pool.get());
                    voxel.y_coords = point_coord_pool.get() + (other_voxel.y_coords - other.point_coord_pool.get());
                    voxel.z_coords = point_coord_pool.get() + (other_voxel.z_coords - other.point_coord_pool.get());
                }
            }
        }

        // ====================================================================
        // UTILITY FUNCTIONS
        // ====================================================================
        
        unsigned int next_power_of_two(unsigned int n) {
            if (n == 0) return 1;
            n--;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            n++;
            return n;
        }

        void write_statistics(const std::string& filepath) const {
            std::ofstream out(filepath);
            if (!out.is_open()) {
                throw std::runtime_error("Failed to open file: " + filepath);
            }
            
            out << std::fixed << std::setprecision(6);
            write_basic_info(out);
            write_table_structure_stats(out);
            write_memory_usage_stats(out);
            write_efficiency_metrics(out);
            
            out.close();
        }

        void write_basic_info(std::ofstream& out) const {
            out << "========================================\n";
            out << "MVT STRUCTURE STATISTICS\n";
            out << "========================================\n\n";
            out << "--- Basic Information ---\n";
            
            Point workspace_size = {
                workspace_aabb_max[0] - workspace_aabb_min[0],
                workspace_aabb_max[1] - workspace_aabb_min[1],
                workspace_aabb_max[2] - workspace_aabb_min[2]
            };
            
            out << "Workspace AABB Size: [" << workspace_size[0] << ", " 
                << workspace_size[1] << ", " << workspace_size[2] << "]\n";
            out << "Query Radius Range: [" << min_query_radius << ", " << max_query_radius << "]\n";
            out << "Point Radius: " << point_radius << "\n";
            out << "Grid Width: " << static_cast<int>(grid_width) << "\n";
            out << "Total Possible Voxels: " << (static_cast<int>(grid_width) * grid_width * grid_width) << "\n";
            out << "Non-empty Voxels: " << voxel_storage.size() << "\n";
            
            write_point_statistics(out);
        }

        void write_point_statistics(std::ofstream& out) const {
            size_t total_points = 0;
            size_t max_points = 0;
            size_t min_points = std::numeric_limits<size_t>::max();
            
            for (const auto& voxel : voxel_storage) {
                total_points += voxel.point_count;
                max_points = std::max(max_points, voxel.point_count);
                if (voxel.point_count > 0) {
                    min_points = std::min(min_points, voxel.point_count);
                }
            }
            
            out << "Total Points: " << total_points << "\n";
            out << "Min Points in Voxel: " << (min_points == std::numeric_limits<size_t>::max() ? 0 : min_points) << "\n";
            out << "Max Points in Voxel: " << max_points << "\n";
            if (!voxel_storage.empty()) {
                out << "Average Points Per Voxel: " << (static_cast<double>(total_points) / voxel_storage.size()) << "\n";
            }
        }

        void write_table_structure_stats(std::ofstream& out) const {
            out << "\n--- Three Level Table Structure ---\n";
            
            size_t non_empty_x = 0, non_empty_y = 0, non_empty_z = 0;
            count_non_empty_tables(non_empty_x, non_empty_y, non_empty_z);
            
            out << "Non-empty X entries: " << non_empty_x << " / " << static_cast<int>(table_array_len) << "\n";
            out << "Non-empty Y entries: " << non_empty_y << " (across all X)\n";
            out << "Non-empty Z entries: " << non_empty_z << " (across all Y)\n";
            out << "Table occupancy: " << (static_cast<double>(non_empty_z) / (grid_width * grid_width * grid_width) * 100.0) << "%\n";
        }

        void count_non_empty_tables(size_t& x_count, size_t& y_count, size_t& z_count) const {
            for (size_t x = 0; x < table_array_len; ++x) {
                if (x_level_table[x] != nullptr) {
                    x_count++;
                    YLevelTable y_table = x_level_table[x];
                    
                    for (size_t y = 0; y < table_array_len; ++y) {
                        if (y_table[y] != nullptr) {
                            y_count++;
                            ZLevelTable z_table = y_table[y];
                            
                            for (size_t z = 0; z < z_table_array_len; ++z) {
                                if (z_table[z] != INVALID_VOXEL_INDEX) {
                                    z_count++;
                                }
                            }
                        }
                    }
                }
            }
        }

        void write_memory_usage_stats(std::ofstream& out) const {
            out << "\n--- Memory Pool Usage ---\n";
            
            write_pool_stats(out, "Point Coordinate Pool",
                           point_coord_pool_size * sizeof(float),
                           point_coord_pool_used * sizeof(float));
            
            write_pool_stats(out, "Pointer Array Pool",
                           pointer_array_pool_size * sizeof(void*),
                           pointer_array_pool_used * sizeof(void*));
            
            write_pool_stats(out, "Voxel Index Pool",
                           voxel_index_pool_size * sizeof(VoxelIndex),
                           voxel_index_pool_used * sizeof(VoxelIndex));
            
            size_t voxel_struct_bytes = voxel_storage.size() * sizeof(Voxel);
            size_t voxel_capacity_bytes = voxel_storage.capacity() * sizeof(Voxel);
            out << "Voxel Storage (metadata):\n";
            out << "  Used: " << voxel_struct_bytes << " bytes (" << (voxel_struct_bytes / 1024.0) << " KB)\n";
            out << "  Capacity: " << voxel_capacity_bytes << " bytes (" << (voxel_capacity_bytes / 1024.0) << " KB)\n";
        }

        void write_pool_stats(std::ofstream& out, const std::string& name, 
                            size_t allocated, size_t used) const {
            double usage_pct = allocated > 0 ? (static_cast<double>(used) / allocated) * 100.0 : 0.0;
            
            out << name << ":\n";
            out << "  Allocated: " << allocated << " bytes (" << (allocated / (1024.0 * 1024.0)) << " MB)\n";
            out << "  Used: " << used << " bytes (" << (used / (1024.0 * 1024.0)) << " MB)\n";
            out << "  Usage: " << usage_pct << "%\n";
        }

        void write_efficiency_metrics(std::ofstream& out) const {
            out << "\n--- Efficiency Metrics ---\n";
            
            size_t total_points = std::accumulate(voxel_storage.begin(), voxel_storage.end(), 0UL,
                [](size_t sum, const Voxel& v) { return sum + v.point_count; });
            
            size_t total_memory = point_coord_pool_size * sizeof(float) + 
                                 pointer_array_pool_size + 
                                 voxel_index_pool_size * sizeof(VoxelIndex) +
                                 voxel_storage.capacity() * sizeof(Voxel);
            
            if (total_points > 0) {
                out << "Bytes per point: " << (static_cast<double>(total_memory) / total_points) << "\n";
            }
            
            out << "\n========================================\n";
        }
    };
}  // namespace vamp::collision