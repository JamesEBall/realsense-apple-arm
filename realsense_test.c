#include <librealsense2/rs.h>
#include <librealsense2/h/rs_context.h>
#include <librealsense2/h/rs_device.h>
#include <librealsense2/h/rs_pipeline.h>
#include <librealsense2/h/rs_frame.h>
#include <librealsense2/h/rs_config.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#define WINDOW_NAME "RealSense D421 - Depth & IR"
#define MIN_DEPTH_MM 100  // 10cm minimum depth
#define MAX_DEPTH_MM 5000 // 5m maximum depth
#define KEY_WIDTH 400
#define KEY_HEIGHT 50

// Function to create depth colormap key
cv::Mat create_depth_key(int min_depth, int max_depth) {
    cv::Mat key(KEY_HEIGHT, KEY_WIDTH, CV_8UC3);
    for(int i = 0; i < key.cols; i++) {
        float normalized = (float)i / key.cols;
        cv::Vec3b color;
        // Rainbow colormap (red->yellow->green->cyan->blue->violet)
        if(normalized < 0.2) {
            // Red to Yellow
            color[0] = 0;
            color[1] = 255 * (normalized * 5);
            color[2] = 255;
        } else if(normalized < 0.4) {
            // Yellow to Green
            color[0] = 0;
            color[1] = 255;
            color[2] = 255 * (1 - (normalized - 0.2) * 5);
        } else if(normalized < 0.6) {
            // Green to Cyan
            color[0] = 255 * ((normalized - 0.4) * 5);
            color[1] = 255;
            color[2] = 0;
        } else if(normalized < 0.8) {
            // Cyan to Blue
            color[0] = 255;
            color[1] = 255 * (1 - (normalized - 0.6) * 5);
            color[2] = 0;
        } else {
            // Blue to Violet
            color[0] = 255;
            color[1] = 0;
            color[2] = 255 * ((normalized - 0.8) * 5);
        }
        key.col(i) = color;
    }
    
    // Add distance labels
    char label[32];
    float step = (max_depth - min_depth) / 5.0f;
    for(int i = 0; i <= 5; i++) {
        float depth = min_depth + i * step;
        if(depth < 1000) {
            snprintf(label, sizeof(label), "%.1fcm", depth/10.0f);
        } else {
            snprintf(label, sizeof(label), "%.1fm", depth/1000.0f);
        }
        cv::putText(key, label, cv::Point(i * KEY_WIDTH/5, KEY_HEIGHT-5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1);
    }
    
    return key;
}

// Function to find min and max depth values in the frame
void find_depth_range(const cv::Mat& depth_image, int& min_depth, int& max_depth) {
    min_depth = MAX_DEPTH_MM;
    max_depth = MIN_DEPTH_MM;
    
    for(int y = 0; y < depth_image.rows; y++) {
        for(int x = 0; x < depth_image.cols; x++) {
            uint16_t depth = depth_image.at<uint16_t>(y, x);
            if(depth > MIN_DEPTH_MM && depth < MAX_DEPTH_MM) {
                min_depth = std::min(min_depth, (int)depth);
                max_depth = std::max(max_depth, (int)depth);
            }
        }
    }
    
    // Ensure we have a valid range
    if(min_depth >= max_depth) {
        min_depth = MIN_DEPTH_MM;
        max_depth = MAX_DEPTH_MM;
    }
}

void check_error(rs2_error* e)
{
    if (e)
    {
        printf("rs_error was raised when calling %s(%s):\n", rs2_get_failed_function(e), rs2_get_failed_args(e));
        printf("    %s\n", rs2_get_error_message(e));
        exit(1);
    }
}

int main()
{
    rs2_error* e = 0;
    
    // Create a context object
    rs2_context* ctx = rs2_create_context(RS2_API_VERSION, &e);
    check_error(e);
    
    // Get the list of devices
    rs2_device_list* device_list = rs2_query_devices(ctx, &e);
    check_error(e);
    
    int device_count = rs2_get_device_count(device_list, &e);
    check_error(e);
    
    printf("Found %d RealSense devices\n", device_count);
    
    if (device_count == 0) {
        printf("No RealSense devices found!\n");
        return 1;
    }
    
    // Get the first device
    rs2_device* dev = rs2_create_device(device_list, 0, &e);
    check_error(e);
    
    // Get device info
    const char* name = rs2_get_device_info(dev, RS2_CAMERA_INFO_NAME, &e);
    check_error(e);
    printf("Device name: %s\n", name);
    
    const char* serial = rs2_get_device_info(dev, RS2_CAMERA_INFO_SERIAL_NUMBER, &e);
    check_error(e);
    printf("Serial number: %s\n", serial);
    
    // Create pipeline and config
    rs2_pipeline* pipeline = rs2_create_pipeline(ctx, &e);
    check_error(e);
    
    rs2_config* config = rs2_create_config(&e);
    check_error(e);
    
    // Enable both depth and infrared streams
    rs2_config_enable_stream(config, RS2_STREAM_DEPTH, 0, 640, 480, RS2_FORMAT_Z16, 30, &e);
    check_error(e);
    rs2_config_enable_stream(config, RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30, &e);
    check_error(e);
    
    // Start the pipeline
    rs2_pipeline_profile* pipeline_profile = rs2_pipeline_start_with_config(pipeline, config, &e);
    check_error(e);
    printf("Pipeline started successfully\n");

    // Create OpenCV window
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    
    // Variables for FPS calculation
    clock_t start_time = clock();
    int frame_count = 0;
    double fps = 0;
    
    // Variables for depth range
    int min_depth = MIN_DEPTH_MM;
    int max_depth = MAX_DEPTH_MM;
    int frames_since_update = 0;
    const int UPDATE_INTERVAL = 30; // Update range every 30 frames

    // Main loop
    while (1) {
        // Get frameset
        rs2_frame* frames = rs2_pipeline_wait_for_frames(pipeline, 5000, &e);
        check_error(e);
        
        // Get depth and IR frames
        rs2_frame* depth_frame = rs2_extract_frame(frames, 0, &e);
        check_error(e);
        rs2_frame* ir_frame = rs2_extract_frame(frames, 1, &e);
        check_error(e);
        
        // Get frame data
        const void* depth_data = rs2_get_frame_data(depth_frame, &e);
        check_error(e);
        const void* ir_data = rs2_get_frame_data(ir_frame, &e);
        check_error(e);
        
        int width = rs2_get_frame_width(depth_frame, &e);
        int height = rs2_get_frame_height(depth_frame, &e);

        // Create OpenCV matrices
        cv::Mat depth_image(height, width, CV_16UC1, (void*)depth_data);
        cv::Mat ir_image(height, width, CV_8UC1, (void*)ir_data);
        
        // Update depth range periodically
        frames_since_update++;
        if(frames_since_update >= UPDATE_INTERVAL) {
            find_depth_range(depth_image, min_depth, max_depth);
            frames_since_update = 0;
        }
        
        // Normalize depth for display
        cv::Mat depth_display;
        cv::normalize(depth_image, depth_display, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        
        // Apply rainbow colormap to depth
        cv::Mat depth_color;
        cv::applyColorMap(depth_display, depth_color, cv::COLORMAP_RAINBOW);
        
        // Create side-by-side display
        cv::Mat display;
        cv::Mat ir_color;
        cv::cvtColor(ir_image, ir_color, cv::COLOR_GRAY2BGR);
        cv::hconcat(std::vector<cv::Mat>{depth_color, ir_color}, display);
        
        // Calculate and display FPS
        frame_count++;
        double time_elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        if (time_elapsed >= 1.0) {
            fps = frame_count / time_elapsed;
            frame_count = 0;
            start_time = clock();
        }
        
        // Create depth key with current range
        cv::Mat depth_key = create_depth_key(min_depth, max_depth);
        
        // Add labels and FPS
        cv::putText(display, "Depth", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,255), 2);
        cv::putText(display, "Infrared", cv::Point(width + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,255), 2);
        char fps_str[32];
        snprintf(fps_str, sizeof(fps_str), "FPS: %.1f", fps);
        cv::putText(display, fps_str, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
        
        // Add depth key to bottom of display
        cv::Mat final_display;
        cv::copyMakeBorder(display, final_display, 0, depth_key.rows + 20, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        cv::putText(final_display, "Depth Scale:", cv::Point(10, display.rows + 15), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
        depth_key.copyTo(final_display(cv::Rect(10, display.rows + 20, depth_key.cols, depth_key.rows)));
        
        // Display the combined image
        cv::imshow(WINDOW_NAME, final_display);
        
        // Handle keyboard input
        char key = (char)cv::waitKey(1);
        if (key == 27) // ESC key
            break;
        
        // Cleanup frames
        rs2_release_frame(depth_frame);
        rs2_release_frame(ir_frame);
        rs2_release_frame(frames);
    }
    
    // Cleanup
    cv::destroyAllWindows();
    rs2_pipeline_stop(pipeline, &e);
    check_error(e);
    rs2_delete_pipeline_profile(pipeline_profile);
    rs2_delete_config(config);
    rs2_delete_pipeline(pipeline);
    rs2_delete_device(dev);
    rs2_delete_device_list(device_list);
    rs2_delete_context(ctx);
    
    return 0;
} 