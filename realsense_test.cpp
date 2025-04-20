#include <librealsense2/rs.hpp>
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
#include <iostream>
#include <chrono>

#define WINDOW_NAME "RealSense D421 - Depth & IR"
#define MIN_DEPTH_MM 100  // 10cm minimum depth
#define MAX_DEPTH_MM 5000 // 5m maximum depth
#define KEY_WIDTH 400
#define KEY_HEIGHT 50

// Function to map depth value to color
cv::Vec3b depth_to_color(uint16_t depth, int max_depth) {
    if (depth < MIN_DEPTH_MM || depth > max_depth) {
        return cv::Vec3b(0, 0, 0); // Black for invalid depths
    }
    
    float normalized = (float)(depth - MIN_DEPTH_MM) / (max_depth - MIN_DEPTH_MM);
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
    
    return color;
}

// Function to create depth colormap key
cv::Mat create_depth_key(int max_depth) {
    cv::Mat key(KEY_HEIGHT, KEY_WIDTH, CV_8UC3);
    for(int i = 0; i < key.cols; i++) {
        float normalized = (float)i / key.cols;
        uint16_t depth = MIN_DEPTH_MM + normalized * (max_depth - MIN_DEPTH_MM);
        key.col(i) = depth_to_color(depth, max_depth);
    }
    
    // Add distance labels
    char label[32];
    float step = (max_depth - MIN_DEPTH_MM) / 5.0f;
    for(int i = 0; i <= 5; i++) {
        float depth = MIN_DEPTH_MM + i * step;
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

// Function to find max depth value in the frame
void find_max_depth(const cv::Mat& depth_image, int& max_depth) {
    max_depth = MIN_DEPTH_MM;
    
    for(int y = 0; y < depth_image.rows; y++) {
        for(int x = 0; x < depth_image.cols; x++) {
            uint16_t depth = depth_image.at<uint16_t>(y, x);
            if(depth > MIN_DEPTH_MM && depth < MAX_DEPTH_MM) {
                max_depth = std::max(max_depth, (int)depth);
            }
        }
    }
    
    // Ensure we have a valid range
    if(max_depth <= MIN_DEPTH_MM) {
        max_depth = MAX_DEPTH_MM;
    }
}

// Function to apply image processing
void process_images(cv::Mat& depth_image, cv::Mat& ir_image) {
    // Find current max depth for visualization
    int max_depth;
    find_max_depth(depth_image, max_depth);

    // Apply contrast enhancement to IR image
    cv::Mat ir_enhanced;
    cv::equalizeHist(ir_image, ir_enhanced);
    ir_image = ir_enhanced;
    
    // Apply slight Gaussian blur to IR image to reduce noise
    cv::GaussianBlur(ir_image, ir_image, cv::Size(3, 3), 0);
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
    // Create a context object. This object owns the handles to all connected realsense devices
    rs2::context ctx;
    
    // Get a list of all connected devices
    auto devices = ctx.query_devices();
    if (devices.size() == 0) {
        std::cerr << "No RealSense devices found!" << std::endl;
        std::cerr << "Please check:" << std::endl;
        std::cerr << "1. Camera is properly connected" << std::endl;
        std::cerr << "2. Camera permissions are granted" << std::endl;
        std::cerr << "3. USB connection is stable" << std::endl;
        return 1;
    }

    // Print device information
    for (auto&& dev : devices) {
        std::cout << "Found device: " << dev.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
        std::cout << "    Serial number: " << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
        std::cout << "    Firmware version: " << dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) << std::endl;
    }

    // Create pipeline
    rs2::pipeline pipe;
    rs2::config cfg;

    // Enable depth and IR streams
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 640, 480, RS2_FORMAT_Y8, 30);

    try {
        // Start pipeline with configuration
        auto profile = pipe.start(cfg);
        
        // Get the device being used
        auto device = profile.get_device();
        std::cout << "Using device: " << device.get_info(RS2_CAMERA_INFO_NAME) << std::endl;

        // Create window
        cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

        // Variables for FPS calculation
        auto start_time = std::chrono::steady_clock::now();
        int frame_count = 0;
        float fps = 0;

        // Variables for depth range
        int max_depth = MAX_DEPTH_MM;
        int frames_since_update = 0;
        const int UPDATE_INTERVAL = 30; // Update range every 30 frames

        while (true) {
            // Wait for the next set of frames
            rs2::frameset frames;
            try {
                frames = pipe.wait_for_frames(5000); // 5 second timeout
            } catch (const rs2::error& e) {
                std::cerr << "Failed to get frames: " << e.what() << std::endl;
                continue;
            }

            try {
                // Get both depth and IR frames
                auto depth_frame = frames.get_depth_frame();
                auto ir_frame = frames.get_infrared_frame();

                if (!depth_frame || !ir_frame) {
                    std::cerr << "Failed to get valid depth or IR frame" << std::endl;
                    continue;
                }

                // Get frame dimensions
                const int width = depth_frame.get_width();
                const int height = depth_frame.get_height();

                // Get frame data
                const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
                const uint8_t* ir_data = reinterpret_cast<const uint8_t*>(ir_frame.get_data());

                if (!depth_data || !ir_data) {
                    std::cerr << "Failed to get frame data" << std::endl;
                    continue;
                }

                // Create OpenCV matrices
                cv::Mat depth_image(height, width, CV_16UC1, (void*)depth_data);
                cv::Mat ir_image(height, width, CV_8UC1, (void*)ir_data);

                // Process images
                process_images(depth_image, ir_image);

                // Update depth range periodically
                frames_since_update++;
                if(frames_since_update >= UPDATE_INTERVAL) {
                    find_max_depth(depth_image, max_depth);
                    frames_since_update = 0;
                }

                // Create color depth image
                cv::Mat depth_color(height, width, CV_8UC3);
                for(int y = 0; y < height; y++) {
                    for(int x = 0; x < width; x++) {
                        uint16_t depth = depth_image.at<uint16_t>(y, x);
                        depth_color.at<cv::Vec3b>(y, x) = depth_to_color(depth, max_depth);
                    }
                }

                // Create side-by-side display
                cv::Mat display;
                cv::Mat ir_color;
                cv::cvtColor(ir_image, ir_color, cv::COLOR_GRAY2BGR);
                cv::hconcat(std::vector<cv::Mat>{depth_color, ir_color}, display);

                // Calculate FPS
                frame_count++;
                auto current_time = std::chrono::steady_clock::now();
                auto time_elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                if (time_elapsed >= 1) {
                    fps = frame_count / (float)time_elapsed;
                    frame_count = 0;
                    start_time = current_time;
                }

                // Create depth key
                cv::Mat depth_key = create_depth_key(max_depth);

                // Add labels and information
                cv::putText(display, "Depth", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,255), 2);
                cv::putText(display, "Infrared", cv::Point(width + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,255), 2);
                
                char info_str[128];
                snprintf(info_str, sizeof(info_str), "FPS: %.1f | Max Depth: %.2fm", fps, max_depth/1000.0f);
                cv::putText(display, info_str, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);

                // Add depth key to bottom of display
                cv::Mat final_display;
                cv::copyMakeBorder(display, final_display, 0, depth_key.rows + 20, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
                cv::putText(final_display, "Depth Scale:", cv::Point(10, display.rows + 15), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
                depth_key.copyTo(final_display(cv::Rect(10, display.rows + 20, depth_key.cols, depth_key.rows)));

                // Display the combined image
                cv::imshow(WINDOW_NAME, final_display);

            } catch (const rs2::error& e) {
                std::cerr << "RealSense error: " << e.what() << std::endl;
                continue;
            } catch (const cv::Exception& e) {
                std::cerr << "OpenCV error: " << e.what() << std::endl;
                continue;
            }

            // Check for 'q' key to quit
            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q') {
                break;
            }
        }

        cv::destroyAllWindows();
        pipe.stop();

    } catch (const rs2::error& e) {
        std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.what() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 