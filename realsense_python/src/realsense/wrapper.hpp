#pragma once

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <stdexcept>

class RealSenseWrapper {
private:
    rs2::pipeline pipe;
    rs2::config cfg;
    int width;
    int height;
    bool running;

public:
    RealSenseWrapper();
    ~RealSenseWrapper();
    void start();
    void stop();
    void get_frames(uint16_t* depth_data, uint8_t* ir_data);
    int get_width() const;
    int get_height() const;
}; 