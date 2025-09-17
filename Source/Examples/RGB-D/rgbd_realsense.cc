#include <iostream>
#include <chrono>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <librealsense2/rs.hpp>     // RealSense SDK
#include "System.h"

using namespace std;

static inline double now_sec()
{
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        cerr << "\nUsage: ./rgbd_realsense path_to_settings\n";
        return 1;
    }

    string settingsFile = string(DEFAULT_RGBD_SETTINGS_DIR) + "/" + string(argv[1]);

    ORB_SLAM2::System SLAM(settingsFile, ORB_SLAM2::System::RGBD, true);

    int main_error = 0;

    std::thread runthread([&]() {
        try {
            rs2::config cfg;
            const int W = 640, H = 480, FPS = 15;
            cfg.enable_stream(RS2_STREAM_COLOR, W, H, RS2_FORMAT_BGR8, FPS);
            cfg.enable_stream(RS2_STREAM_DEPTH, W, H, RS2_FORMAT_Z16, FPS);

            rs2::pipeline pipe;
            rs2::pipeline_profile profile = pipe.start(cfg);

            auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
            const float depth_scale = depth_sensor.get_depth_scale();              // meters per unit
            const float recommended_factor = 1.0f / depth_scale;                  // e.g. scale=0.001 -> 1000
            cout << "[RealSense] depth_scale = " << depth_scale
                 << " (m/unit). Set YAML: DepthMapFactor = " << recommended_factor << endl;

            rs2::align align_to_color(RS2_STREAM_COLOR);

            cout << "RealSense started. WxH=" << W << "x" << H << " FPS=" << FPS << endl;

            while (true) {
                if (SLAM.isFinished()) break;

                rs2::frameset fs = pipe.wait_for_frames();
                fs = align_to_color.process(fs);

                rs2::video_frame color = fs.get_color_frame();
                rs2::depth_frame depth = fs.get_depth_frame();
                if (!color || !depth) {
                    cerr << "Frames not ready\n";
                    continue;
                }

                cv::Mat imRGB(cv::Size(color.get_width(), color.get_height()), CV_8UC3,
                              (void*)color.get_data(), cv::Mat::AUTO_STEP);
                cv::Mat imD(cv::Size(depth.get_width(), depth.get_height()), CV_16U,
                            (void*)depth.get_data(), cv::Mat::AUTO_STEP);

                cv::Mat imRGB_clone = imRGB.clone();
                cv::Mat imD_clone   = imD.clone();

                double tsec = color.get_timestamp() * 1e-3;

                SLAM.TrackRGBD(imRGB_clone, imD_clone, tsec);
            }
        } catch (const rs2::error &e) {
            cerr << "RealSense error: " << e.get_failed_function()
                 << "(" << e.get_failed_args() << "): " << e.what() << endl;
            main_error = 1;
        } catch (const std::exception &e) {
            cerr << "Exception: " << e.what() << endl;
            main_error = 1;
        }

        SLAM.StopViewer();
    });

    SLAM.StartViewer();

    runthread.join();

    SLAM.Shutdown();
    if (main_error == 0) {
        SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    }
    return main_error;
}
