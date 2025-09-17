#include <iostream>
#include <chrono>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "System.h"

using namespace std;

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        cerr << "\nUsage: ./mono_network path_to_settings [ipcam_url]\n";
        return 1;
    }

    // Settings (same as other demos)
    string settingsFile = string(DEFAULT_MONO_SETTINGS_DIR) + "/" + string(argv[1]);

    // Create SLAM system. (enable viewer)
    ORB_SLAM2::System SLAM(settingsFile, ORB_SLAM2::System::MONOCULAR, true);

    // Optional URL
    string url = (argc == 3) ? string(argv[2]) : "http://admin:admin@10.171.206.186:8081/video";

    // Error flag to propagate from worker thread
    int main_error = 0;

    // ---- Tracking thread: open camera + feed frames ----
    std::thread runthread([&]() {
        cv::VideoCapture cap(url, cv::CAP_FFMPEG);
        if (!cap.isOpened()) {
            cerr << "Cannot open camera stream: " << url << endl;
            main_error = 1;
            SLAM.StopViewer(); // in case viewer already started
            return;
        }

        cout << "Stream opened. WxH=" << cap.get(cv::CAP_PROP_FRAME_WIDTH)
             << "x" << cap.get(cv::CAP_PROP_FRAME_HEIGHT)
             << " fourcc=" << (int)cap.get(cv::CAP_PROP_FOURCC) << endl;

        // Try to pace by FPS if available
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0 || std::isnan(fps)) fps = 30.0;
        const double T = 1.0 / fps;

        cv::Mat frame, gray;
        while (true) {
            if (SLAM.isFinished()) break;

            if (!cap.read(frame) || frame.empty()) {
                cerr << "Failed to grab frame\n";
                main_error = 1;
                break;
            }

            // Convert to grayscale (monocular)
            switch (frame.type()) {
                case CV_8UC1: gray = frame; break;
                case CV_8UC3: cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); break;
                case CV_8UC4: cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY); break;
                default: {
                    cv::Mat tmp8u;
                    frame.convertTo(tmp8u, CV_8U);
                    if (tmp8u.channels() == 3)      cv::cvtColor(tmp8u, gray, cv::COLOR_BGR2GRAY);
                    else if (tmp8u.channels() == 4) cv::cvtColor(tmp8u, gray, cv::COLOR_BGRA2GRAY);
                    else                             gray = tmp8u;
                    break;
                }
            }

            auto t1 = std::chrono::steady_clock::now();
            double tframe = (double)cv::getTickCount() / cv::getTickFrequency();

            // Feed frame to SLAM
            SLAM.TrackMonocular(gray, tframe);

            auto t2 = std::chrono::steady_clock::now();
            double ttrack = std::chrono::duration<double>(t2 - t1).count();
            if (ttrack < T) std::this_thread::sleep_for(std::chrono::duration<double>(T - ttrack));
        }

        // End of stream / error: stop viewer so main thread can return from StartViewer()
        SLAM.StopViewer();
    });

    // ---- Viewer on main thread (blocking), same as mono_kitti ----
    SLAM.StartViewer();

    // Wait tracking thread
    runthread.join();

    if (main_error != 0) {
        // Clean shutdown even on error
        SLAM.Shutdown();
        return main_error;
    }

    // Normal shutdown
    SLAM.Shutdown();
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    return 0;
}
