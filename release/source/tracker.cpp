#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "../headers/color_tracker.hpp"

namespace py = pybind11;

class PendulumTracker {
private:
    std::vector<HSVRange> redRanges, blueRanges;
    double minArea = 80, maxArea = 5000, minCircularity = 0.6;

    static cv::Mat toMat(py::array_t<unsigned char>& a) {
        py::buffer_info buf = a.request();
        return cv::Mat((int)buf.shape[0], (int)buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
    }

public:
    PendulumTracker() {
        // Starting values, tune them with debug_masks()
        redRanges  = makeRanges(165, 120, 70, 10, 255, 255);
        blueRanges = makeRanges(95, 120, 50, 130, 255, 255);
    }

    // Returns (x1, y1, x2, y2). Mass 1 = red, mass 2 = blue. (0,0) means "not found".
    py::tuple track_frame(py::array_t<unsigned char> input_frame) {
        cv::Mat frame = toMat(input_frame);          // BGR, no copy
        double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        if (frame.empty()) return py::make_tuple(x1, y1, x2, y2);

        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        cv::Point2d c;
        if (findBlob(colorMask(hsv, redRanges), minArea, maxArea, minCircularity, c)) { x1 = c.x; y1 = c.y; }
        if (findBlob(colorMask(hsv, blueRanges), minArea, maxArea, minCircularity, c)) { x2 = c.x; y2 = c.y; }

        return py::make_tuple(x1, y1, x2, y2);
    }

    // Tuning helpers
    void set_red_range(int hLo, int sLo, int vLo, int hHi, int sHi, int vHi) {
        redRanges = makeRanges(hLo, sLo, vLo, hHi, sHi, vHi);
    }
    void set_blue_range(int hLo, int sLo, int vLo, int hHi, int sHi, int vHi) {
        blueRanges = makeRanges(hLo, sLo, vLo, hHi, sHi, vHi);
    }
    void set_blob_filters(double minA, double maxA, double minCirc) {
        minArea = minA; maxArea = maxA; minCircularity = minCirc;
    }

    // Returns (red_mask, blue_mask) as 2D uint8 arrays to see what the tracker sees
    py::tuple debug_masks(py::array_t<unsigned char> input_frame) {
        cv::Mat frame = toMat(input_frame), hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::Mat r = colorMask(hsv, redRanges), b = colorMask(hsv, blueRanges);
        py::array_t<unsigned char> ra({r.rows, r.cols}, r.data);   // copies
        py::array_t<unsigned char> ba({b.rows, b.cols}, b.data);
        return py::make_tuple(ra, ba);
    }
};

PYBIND11_MODULE(pendulum_tracker, m) {
    py::class_<PendulumTracker>(m, "PendulumTracker")
        .def(py::init<>())
        .def("track_frame", &PendulumTracker::track_frame)
        .def("set_red_range", &PendulumTracker::set_red_range)
        .def("set_blue_range", &PendulumTracker::set_blue_range)
        .def("set_blob_filters", &PendulumTracker::set_blob_filters)
        .def("debug_masks", &PendulumTracker::debug_masks);
}