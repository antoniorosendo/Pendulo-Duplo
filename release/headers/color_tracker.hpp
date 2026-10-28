#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct HSVRange { cv::Scalar lo, hi; };

// OpenCV 8-bit hue is 0..179. Red wraps around 0, so if hLo > hHi
// the range is split in two (e.g. 165..179 and 0..10).
inline std::vector<HSVRange> makeRanges(int hLo, int sLo, int vLo,
                                        int hHi, int sHi, int vHi) {
    if (hLo <= hHi)
        return {{cv::Scalar(hLo, sLo, vLo), cv::Scalar(hHi, sHi, vHi)}};
    return {{cv::Scalar(hLo, sLo, vLo), cv::Scalar(179, sHi, vHi)},
            {cv::Scalar(0,   sLo, vLo), cv::Scalar(hHi, sHi, vHi)}};
}

// Binary mask of the pixels inside the HSV range(s), cleaned with morphology
inline cv::Mat colorMask(const cv::Mat& hsv, const std::vector<HSVRange>& ranges) {
    cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8UC1), tmp;
    for (const auto& r : ranges) {
        cv::inRange(hsv, r.lo, r.hi, tmp);
        cv::bitwise_or(mask, tmp, mask);
    }
    static const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  k);  // removes small noise
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);  // fills holes (highlights)
    return mask;
}

// Finds the best round blob in the mask. Returns false if none passes the filters.
inline bool findBlob(const cv::Mat& mask, double minArea, double maxArea,
                     double minCircularity, cv::Point2d& center) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double bestScore = 0;
    bool found = false;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < minArea || area > maxArea) continue;

        double perimeter = cv::arcLength(c, true);
        if (perimeter <= 0) continue;
        double circularity = 4.0 * CV_PI * area / (perimeter * perimeter); // 1.0 = perfect circle
        if (circularity < minCircularity) continue;

        double score = area * circularity;
        if (score > bestScore) {
            cv::Moments m = cv::moments(c);
            if (m.m00 == 0) continue;
            center = cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
            bestScore = score;
            found = true;
        }
    }
    return found;
}