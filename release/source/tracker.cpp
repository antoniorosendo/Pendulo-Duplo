#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "../headers/color_tracker.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <array>
#include <chrono>
#include <atomic>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <unordered_map>
#include <unistd.h>
#include <sys/syscall.h>

namespace py = pybind11;

class PendulumTracker {
private:
    struct FrameTask {
        cv::Mat frame;
        int frame_id;
        double elapsed;
    };

    struct TrackingResult {
        int frame_id;
        double elapsed;
        double x1, y1;
        double x2, y2;
    };

    static constexpr std::size_t NUM_WORKERS = 6;

    std::queue<FrameTask> frame_queue;
    std::mutex frame_mutex;
    std::condition_variable frame_available;

    std::queue<TrackingResult> writer_queue;
    std::mutex writer_mutex;
    std::condition_variable writer_available;

    std::queue<TrackingResult> gui_queue;
    std::mutex gui_mutex;

    std::atomic<bool> stopping{false};
    std::mutex lifecycle_mutex;
    bool started = false;

    std::array<std::thread, NUM_WORKERS> workers;
    std::thread writer_thread;
    
    // Armazenamento dos TIDs para o Python
    std::vector<int> worker_tids;
    int writer_tid = 0;
    std::mutex tid_mutex;
    
    std::string csv_filename;

    std::mutex config_mutex;
    std::vector<HSVRange> redRanges, blueRanges;
    double minArea = 80, maxArea = 5000, minCircularity = 0.6;

    static cv::Mat toMat(py::array_t<unsigned char>& a) {
        py::buffer_info buf = a.request();
        if (buf.ndim != 3 || buf.shape[2] != 3) {
            throw std::runtime_error("O frame deve ser um array NumPy HxWx3 (BGR).");
        }
        return cv::Mat(
            static_cast<int>(buf.shape[0]),
            static_cast<int>(buf.shape[1]),
            CV_8UC3,
            static_cast<unsigned char*>(buf.ptr)
        );
    }

    TrackingResult process_frame(const FrameTask& task) {
        TrackingResult result{task.frame_id, task.elapsed, 0, 0, 0, 0};
        if (task.frame.empty()) return result;

        std::vector<HSVRange> red_ranges, blue_ranges;
        double min_area, max_area, min_circularity;
        {
            std::lock_guard<std::mutex> lock(config_mutex);
            red_ranges = redRanges;
            blue_ranges = blueRanges;
            min_area = minArea;
            max_area = maxArea;
            min_circularity = minCircularity;
        }

        cv::Mat hsv;
        cv::cvtColor(task.frame, hsv, cv::COLOR_BGR2HSV);

        cv::Point2d center;
        if (findBlob(colorMask(hsv, red_ranges), min_area, max_area, min_circularity, center)) {
            result.x1 = center.x; result.y1 = center.y;
        }
        if (findBlob(colorMask(hsv, blue_ranges), min_area, max_area, min_circularity, center)) {
            result.x2 = center.x; result.y2 = center.y;
        }
        return result;
    }

    void worker_loop() {
        // Registra o TID nativo no Linux
        int tid = syscall(SYS_gettid);
        {
            std::lock_guard<std::mutex> lock(tid_mutex);
            worker_tids.push_back(tid);
        }

        while (true) {
            FrameTask task;
            {
                std::unique_lock<std::mutex> lock(frame_mutex);
                frame_available.wait(lock, [this] {
                    return stopping.load() || !frame_queue.empty();
                });
                if (stopping.load() && frame_queue.empty()) return;
                
                task = std::move(frame_queue.front());
                frame_queue.pop();
            }

            TrackingResult result = process_frame(task);
            
            {
                std::lock_guard<std::mutex> lock(writer_mutex);
                writer_queue.push(result);
            }
            writer_available.notify_one();
            
            {
                std::lock_guard<std::mutex> lock(gui_mutex);
                gui_queue.push(result);
            }
        }
    }

    void writer_loop() {
        writer_tid = syscall(SYS_gettid);
        
        std::ofstream file(csv_filename);
        file << "time;x1;y1;x2;y2\n";
        
        int expected_frame = 0;
        std::unordered_map<int, TrackingResult> out_of_order_buffer;

        while (true) {
            TrackingResult result;
            {
                std::unique_lock<std::mutex> lock(writer_mutex);
                writer_available.wait(lock, [this] {
                    return stopping.load() || !writer_queue.empty();
                });
                if (stopping.load() && writer_queue.empty()) break;
                
                result = writer_queue.front();
                writer_queue.pop();
            }
            
            // Reordenação cronológica para garantir a integridade da análise física
            out_of_order_buffer[result.frame_id] = result;
            
            while (out_of_order_buffer.find(expected_frame) != out_of_order_buffer.end()) {
                auto& res = out_of_order_buffer[expected_frame];
                // Formatação direta no C++
                file << res.elapsed << ";" << res.x1 << ";" << res.y1 << ";" 
                     << res.x2 << ";" << res.y2 << "\n";
                out_of_order_buffer.erase(expected_frame);
                expected_frame++;
            }
        }
        file.close();
    }

public:
    PendulumTracker() {
        redRanges  = makeRanges(165, 120, 70, 10, 255, 255);
        blueRanges = makeRanges(95, 120, 50, 130, 255, 255);
    }

    ~PendulumTracker() {
        stop_tracking();
    }
    
    void start_tracking(const std::string& filename) {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if (started) return;
        
        csv_filename = filename;
        stopping.store(false);
        started = true;
        
        worker_tids.clear();
        writer_tid = 0;

        // Limpa as filas de execuções anteriores
        std::queue<FrameTask> empty_fq; std::swap(frame_queue, empty_fq);
        std::queue<TrackingResult> empty_wq; std::swap(writer_queue, empty_wq);
        std::queue<TrackingResult> empty_gq; std::swap(gui_queue, empty_gq);

        for (auto& worker : workers) {
            worker = std::thread(&PendulumTracker::worker_loop, this);
        }
        writer_thread = std::thread(&PendulumTracker::writer_loop, this);
    }

    void submit_frame(py::array_t<unsigned char> input_frame, int frame_id, double elapsed) {
        if (!started) return;
        cv::Mat view = toMat(input_frame);
        cv::Mat owned_frame = view.clone(); // Cópia isolada para o worker
        
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame_queue.push(FrameTask{std::move(owned_frame), frame_id, elapsed});
        }
        frame_available.notify_one();
    }

    py::list get_latest_results() {
        py::list out;
        std::lock_guard<std::mutex> lock(gui_mutex);
        while (!gui_queue.empty()) {
            auto r = gui_queue.front();
            gui_queue.pop();
            out.append(py::make_tuple(r.frame_id, r.elapsed, r.x1, r.y1, r.x2, r.y2));
        }
        return out;
    }

    void stop_tracking() {
        {
            std::lock_guard<std::mutex> lock(lifecycle_mutex);
            if (!started) return;
            stopping.store(true);
        }
        frame_available.notify_all();
        writer_available.notify_all();
        
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
        if (writer_thread.joinable()) writer_thread.join();
        
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        started = false;
    }
    
    py::dict get_thread_ids() {
        py::dict tids;
        py::list w_tids;
        std::lock_guard<std::mutex> lock(tid_mutex);
        for (int tid : worker_tids) {
            w_tids.append(tid);
        }
        tids["workers"] = w_tids;
        tids["writer"] = writer_tid;
        return tids;
    }

    void set_red_range(int hLo, int sLo, int vLo, int hHi, int sHi, int vHi) {
        std::lock_guard<std::mutex> lock(config_mutex);
        redRanges = makeRanges(hLo, sLo, vLo, hHi, sHi, vHi);
    }
    void set_blue_range(int hLo, int sLo, int vLo, int hHi, int sHi, int vHi) {
        std::lock_guard<std::mutex> lock(config_mutex);
        blueRanges = makeRanges(hLo, sLo, vLo, hHi, sHi, vHi);
    }
    void set_blob_filters(double minA, double maxA, double minCirc) {
        std::lock_guard<std::mutex> lock(config_mutex);
        minArea = minA; maxArea = maxA; minCircularity = minCirc;
    }
    py::tuple debug_masks(py::array_t<unsigned char> input_frame) {
        cv::Mat frame = toMat(input_frame), hsv;
        std::vector<HSVRange> red_ranges, blue_ranges;
        {
            std::lock_guard<std::mutex> lock(config_mutex);
            red_ranges = redRanges; blue_ranges = blueRanges;
        }
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::Mat r = colorMask(hsv, red_ranges), b = colorMask(hsv, blue_ranges);
        py::array_t<unsigned char> ra({r.rows, r.cols}, r.data);
        py::array_t<unsigned char> ba({b.rows, b.cols}, b.data);
        return py::make_tuple(ra, ba);
    }
};

PYBIND11_MODULE(pendulum_tracker, m) {
    py::class_<PendulumTracker>(m, "PendulumTracker")
        .def(py::init<>())
        .def("start_tracking", &PendulumTracker::start_tracking)
        .def("submit_frame", &PendulumTracker::submit_frame)
        .def("get_latest_results", &PendulumTracker::get_latest_results)
        .def("stop_tracking", &PendulumTracker::stop_tracking)
        .def("get_thread_ids", &PendulumTracker::get_thread_ids)
        .def("set_red_range", &PendulumTracker::set_red_range)
        .def("set_blue_range", &PendulumTracker::set_blue_range)
        .def("set_blob_filters", &PendulumTracker::set_blob_filters)
        .def("debug_masks", &PendulumTracker::debug_masks);
}