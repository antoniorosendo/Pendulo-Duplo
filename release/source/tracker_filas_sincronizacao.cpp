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

namespace py = pybind11;

class PendulumTracker {
private:
    // Um frame que aguarda processamento pelos workers.
    // elapsed é o tempo relativo à captura, medido no Python no momento
    // em que o frame foi capturado (não o tempo em que o worker o processa).
    struct FrameTask {
        cv::Mat frame;   // imagem própria do motor; deverá ser uma cópia do frame
        int frame_id;    // identificador sequencial atribuído na captura
        double elapsed;  // tempo decorrido desde o início da captura, em segundos
    };

    // Resultado do rastreamento de um frame.
    // frame_id e elapsed permitem preservar a ordem e o tempo físico original,
    // mesmo quando os workers terminarem em uma ordem diferente.
    struct TrackingResult {
        int frame_id;
        double elapsed;
        double x1, y1;   // coordenadas da primeira massa (vermelha)
        double x2, y2;   // coordenadas da segunda massa (azul)
    };

    // Quantidade de workers prevista para a etapa de processamento.
    static constexpr std::size_t NUM_WORKERS = 6;

    // Fila 1: frames aguardando processamento.
    std::queue<FrameTask> frame_queue;
    std::mutex frame_mutex;
    std::condition_variable frame_available;

    // Fila 2: resultados aguardando gravação no CSV.
    // Será consumida exclusivamente pela thread de escrita.
    std::queue<TrackingResult> writer_queue;
    std::mutex writer_mutex;
    std::condition_variable writer_available;

    // Fila 3: cópia dos resultados que a GUI poderá consultar.
    // É separada da fila de escrita para que GUI e escritor não consumam
    // acidentalmente os mesmos itens da mesma fila.
    std::queue<TrackingResult> gui_queue;
    std::mutex gui_mutex;

    // Estado compartilhado do ciclo de vida do motor.
    // atomic permite que os workers consultem stopping sem data race.
    std::atomic<bool> stopping{false};
    std::mutex lifecycle_mutex;
    bool started = false;

    // Os objetos thread ficam guardados na classe para que, na etapa de
    // inicialização/encerramento, possamos gerenciá-los com join().
    std::array<std::thread, NUM_WORKERS> workers;
    std::thread writer_thread;

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