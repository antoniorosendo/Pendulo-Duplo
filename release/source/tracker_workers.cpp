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

    // Protege os parâmetros caso a GUI ajuste os filtros enquanto um worker
    // estiver processando um frame.
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

    // Executa a detecção de um único frame. Cada worker chama esta função.
    TrackingResult process_frame(const FrameTask& task) {
        TrackingResult result{task.frame_id, task.elapsed, 0, 0, 0, 0};
        if (task.frame.empty()) {
            return result;
        }

        // Copia a configuração sob mutex e libera o mutex antes do processamento.
        std::vector<HSVRange> red_ranges;
        std::vector<HSVRange> blue_ranges;
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
        if (findBlob(colorMask(hsv, red_ranges),
                     min_area, max_area, min_circularity, center)) {
            result.x1 = center.x;
            result.y1 = center.y;
        }
        if (findBlob(colorMask(hsv, blue_ranges),
                     min_area, max_area, min_circularity, center)) {
            result.x2 = center.x;
            result.y2 = center.y;
        }
        return result;
    }

    // Worker: aguarda um frame, processa-o e publica cópias do resultado
    // nas filas independentes destinadas ao CSV e à GUI.
    void worker_loop() {
        while (true) {
            FrameTask task;
            {
                std::unique_lock<std::mutex> lock(frame_mutex);
                frame_available.wait(lock, [this] {
                    return stopping.load() || !frame_queue.empty();
                });
                if (stopping.load() && frame_queue.empty()) {
                    return;
                }
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

    // Inicia os seis workers uma única vez. A submissão de frames será
    // conectada à API Python na etapa assíncrona seguinte.
    void start_workers() {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if (started) return;
        stopping.store(false);
        started = true;
        try {
            for (auto& worker : workers) {
                worker = std::thread(&PendulumTracker::worker_loop, this);
            }
        } catch (...) {
            stopping.store(true);
            frame_available.notify_all();
            for (auto& worker : workers) {
                if (worker.joinable()) worker.join();
            }
            started = false;
            throw;
        }
    }

    // Enfileira uma cópia própria do frame. Ainda não é exposto ao Python.
    void enqueue_frame(py::array_t<unsigned char> input_frame,
                       int frame_id, double elapsed) {
        cv::Mat view = toMat(input_frame);
        cv::Mat owned_frame = view.clone();
        start_workers();
        {
            std::lock_guard<std::mutex> lock(lifecycle_mutex);
            if (stopping.load()) {
                throw std::runtime_error("O motor de rastreamento está sendo encerrado.");
            }
        }
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame_queue.push(FrameTask{std::move(owned_frame), frame_id, elapsed});
        }
        frame_available.notify_one();
    }

    void stop_workers() {
        {
            std::lock_guard<std::mutex> lock(lifecycle_mutex);
            if (!started) return;
            stopping.store(true);
        }
        frame_available.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        started = false;
    }

public:
    PendulumTracker() {
        // Starting values, tune them with debug_masks()
        redRanges  = makeRanges(165, 120, 70, 10, 255, 255);
        blueRanges = makeRanges(95, 120, 50, 130, 255, 255);
    }

    ~PendulumTracker() {
        stop_workers();
    }

    // Returns (x1, y1, x2, y2). Mass 1 = red, mass 2 = blue. (0,0) means "not found".
    py::tuple track_frame(py::array_t<unsigned char> input_frame) {
        cv::Mat frame = toMat(input_frame);  // API síncrona antiga
        TrackingResult result = process_frame(FrameTask{frame, -1, 0.0});
        return py::make_tuple(result.x1, result.y1, result.x2, result.y2);
    }

    // Tuning helpers
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

    // Returns (red_mask, blue_mask) as 2D uint8 arrays to see what the tracker sees
    py::tuple debug_masks(py::array_t<unsigned char> input_frame) {
        cv::Mat frame = toMat(input_frame), hsv;
        std::vector<HSVRange> red_ranges, blue_ranges;
        {
            std::lock_guard<std::mutex> lock(config_mutex);
            red_ranges = redRanges;
            blue_ranges = blueRanges;
        }
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::Mat r = colorMask(hsv, red_ranges), b = colorMask(hsv, blue_ranges);
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