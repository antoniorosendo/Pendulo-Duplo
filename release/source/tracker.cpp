#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

class PendulumTracker {
private:
    cv::Mat templ1;
    cv::Mat templ2;

public:
    // Construtor carrega os templates uma única vez
    PendulumTracker(const std::string& t1_path, const std::string& t2_path) {
        templ1 = cv::imread(t1_path, cv::IMREAD_COLOR);
        templ2 = cv::imread(t2_path, cv::IMREAD_COLOR);
    }

    // Recebe a imagem do Python (Numpy Array) e faz o rastreamento
    py::tuple track_frame(py::array_t<unsigned char> input_frame) {
        // Converte o array do Python (numpy) para cv::Mat do OpenCV sem copiar memória
        py::buffer_info buf = input_frame.request();
        cv::Mat matFrame(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

        double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        int match_method = cv::TM_CCOEFF_NORMED;

        if (!templ1.empty() && !matFrame.empty()) {
            cv::Mat result1;
            cv::matchTemplate(matFrame, templ1, result1, match_method);
            double minVal1, maxVal1; cv::Point minLoc1, maxLoc1;
            cv::minMaxLoc(result1, &minVal1, &maxVal1, &minLoc1, &maxLoc1);
            x1 = maxLoc1.x + templ1.cols / 2.0;
            y1 = maxLoc1.y + templ1.rows / 2.0;
        }

        if (!templ2.empty() && !matFrame.empty()) {
            cv::Mat result2;
            cv::matchTemplate(matFrame, templ2, result2, match_method);
            double minVal2, maxVal2; cv::Point minLoc2, maxLoc2;
            cv::minMaxLoc(result2, &minVal2, &maxVal2, &minLoc2, &maxLoc2);
            x2 = maxLoc2.x + templ2.cols / 2.0;
            y2 = maxLoc2.y + templ2.rows / 2.0;
        }

        // Retorna as coordenadas direto para o Python
        return py::make_tuple(x1, y1, x2, y2);
    }
};

// Vínculo do Pybind11 criando o módulo "pendulum_tracker"
PYBIND11_MODULE(pendulum_tracker, m) {
    py::class_<PendulumTracker>(m, "PendulumTracker")
        .def(py::init<const std::string&, const std::string&>())
        .def("track_frame", &PendulumTracker::track_frame);
}