#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <array>
#include <tuple>
#include <time.h>
#include <iostream>
#include <string>
#include "headers/guiFunctions.h"
#include "headers/color_tracker.hpp"
#include <fstream>

#define PI 3.14159265

using namespace std;
using namespace cv;

// ---------------------------------------------------------------------------
// Arquitetura de 8 threads:
//   1 thread de captura (main)  +  NUM_WORKERS de processamento  +  1 de escrita
// ---------------------------------------------------------------------------
constexpr int NUM_WORKERS = 6;

bool is_graph_activated = false;
int graphPoints = 100;
int* ptrGraphPoints = &graphPoints;
int expectedFrameNumber = 0;
bool is_tracking = false;

void startTracking(int status, void* data){
    is_tracking = true;
    cout << "--- RASTREAMENTO INICIADO ---" << endl;
}

// Função do botão STOP
void stopProgram(int status, void* data){
    cout << "--- ENCERRANDO PROGRAMA ---" << endl;
    exit(0); // O exit(0) mata todas as threads na hora. Adeus Ctrl+C!
}

void toggleView(int status, void* data){
    if(status == 0) is_graph_activated = false;
    else is_graph_activated = true;
}

void checkTrackbar(int trackPos, void* data) {
    if(trackPos < 1) *ptrGraphPoints = 1;
}

/// Global Variables
Mat frame;
double myMatrix[3][3];

String image_window = "Window";

// Filas para o gráfico de desenho
std::queue<Point2d> pointsVector1;
std::queue<Point2d> pointsVector2;
Mat plot_image;

// (Mat, número do frame, tempo)
using FrameInfo  = std::tuple<Mat, int, double>;
// (Mat, número do frame, tempo, x1, y1, x2, y2)
using ResultInfo = std::tuple<Mat, int, double, double, double, double, double>;

// Cada worker tem suas próprias filas e seu próprio mutex.
// O mutex protege as DUAS filas (frames e results).
struct Worker {
    std::queue<FrameInfo>  frames;
    std::queue<ResultInfo> results;
    std::mutex mx;
    std::condition_variable cvFrames;   // avisa o worker: chegou frame novo
    std::condition_variable cvResults;  // avisa o escritor: resultado pronto
};
std::array<Worker, NUM_WORKERS> workers;

/// Function Headers
tuple<double, double, double, double> MatchingMethod( int, void*, const string&, Mat croppedFrame);
void frameComputation(int workerId);
[[noreturn]] void writeFile(const bool perspectiveCorrection);

/** @function main */
int main(int argc, char *argv[]) {

    string calibrationFile = "calibration.txt";

    if (argc > 2) {
        cerr << "Too many arguments" << endl;
        exit(1);
    }
    if (argc == 2) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "-help") == 0) { exit(0); }
        if (strcmp(argv[1], "-g") == 0 || strcmp(argv[1], "-graph") == 0) { drawGraph(); exit(0); }
        if (strcmp(argv[1], "-c") == 0 || strcmp(argv[1], "-calibrate") == 0) { calibrateCamera(); exit(0); }
        cerr << argv[1] << " is an unknown option" << endl;
        exit(1);
    }

    // Cada worker já é uma thread nossa: impede o OpenCV de criar threads
    // internas extras, para o programa ter exatamente 8 threads de trabalho.
    cv::setNumThreads(1);

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double> TimeCast;

    //WEBCAM
    VideoCapture capture(0, cv::CAP_V4L2); 
    capture.set(CAP_PROP_FPS, 30);

    // PLANO B: Lendo vídeo gravado para apresentação
    // VideoCapture capture("../videos/teste.mp4"); 

    if(!capture.isOpened()) {
        cerr << "ERROR! Não foi possível abrir o arquivo de vídeo." << endl;
        return -1;
    }

    Mat originalFrame;
    capture.read(originalFrame);
    int frameWidth = originalFrame.size().width;
    int frameHeight = originalFrame.size().height;
    plot_image = Mat::zeros( frameHeight, frameWidth, CV_8UC3);

    ///ask if the user wants perspectiveCorrection (Simulando 'n' por padrão para teste)
    bool perspectiveCorrection = false; 

    /// starting the NUM_WORKERS threads that handle the frame computation + 1 writer thread
    for (int i = 0; i < NUM_WORKERS; i++) {
        std::thread(frameComputation, i).detach();
    }
    std::thread(writeFile, perspectiveCorrection).detach();

    auto startTime = Time::now();
    auto newTime = Time::now();
    TimeCast elapsed = newTime - startTime;
    bool start = false;
    int frame_number = 0;

    plot_image = cv::Scalar(255, 255, 255);

    while(true) {
        capture.read(frame);
        if (frame.empty()) {
            cerr << "Fim do vídeo ou frame vazio ignorado.\n";
            break; // Se for arquivo de vídeo, break está correto pois o vídeo acaba.
        }

        newTime = Time::now();
        if(!start){
            startTime = newTime;
            start = true;
        }
        elapsed = newTime - startTime;

        // Distribui os frames em rodízio entre os workers: 0,1,2,3,4,5,0,1,...
        Worker& w = workers[frame_number % NUM_WORKERS];
        {
            std::lock_guard<std::mutex> lock(w.mx);
            w.frames.push(std::make_tuple(frame.clone(), frame_number, elapsed.count()));
        }
        w.cvFrames.notify_one();

        frame_number++;
        // Para não ler o vídeo mais rápido que as threads processam (simulando a câmera)
        //std::this_thread::sleep_for(std::chrono::milliseconds(30)); 
    }

    // Mantém o programa rodando para as threads terminarem de gravar o arquivo
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return 0;
}

static const auto RED_RANGES  = makeRanges(165, 120, 70, 10, 255, 255);
static const auto BLUE_RANGES = makeRanges(95, 120, 50, 130, 255, 255);

tuple<double, double, double, double> MatchingMethod(int, void*, const string&, Mat croppedFrame) {
    Mat hsv;
    cvtColor(croppedFrame, hsv, COLOR_BGR2HSV);

    Point2d c1(0, 0), c2(0, 0);
    bool ok1 = findBlob(colorMask(hsv, RED_RANGES),  80, 5000, 0.6, c1);
    bool ok2 = findBlob(colorMask(hsv, BLUE_RANGES), 80, 5000, 0.6, c2);

    if (ok1) {
        rectangle(croppedFrame, Point(c1.x - 25, c1.y - 25), Point(c1.x + 25, c1.y + 25), Scalar(0, 0, 255), 2);
        circle(croppedFrame, c1, 3, Scalar(0, 255, 0), -1);
    }
    if (ok2) {
        rectangle(croppedFrame, Point(c2.x - 25, c2.y - 25), Point(c2.x + 25, c2.y + 25), Scalar(255, 0, 0), 2);
        circle(croppedFrame, c2, 3, Scalar(0, 255, 0), -1);
    }

    return std::make_tuple(c1.x, c1.y, c2.x, c2.y);
}

void frameComputation(int workerId){
    Worker& w = workers[workerId];
    const string name = "worker" + std::to_string(workerId);

    while(true){
        FrameInfo info;
        {
            // Dorme até chegar um frame (nada de espera ativa gastando 100% de CPU)
            std::unique_lock<std::mutex> lock(w.mx);
            w.cvFrames.wait(lock, [&]{ return !w.frames.empty(); });
            info = std::move(w.frames.front());
            w.frames.pop();
        }

        Mat myFrame;
        int myFrameNumber;
        double ourElapsed;
        tie(myFrame, myFrameNumber, ourElapsed) = info;

        if (myFrame.empty()) break;

        double pos_X1, pos_Y1, pos_X2, pos_Y2;
        tie(pos_X1, pos_Y1, pos_X2, pos_Y2) = MatchingMethod(0, 0, name, myFrame);

        {
            std::lock_guard<std::mutex> lock(w.mx);
            w.results.push(std::make_tuple(myFrame, myFrameNumber, ourElapsed,
                                           pos_X1, pos_Y1, pos_X2, pos_Y2));
        }
        w.cvResults.notify_one();
    }
}

[[noreturn]] void writeFile(const bool perspectiveCorrection){
    time_t theTime = time(NULL);
    struct tm *aTime = localtime(&theTime);

    string day = std::to_string(aTime->tm_mday);
    string month = std::to_string(aTime->tm_mon + 1);
    int year = aTime->tm_year + 1900;
    string hour = std::to_string(aTime->tm_hour);
    string min = std::to_string(aTime->tm_min);

    std::ostringstream oss;
    oss << "../../CSV/pendulo" << year << "_" << (month.length()==1?"0":"") << month << "_" << (day.length()==1?"0":"") << day << "_" << (hour.length()==1?"0":"") << hour << "_" << (min.length()==1?"0":"") << min;

    std::string file_name = oss.str() + "_N.csv";

    ofstream txt_file(file_name);
    if (txt_file.is_open()) cout << "Opened file "<< file_name <<"\n";
    
    // NOVO CABEÇALHO PARA 2 PONTOS
    txt_file << "time;x1;y1;x2;y2\n";

    int frameNumber_X = -1;
    double elapsed_X = -1.0;
    double pos_X1, pos_Y1, pos_X2, pos_Y2;
    Mat extracted_Mat_X;

    namedWindow(image_window, WINDOW_GUI_EXPANDED);
    // Injetando os novos botões (QT_PUSH_BUTTON cria botões clicáveis normais)
    createButton("Play", startTracking, NULL, QT_PUSH_BUTTON, 0);
    createButton("Stop", stopProgram, NULL, QT_PUSH_BUTTON, 0);
    
    createButton("Show graph", toggleView, NULL, QT_CHECKBOX, 0);
    createTrackbar("Number of graph points", image_window, ptrGraphPoints, 1000, checkTrackbar, NULL);

    while(true){
        // Lê os resultados NA ORDEM dos frames: o frame n está sempre no worker n % NUM_WORKERS
        Worker& w = workers[expectedFrameNumber % NUM_WORKERS];
        {
            std::unique_lock<std::mutex> lock(w.mx);
            w.cvResults.wait(lock, [&]{ return !w.results.empty(); });
            tie(extracted_Mat_X, frameNumber_X, elapsed_X, pos_X1, pos_Y1, pos_X2, pos_Y2) = w.results.front();
            w.results.pop();
        }
        
        expectedFrameNumber++;

        if(is_tracking) {
            // Salva os 4 dados no CSV
            txt_file << fixed << elapsed_X << ";" << (int)(pos_X1) << ";" << (int)(pos_Y1) << ";" << (int)(pos_X2) << ";" << (int)(pos_Y2) << "\n";
            txt_file.flush();
            
            /// ---- PLOTANDO O GRÁFICO NA TELA ----
            pointsVector1.push(Point2d(pos_X1, pos_Y1));
            pointsVector2.push(Point2d(pos_X2, pos_Y2));

            cv::line(plot_image, Point2d(pos_X1, pos_Y1), Point2d(pos_X1, pos_Y1), cv::Scalar(0,0,255), 2); // Linha Vermelha
            cv::line(plot_image, Point2d(pos_X2, pos_Y2), Point2d(pos_X2, pos_Y2), cv::Scalar(255,0,0), 2); // Linha Azul

            if (pointsVector1.size() >= graphPoints){
                while(pointsVector1.size() > graphPoints){
                    Point2d lastPoint1 = pointsVector1.front();
                    Point2d lastPoint2 = pointsVector2.front();
                    // Apaga os pontos antigos (usando a cor preta que ajustamos)
                    cv::line(plot_image, lastPoint1, lastPoint1, cv::Scalar(0,0,0), 2);
                    cv::line(plot_image, lastPoint2, lastPoint2, cv::Scalar(0,0,0), 2);
                    pointsVector1.pop();
                    pointsVector2.pop();
                }
            }
        } // Fim do if(is_tracking)

        if(is_graph_activated) imshow(image_window, plot_image);
        else imshow(image_window, extracted_Mat_X);

        waitKey(1);
    }
}