#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <tuple>
#include <time.h>
#include <iostream>
#include <string>
#include "headers/guiFunctions.h"
#include <fstream>

#define PI 3.14159265

using namespace std;
using namespace cv;

bool is_graph_activated = false;
int graphPoints = 100;
int* ptrGraphPoints = &graphPoints;
int expectedFrameNumber = 0;

void toggleView(int status, void* data){
    if(status == 0) is_graph_activated = false;
    else is_graph_activated = true;
}

void checkTrackbar(int trackPos, void* data) {
    if(trackPos < 1) *ptrGraphPoints = 1;
}

/// Global Variables
Mat frame;
Mat templ1; // Template para a massa 1
Mat templ2; // Template para a massa 2
double myMatrix[3][3];

String image_window = "Window";
Mat result_A;
Mat result_B;

// Filas para o gráfico de desenho
std::queue<Point2d> pointsVector1;
std::queue<Point2d> pointsVector2;
Mat plot_image;

int match_method = 5; // CCOEFF_NORMED (ZNCC)

/// Filas atualizadas: tuple agora carrega (Mat, int, double, x1, y1, x2, y2)
std::queue<std::tuple<Mat, int, double>> frameQueue_A;
std::queue<std::tuple<Mat, int, double>> frameQueue_B;

std::queue<tuple<Mat, int, double, double, double, double, double>> resultQueue_A;
std::queue<tuple<Mat, int, double, double, double, double, double>> resultQueue_B;

struct request{
    std::mutex mx;
    std::condition_variable cv;
};
request requestA;
request requestB;

/// Function Headers
tuple<double, double, double, double> MatchingMethod( int, void*, const string&, Mat croppedFrame);
void frameComputation(const string& whichThread);
[[noreturn]] void writeFile(const bool perspectiveCorrection);

/** @function main */
int main(int argc, char *argv[]) {

    string calibrationFile = "calibration.txt";
    string templateFile1 = "../images/template1.jpg";
    string templateFile2 = "../images/template2.jpg";

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

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double> TimeCast;

    /// Load templates
    if(fileExist(templateFile1) && fileExist(templateFile2)) {
        templ1 = imread(templateFile1, 1);
        templ2 = imread(templateFile2, 1);
    } else {
        cerr << "ERROR. Can't find template files \n Closing..." << endl;
        exit(-1);
    }

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

    /// starting the two threads that handle the frame computation here
    std::thread thread1 (frameComputation, "threadA");
    std::thread thread2 (frameComputation, "threadB");
    std::thread thread3 (writeFile, perspectiveCorrection);
    thread1.detach();
    thread2.detach();
    thread3.detach();

    auto startTime = Time::now();
    auto newTime = Time::now();
    TimeCast elapsed = newTime - startTime;
    bool start = false;
    std::tuple<Mat, int, double> frameInfo;
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

        frameInfo = std::make_tuple(frame.clone(), frame_number, elapsed.count());

        if (frame_number % 2 == 0) frameQueue_A.push(frameInfo);
        else frameQueue_B.push(frameInfo);

        frame_number++;
        // Para não ler o vídeo mais rápido que as threads processam (simulando a câmera)
        //std::this_thread::sleep_for(std::chrono::milliseconds(30)); 
    }

    // Mantém o programa rodando para as threads terminarem de gravar o arquivo
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return 0;
}


tuple<double, double, double, double> MatchingMethod( int, void*, const string& whichThread, Mat croppedFrame) {
    Mat *result_X;
    if(whichThread == "threadA") result_X = &result_A;
    else result_X = &result_B;

    /// ---- MASSA 1 ----
    int result_cols1 = croppedFrame.cols - templ1.cols + 1;
    int result_rows1 = croppedFrame.rows - templ1.rows + 1;
    result_X->create(result_rows1, result_cols1, CV_32FC1);
    matchTemplate( croppedFrame, templ1, *result_X, match_method );
    normalize( *result_X, *result_X, 0, 1, NORM_MINMAX, -1, Mat() );

    double minVal1, maxVal1; Point minLoc1, maxLoc1, matchLoc1;
    minMaxLoc( *result_X, &minVal1, &maxVal1, &minLoc1, &maxLoc1, Mat() );
    matchLoc1 = (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED) ? minLoc1 : maxLoc1;

    /// ---- MASSA 2 ----
    int result_cols2 = croppedFrame.cols - templ2.cols + 1;
    int result_rows2 = croppedFrame.rows - templ2.rows + 1;
    result_X->create(result_rows2, result_cols2, CV_32FC1);
    matchTemplate( croppedFrame, templ2, *result_X, match_method );
    normalize( *result_X, *result_X, 0, 1, NORM_MINMAX, -1, Mat() );

    double minVal2, maxVal2; Point minLoc2, maxLoc2, matchLoc2;
    minMaxLoc( *result_X, &minVal2, &maxVal2, &minLoc2, &maxLoc2, Mat() );
    matchLoc2 = (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED) ? minLoc2 : maxLoc2;

    /// Desenhando na imagem de saída
    rectangle( croppedFrame, matchLoc1, Point( matchLoc1.x + templ1.cols , matchLoc1.y + templ1.rows ), Scalar(0,0,255), 2, 8, 0 ); // Vermelho
    rectangle( croppedFrame, matchLoc2, Point( matchLoc2.x + templ2.cols , matchLoc2.y + templ2.rows ), Scalar(255,0,0), 2, 8, 0 ); // Azul
    
    circle(croppedFrame, Point(matchLoc1.x + templ1.cols/2, matchLoc1.y + templ1.rows/2), 0, Scalar(0,255,0), 5);
    circle(croppedFrame, Point(matchLoc2.x + templ2.cols/2, matchLoc2.y + templ2.rows/2), 0, Scalar(0,255,0), 5);

    return std::make_tuple(
        matchLoc1.x + templ1.cols/2.0, matchLoc1.y + templ1.rows/2.0,
        matchLoc2.x + templ2.cols/2.0, matchLoc2.y + templ2.rows/2.0
    );
}

void frameComputation(const string& whichThread){
    Mat myFrame;
    int myFrameNumber;
    double ourElapsed;
    std::queue<std::tuple<Mat, int, double>> *frameQueue_X;
    std::queue<tuple<Mat, int, double, double, double, double, double>> *resultQueue_X;
    
    double pos_X1, pos_Y1, pos_X2, pos_Y2;
    request *requestX;

    if(whichThread == "threadA"){
        frameQueue_X = &frameQueue_A;
        resultQueue_X = &resultQueue_A;
        requestX = &requestA;
    } else {
        frameQueue_X = &frameQueue_B;
        resultQueue_X = &resultQueue_B;
        requestX = &requestB;
    }

    while(true){
        if(!frameQueue_X->empty()){
            tie(myFrame, myFrameNumber, ourElapsed) = frameQueue_X->front();
            frameQueue_X->pop();

            if (myFrame.empty()) break;

            tuple<double, double, double, double> myResult;
            myResult = MatchingMethod(0, 0, whichThread, myFrame);
            tie(pos_X1, pos_Y1, pos_X2, pos_Y2) = myResult;

            resultQueue_X->push(std::make_tuple(myFrame.clone(), myFrameNumber, ourElapsed, pos_X1, pos_Y1, pos_X2, pos_Y2));

            std::lock_guard<std::mutex> lock(requestX->mx);
            requestX->cv.notify_one();
        }
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
    oss << "../PendulumCsv/" << year << "_" << (month.length()==1?"0":"") << month << "_" << (day.length()==1?"0":"") << day << "_" << (hour.length()==1?"0":"") << hour << "_" << (min.length()==1?"0":"") << min;
    
    std::string file_name = oss.str() + "_N.csv";

    ofstream txt_file(file_name);
    if (txt_file.is_open()) cout << "Opened file "<< file_name <<"\n";
    
    // NOVO CABEÇALHO PARA 2 PONTOS
    txt_file << "time;x1;y1;x2;y2\n";

    int frameNumber_X = -1;
    double elapsed_X = -1.0;
    double pos_X1, pos_Y1, pos_X2, pos_Y2;
    std::queue<tuple<Mat, int, double, double, double, double, double>> *resultQueue_X;
    Mat extracted_Mat_X;

    namedWindow(image_window, WINDOW_GUI_EXPANDED);
    createButton("Show graph", toggleView, NULL, QT_CHECKBOX, 0);
    createTrackbar("Number of graph points", image_window, ptrGraphPoints, 1000, checkTrackbar, NULL);

    while(true){
        if (expectedFrameNumber % 2 == 0){
            std::unique_lock<std::mutex> lock(requestA.mx);
            requestA.cv.wait(lock, []{return !resultQueue_A.empty();});
            resultQueue_X = &resultQueue_A;
        } else {
            std::unique_lock<std::mutex> lock(requestB.mx);
            requestB.cv.wait(lock, []{return !resultQueue_B.empty();});
            resultQueue_X = &resultQueue_B;
        }

        tie(extracted_Mat_X, frameNumber_X, elapsed_X, pos_X1, pos_Y1, pos_X2, pos_Y2) = resultQueue_X->front();
        resultQueue_X->pop();

        // Salva os 4 dados no CSV
        txt_file << fixed << elapsed_X << ";" << (int)(pos_X1) << ";" << (int)(pos_Y1) << ";" << (int)(pos_X2) << ";" << (int)(pos_Y2) << "\n";
        txt_file.flush();
        
        expectedFrameNumber++;

        /// ---- PLOTANDO O GRÁFICO NA TELA ----
        pointsVector1.push(Point2d(pos_X1, pos_Y1));
        pointsVector2.push(Point2d(pos_X2, pos_Y2));

        cv::line(plot_image, Point2d(pos_X1, pos_Y1), Point2d(pos_X1, pos_Y1), cv::Scalar(0,0,255), 2); // Linha Vermelha
        cv::line(plot_image, Point2d(pos_X2, pos_Y2), Point2d(pos_X2, pos_Y2), cv::Scalar(255,0,0), 2); // Linha Azul

        if (pointsVector1.size() >= graphPoints){
            while(pointsVector1.size() > graphPoints){
                Point2d lastPoint1 = pointsVector1.front();
                Point2d lastPoint2 = pointsVector2.front();
                cv::line(plot_image, lastPoint1, lastPoint1, cv::Scalar(255,255,255), 2);
                cv::line(plot_image, lastPoint2, lastPoint2, cv::Scalar(255,255,255), 2);
                pointsVector1.pop();
                pointsVector2.pop();
            }
        }

        if(is_graph_activated) imshow(image_window, plot_image);
        else imshow(image_window, extracted_Mat_X);

        waitKey(1);
    }
}