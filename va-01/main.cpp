#include <iostream>
#include <climits>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void pintaImagen(std::string ventana, Mat imagen);
void pintaPunto(Mat imagen, Point punto, int radio, Scalar color);
vector<Point2f> eigen(const Mat &imagen, int ladoVentana, int distancia, float calidad, float k, int esquinas);
vector<Point2f> harris(const cv::Mat& imagen, int ladoVentana, int distancia, float calidad, float k, int esquinas);
vector<Point2i> correlacion(const Mat &imagen1, vector<Point2f> &puntos1, const Mat &imagen2, vector<Point2f> &puntos2, int ladoVentana);

void ejercicio_correspondencias();
void ejercicio_autovalores();
void ejercicio_pintar_punto();
void ejercicio_harris();
void flujo();

void pintaImagen(std::string ventana, Mat imagen)
{
    namedWindow(ventana);
    Mat imagen2 = Mat(Size(512, 512), imagen.type());
    imshow(ventana, imagen);
    waitKey();
    destroyWindow(ventana);
}

void pintaPunto(Mat imagen, Point punto, int radio, Scalar color)
{
    circle(imagen, punto, radio, color, -1);
}

vector<Point2f> eigen(const Mat &imagen, int ladoVentana , int distancia, float calidad, float k, int esquinas)
{
    vector<Point2f> puntos;
    goodFeaturesToTrack(imagen, puntos, esquinas, calidad, distancia, noArray(), ladoVentana, false, k);
    return puntos;
}

vector<Point2f> harris(const Mat &imagen, int ladoVentana, int distancia, float calidad, float k, int esquinas)
{
    vector<Point2f> puntos;
    vector<Point2d> puntos2;
    //nivel de calidad->busca el maximo y se queda con aquellos puntos que estan entre [0,1]*maximo
    //float calidad = 0.1;
    //int distancia = 5;//puntuacion inducida por los puntos que estan en la zona de un punto fuerte
    //float k = 0.04;
    goodFeaturesToTrack(imagen, puntos, esquinas, calidad, distancia, noArray(), ladoVentana, true, k);
    //la ventana del cornersubpix es el entorno en el que se va a buscar el punto real
    cornerSubPix(imagen, puntos, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.001));
    return puntos;
}

vector<Point2i> correlacion(const Mat &imagen1, vector<Point2f> &puntosImagen1, const Mat &imagen2, vector<Point2f> &puntosImagen2, int ladoVentana)
{
    //Size ventana = Size(ladoVentana, ladoVentana);
    float distanciaMaxima = 10;
    int offsetX = -ladoVentana/2;
    int offsetY = -ladoVentana/2;
    Mat valor;
    vector<Point2f> *puntosColumnas;
    vector<Point2f> *puntosFilas;
    Mat imagenColumnas, imagenFilas;
    bool puntosImagen1EsFilas;
    //para hacer mas facil el rellenado de la matriz se van a colocar como columnas el mayor conjunto de puntos
    if (puntosImagen1.size() > puntosImagen2.size()) {
        imagenColumnas = imagen1;
        puntosColumnas = &puntosImagen1;
        imagenFilas = imagen2;
        puntosFilas = &puntosImagen2;
        puntosImagen1EsFilas = false;
    } else {
        imagenFilas = imagen1;
        puntosFilas = &puntosImagen1;
        imagenColumnas = imagen2;
        puntosColumnas = &puntosImagen2;
        puntosImagen1EsFilas = true;
    }

    //float matrizCorrelacion[puntosFilas->size()][puntosColumnas->size()];
    float **matrizCorrelacion = new float* [puntosFilas->size()];
    for (uint i = 0; i < puntosFilas->size(); i++){
        matrizCorrelacion[i] = new float[puntosColumnas->size()];
    }

    for (uint i = 0; i < puntosFilas->size(); i++) {
        //cout << (int)puntosFilas->at(i).x + offsetX << ' ' << (int)puntosFilas->at(i).y + offsetY << endl;
        if (!(puntosFilas->at(i).x + offsetX >=0 && puntosFilas->at(i).x + ladoVentana/2< imagenFilas.cols
                && puntosFilas->at(i).y + offsetY >= 0 && puntosFilas->at(i).y + ladoVentana/2 < imagenFilas.rows)) {
            //si la ventana cae fuera de la imagen, este punto no va a correlar con ninguno, salta al siguiente.
            for (uint j = i; j < puntosColumnas->size(); j++) {
                matrizCorrelacion[i][j] = -1;
                if (i < puntosColumnas->size() && j < puntosFilas->size()) {
                    //se duplica la zona diagonal inferior de la parte "cuadrada" de la matriz
                    matrizCorrelacion[j][i] = matrizCorrelacion[i][j];
                }
            }
            continue;
        }
        Mat region1 = Mat(imagenFilas, Rect((int)(puntosFilas->at(i).x + offsetX), (int)(puntosFilas->at(i).y + offsetY), ladoVentana, ladoVentana));
        //comenzandolo en i solo se calcula la parte que esta por encima o a la derecha de la diagonal
        for (uint j = i; j < puntosColumnas->size(); j++) {
            if (!(puntosColumnas->at(j).x + offsetX >= 0 && puntosColumnas->at(j).x + ladoVentana/2 < imagenColumnas.cols
                    && puntosColumnas->at(j).y + offsetY >= 0 && puntosColumnas->at(j).y + ladoVentana/2 < imagenColumnas.rows)) {
                matrizCorrelacion[i][j] = -1;
                //los bordes son peligrosos, si se sale fuera de la imagen -> correlacion -1 y salta al siguiente
                if (i < puntosColumnas->size() && j < puntosFilas->size()) {
                    //se duplica la zona diagonal inferior de la parte "cuadrada" de la matriz
                    matrizCorrelacion[j][i] = matrizCorrelacion[i][j];
                }
                continue;

            }
            //solo se miran los puntos que estan a como maximo distanciaMaxima

            float norma = sqrt(pow(puntosFilas->at(i).x - puntosColumnas->at(j).x, 2) + pow(puntosFilas->at(i).y - puntosColumnas->at(j).y, 2));
            if (norma > distanciaMaxima) {
                //si estan demasiado lejos -> correlacion = -1 y salta al siguiente
                matrizCorrelacion[i][j] = -1;
                if (i < puntosColumnas->size() && j < puntosFilas->size()) {
                    //se duplica la zona diagonal inferior de la parte "cuadrada" de la matriz
                    matrizCorrelacion[j][i] = matrizCorrelacion[i][j];
                }
                //continue;
            }

            Mat region2  = Mat(imagenColumnas, Rect((int)(puntosColumnas->at(j).x + offsetX), (int)puntosColumnas->at(j).y + offsetY, ladoVentana, ladoVentana));
            matchTemplate(region1, region2, valor, CV_TM_CCORR_NORMED);
            if (valor.at<float>(0, 0) > 0.95) {
                matrizCorrelacion[i][j] = valor.at<float>(0, 0);
            } else {
                matrizCorrelacion[i][j] = -1;
            }

            if (i < puntosColumnas->size() && j < puntosFilas->size()) {
                //se duplica la zona diagonal inferior de la parte "cuadrada" de la matriz
                matrizCorrelacion[j][i] = matrizCorrelacion[i][j];
            }
        }
    }

    vector<uint>maximosFila, maximosColumna;
    float maximo;
    int indiceMaximo;

    //maximos de las columnas
    for (uint i = 0; i < puntosFilas->size(); i++) {
        maximo = -2;
        indiceMaximo = -1;
        for (uint j = 0; j < puntosColumnas->size(); j++) {
            if (maximo <= matrizCorrelacion[i][j]) {
                if (maximo == matrizCorrelacion[i][j]) {
                    //significa que tenemos dos maximos, por ahora, no interesa
                    maximo = -1;
                    indiceMaximo = -1;
                } else {
                    indiceMaximo = j;
                    maximo = matrizCorrelacion[i][j];
                }

            }
        }
        if (maximo < 0.8) {
            indiceMaximo = -1;
        }
        maximosFila.push_back(indiceMaximo);
    }

    //maximos de las filas
    for (uint j = 0; j < puntosColumnas->size(); j++) {
        maximo = -1;
        indiceMaximo = -2;
        for (uint i = 0; i < puntosFilas->size(); i++) {
            if (maximo <= matrizCorrelacion[i][j]) {
                if (maximo == matrizCorrelacion[i][j]) {
                    //significa que tenemos dos maximos, por ahora, no interesa
                    maximo = -1;
                    indiceMaximo = -1;
                } else {
                    indiceMaximo = i;
                    maximo = matrizCorrelacion[i][j];
                }
            }
        }
        if (maximo < 0.8) {
            indiceMaximo = -1;
        }

        maximosColumna.push_back(indiceMaximo);
        if (indiceMaximo == -1){
            cout << maximosColumna.back() << endl;
        }
    }

    vector<Point2i> correspondencias;

    for (uint i = 0; i < maximosFila.size(); i++) {
        if (maximosColumna[maximosFila[i]] == i) {
            //de esta manera el x de cada tupla es el indice que corresponde a interes1
            if (puntosImagen1EsFilas) {
                correspondencias.push_back(Point2d(i, maximosFila[i]));
            } else {
                correspondencias.push_back(Point2d(maximosFila[i], i));
            }
        }
    }

    for (uint i = 0; i < puntosFilas->size(); i++){
       delete [] matrizCorrelacion[i];
    }
    delete [] matrizCorrelacion;

    return correspondencias;
}

void flujo()
{
    namedWindow("Flujo");

    int nFrames = 10;
    Mat *frames = new Mat[nFrames];
    frames[0] = imread("imagenes/FRAME1.bmp");
    frames[1] = imread("imagenes/FRAME2.bmp");
    frames[2] = imread("imagenes/FRAME3.bmp");
    frames[3] = imread("imagenes/FRAME4.bmp");
    frames[4] = imread("imagenes/FRAME5.bmp");
    frames[5] = imread("imagenes/FRAME6.bmp");
    frames[6] = imread("imagenes/FRAME7.bmp");
    frames[7] = imread("imagenes/FRAME8.bmp");
    frames[8] = imread("imagenes/FRAME9.bmp");
    frames[9] = imread("imagenes/FRAME10.bmp");
    int ladoVentana = 9;
    int distancia = 5;
    float calidad = 0.001;
    float k = 0.04;
    /****************************/
    //para hacerlo con la webcam
    /*
    VideoCapture cam(0);
    Mat frame;
    for (int i =0 ; i < nFrames; i++) {
        cam >> frame;
    frame.copyTo(frames[i]);
    waitKey(1);
    }
    */
    /***************************/



    for (int i=0; i<nFrames; i++) {
        cvtColor(frames[i], frames[i], CV_RGB2GRAY);
    }

    vector<Point2f> puntos_harris = harris(frames[0], ladoVentana, distancia, calidad, k, 1000);

    Mat auxFrame;
    frames[0].copyTo(auxFrame);
    cvtColor(auxFrame, auxFrame, CV_GRAY2RGB);
    for (vector<Point2f>::iterator it = puntos_harris.begin(); it!=puntos_harris.end(); it++) {
        pintaPunto(auxFrame, *it, 2, Scalar(0, 255, 0));
    }

    imshow("Flujo", auxFrame);
    waitKey();

    for (int i=1; i<nFrames; i++) {
        vector<Point2f> puntosHarrisDesplazados;
        vector<float> error;
        vector<uchar> status;

        calcOpticalFlowPyrLK(frames[i-1], frames[i], puntos_harris, puntosHarrisDesplazados, status, error);

        auxFrame = frames[i];
        cvtColor(auxFrame, auxFrame, CV_GRAY2RGB);
        //puntos harris del frame anterior
        for (vector<Point2f>::iterator it = puntos_harris.begin(); it != puntos_harris.end(); it++) {
            pintaPunto(auxFrame, *it, 2, Scalar(255, 0, 0));
        }

        //puntos correspondientes en el frame actual y recta de direccion del movimiento
        for (uint j=0; j < status.size(); j++) {
            if (status[j]) {
                pintaPunto(auxFrame, puntosHarrisDesplazados[j], 2, Scalar(0, 0, 255));
                line(auxFrame, puntos_harris[j], puntosHarrisDesplazados[j], Scalar(0, 255, 0));
                //esto era para mostrar el vector de tamaño fijo en lugar del vector del tamaño del movimiento
                //Point2f vDesplazamiento(puntosHarrisDesplazados[j].x - puntos_harris[j].x, puntosHarrisDesplazados[j].y - puntos_harris[j].y);
                //float moduloDesplazamiento = sqrt(vDesplazamiento.x * vDesplazamiento.x + vDesplazamiento.y * vDesplazamiento.y);
                //vDesplazamiento.x /= moduloDesplazamiento;
                //vDesplazamiento.y /= moduloDesplazamiento;
                //Point2f puntoRectaDesplazamiento(15 * vDesplazamiento.x + puntos_harris[j].x, 15 * vDesplazamiento.y + puntos_harris[j].y);
                //line(auxFrame, puntos_harris[j], puntoRectaDesplazamiento, Scalar(0, 255, 0));
            }
        }
        imshow("Flujo", auxFrame);
        //no se deben recalcular los puntos harris
        //puntos_harris = harris(frames[i], ladoVentana, distancia, calidad, k, 1000);
        puntos_harris.clear();
        for (uint j=0; j < status.size(); j++) {
            if (status[j]) {
                puntos_harris.push_back(puntosHarrisDesplazados[j]);
            }
        }
        waitKey();
    }

    delete [] frames;

    return;
}

void ejercicio_pintar_punto()
{
    Mat im;
    im = imread("imagenes/lena.jpg");
    pintaPunto(im, Point2d(im.cols/2, im.rows/2), 10, Scalar(0, 255, 0));
    pintaImagen("Pintar punto", im);
}

void ejercicio_harris()
{
    int ladoVentana = 9;
    int distancia = 5;
    float calidad = 0.01;
    float k = 0.04;

    Mat imagen1, imagenGris1;
    imagen1 = imread("imagenes/lena.jpg");

    cvtColor(imagen1, imagenGris1, CV_BGR2GRAY);

    vector<Point2f> puntosHarrisImagen1 = harris(imagenGris1, ladoVentana, distancia, calidad, k, 1000);

    for (vector<Point2f>::iterator it = puntosHarrisImagen1.begin(); it!=puntosHarrisImagen1.end(); it++) {
        pintaPunto(imagen1, *it, 1, Scalar(0, 255, 0));
    }

    pintaImagen("Harris", imagen1);
}

void ejercicio_autovalores()
{
    int ladoVentana = 9;
    int distancia = 5;
    float calidad = 0.001;
    float k = 0.04;
    int puntos = 1000;

    Mat imagen1, imagenGris1;
    imagen1 = imread("imagenes/lena.jpg");

    cvtColor(imagen1, imagenGris1, CV_BGR2GRAY);

    vector<Point2f> puntosEigen = eigen(imagenGris1, ladoVentana, distancia, calidad, k, puntos);

    for (vector<Point2f>::iterator it = puntosEigen.begin(); it != puntosEigen.end(); it++) {
        pintaPunto(imagen1, *it, 1, Scalar(0, 255, 0));
    }

    pintaImagen("Autovalores", imagen1);
}

void ejercicio_correspondencias()
{
    int ladoVentana = 9;
    int distancia = ladoVentana;
    float calidad = 0.001;
    float k = 0.04;
    int puntos = 1000;

    Mat imagen1, imagen2, imagenGris1, imagenGris2;
    imagen1 = imread("imagenes/lena.jpg");
    imagen2 = imread("imagenes/lena.jpg");

    cvtColor(imagen1, imagenGris1, CV_BGR2GRAY);
    cvtColor(imagen2, imagenGris2, CV_BGR2GRAY);

    vector<Point2f> puntosHarrisImagen1 = harris(imagenGris1, ladoVentana, distancia, calidad, k, puntos);
    vector<Point2f> puntosHarrisImagen2 = harris(imagenGris2, ladoVentana, distancia, calidad, k, puntos);

    for (vector<Point2f>::iterator it = puntosHarrisImagen2.begin(); it!=puntosHarrisImagen2.end(); it++) {
        pintaPunto(imagen2, *it, 1, Scalar(0, 255, 0));
    }

    for (vector<Point2f>::iterator it = puntosHarrisImagen1.begin(); it!=puntosHarrisImagen1.end(); it++) {
        pintaPunto(imagen1, *it, 1, Scalar(0, 255, 0));
    }

    Mat compuesta (max(imagen1.rows, imagen2.rows), imagen1.cols+imagen2.cols, imagen1.type());
    Mat fragmentoCompuesta1(compuesta, Range(0, imagen1.rows), Range(0, imagen1.cols));
    Mat fragmentoCompuesta2(compuesta, Range(0, imagen2.rows), Range(imagen1.cols, imagen1.cols + imagen2.cols));

    imagen1.copyTo(fragmentoCompuesta1);
    imagen2.copyTo(fragmentoCompuesta2);

    vector<Point2i> correspondencias = correlacion(imagenGris1, puntosHarrisImagen1, imagenGris2, puntosHarrisImagen2, ladoVentana);

    for (uint i=0; i < puntosHarrisImagen2.size(); i++) {
        puntosHarrisImagen2[i].x += imagen1.cols;
    }

    for (uint i=0; i<correspondencias.size(); i++) {
        line(compuesta, puntosHarrisImagen1[correspondencias[i].x], puntosHarrisImagen2[correspondencias[i].y], Scalar(255));
    }

    pintaImagen("Correlacion", compuesta);
}

int main()
{
    ejercicio_pintar_punto();
    waitKey();
    ejercicio_autovalores();
    waitKey();
    ejercicio_harris();
    waitKey();
    ejercicio_correspondencias();
    waitKey();
    flujo();
    waitKey();
    return 0;
}



// kate: indent-mode cstyle; space-indent on; indent-width 0;
