#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

typedef struct {
    double error;
    Mat k;
    Mat distCoeffs;
    vector<Mat> rvects;
    vector<Mat> tvects;
} datosCalibracionCamara;

void pintaImagen(Mat imagen, std::string ventana);
pair<vector<Mat>, vector<Mat> > leerImagenes ();
vector<vector<Point2f> > buscarPuntosTableros(const vector<Mat> imagenes, Size tablero);
// pair<double, pair<Mat, Mat> > calibrarCamara(vector<vector<Point2f> > puntos, Size tablero, Size dimensionImagenes);
datosCalibracionCamara calibrarCamara(vector<vector<Point2f> > puntos, Size tablero, Size dimensionImagenes);
pair<pair<Mat, Mat>, pair<Mat, Mat> > calibrarCamaras(const pair<vector<Mat>, vector<Mat> > &imagenes);

void pintaImagen(Mat imagen, std::string ventana) {
    namedWindow(ventana);
    imshow(ventana, imagen);
    waitKey();
    destroyWindow(ventana);
}

pair<vector<Mat>, vector<Mat> > leerImagenes () {
    pair<vector<Mat>, vector<Mat> > imagenes;
    imagenes.first = vector<Mat>();
    imagenes.second = vector<Mat>();
    int n_imagenes = 14;
    char ruta[50];
    for (int i = 0; i < n_imagenes; i++) {
        Mat aux;
        sprintf(ruta, "./imagenes/left%02d.jpg", i+1);
        aux = imread(ruta);
        cvtColor(aux, aux, CV_RGB2GRAY);
        imagenes.first.push_back(aux);

        sprintf(ruta, "./imagenes/right%02d.jpg", i+1);
        aux = imread(ruta);
        cvtColor(aux, aux, CV_RGB2GRAY);
        imagenes.second.push_back(aux);
    }

    return imagenes;
}

vector<vector<Point2f> > buscarPuntosTablero(const vector<Mat> imagenes, Size tablero){
    vector<vector<Point2f> > conjuntoPuntosFotos;
    for(uint i=0; i< imagenes.size(); i++){
        vector<Point2f> puntosFoto;
        puntosFoto.clear();
        bool valida = findChessboardCorners(imagenes[i], tablero, puntosFoto);
        if(valida){
            cornerSubPix(imagenes[i], puntosFoto, Size(5, 5), Size(-1, -1),
                         TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100000, 0.00001));
            conjuntoPuntosFotos.push_back(puntosFoto);
        }

//#define DEBUG
#ifdef DEBUG
        Mat aux;
        imagenes[i].copyTo(aux);
        drawChessboardCorners(aux, tablero, puntosFoto, valida);
        pintaImagen(aux, "IZQ");
#endif

    }
    return conjuntoPuntosFotos;
}

typedef struct{

} calibracionStereo;

pair<Mat, Mat> calcularRTcaso1(Size tablero, vector<vector<Point2f> > puntos1, vector<vector<Point2f> > puntos2,
                     Size tamImagenes, pair<Mat,Mat> camaras, pair<Mat, Mat> distCoeffs){
    vector<vector<Point3f> > mundos;
    mundos.push_back(vector<Point3f>());

    vector<vector<Point2f> >puntos1Validos, puntos2Validos;
    //parejas de puntos donde ambas imagenes son validas, los vectores vacios
    //indican que la imagen no fue apta para calibracion
    for (uint i=0; i<puntos1.size(); i++){
        if (!(puntos1.empty() || puntos2.empty())){
            puntos1Validos.push_back(puntos1[i]);
            puntos2Validos.push_back(puntos2[i]);
        }
    }

    //genera la cuadricula
    for(int i = 0; i < tablero.height; i++){
        for(int j = 0; j < tablero.width; j++){
            mundos.back().push_back(Point3f(i * 30, j * 30, 0));
        }
    }

    //replica los datos de la cuadricula
    for (uint i=1; i<puntos1Validos.size(); i++){
        mundos.push_back(mundos.front());
    }

    Mat r, t, e, f;
    stereoCalibrate(mundos, puntos1Validos, puntos2Validos, camaras.first, distCoeffs.first, camaras.second, distCoeffs.second, tamImagenes, r, t, e, f);
    Rodrigues(r, r);
    return pair<Mat, Mat> (r, t);
}

pair<pair<Mat, Mat>, pair<Mat, Mat> > calibrarCamaras(const pair<vector<Mat>, vector<Mat> > &imagenes){
    Size tablero(9, 6);
    vector<vector<Point2f> > conjuntoPuntosIzquierda = buscarPuntosTablero(imagenes.first,  tablero);
    vector<vector<Point2f> > conjuntoPuntosDerecha   = buscarPuntosTablero(imagenes.second, tablero);
    datosCalibracionCamara datosCalibracionIzquierda =  calibrarCamara(conjuntoPuntosIzquierda, tablero, imagenes.first.front().size());
    datosCalibracionCamara datosCalibracionDerecha   =  calibrarCamara(conjuntoPuntosDerecha,   tablero, imagenes.first.front().size());

    Mat rotacionMedia = Mat::zeros(3, 1, 6);
    Mat traslacionMedia  = Mat::zeros(3, 1, 6);
    cout << "solvePnP:\n";
    cout << "\t[R] [T]:\n";
    for (uint i=0; i<datosCalibracionDerecha.rvects.size(); i++){
        Mat rIzq, rDer;
        Rodrigues(datosCalibracionIzquierda.rvects[i], rIzq);
        Rodrigues(datosCalibracionDerecha.rvects[i], rDer);
        Mat r = rDer * rIzq.t();
        Mat t = datosCalibracionDerecha.tvects[i] - r * datosCalibracionIzquierda.tvects[i];
        Rodrigues(r, r);
        cout << "\t\t" << setprecision(20) << r << ' ' << setprecision(20) <<t << endl;
        rotacionMedia += r;
        traslacionMedia +=t;
    }
    cout << "\n\tMedias:   " << rotacionMedia/datosCalibracionDerecha.rvects.size() << " " << traslacionMedia/datosCalibracionDerecha.rvects.size() << endl;
    cout << "===========================================================================================================================================\n";
    cout << "Matrices de las camaras\n";
    cout << "Izquierda\n";
    cout << datosCalibracionIzquierda.k << endl;
    cout << "Derecha\n";
    cout << datosCalibracionDerecha.k << endl;

    pair<Mat, Mat> r_t = calcularRTcaso1(tablero, conjuntoPuntosIzquierda, conjuntoPuntosDerecha, imagenes.first.front().size(),
                                         pair<Mat, Mat>(datosCalibracionIzquierda.k, datosCalibracionDerecha.k),
                                         pair<Mat, Mat>(datosCalibracionIzquierda.distCoeffs, datosCalibracionDerecha.distCoeffs));
    cout << "\nValores [R] [T] calculados con stereoCalibrate\n";
    cout << "\t" << r_t.first  << " "<< r_t.second << endl;
    return pair<pair<Mat, Mat>, pair<Mat, Mat> >(pair<Mat, Mat>(datosCalibracionIzquierda.k, datosCalibracionIzquierda.distCoeffs),
                                                 pair<Mat, Mat>(datosCalibracionDerecha.k,   datosCalibracionDerecha.distCoeffs));
}

datosCalibracionCamara calibrarCamara(vector<vector<Point2f> > puntos, Size tablero, Size dimensionImagenes){
    vector<vector<Point2f> > puntosValidos;
    for (uint i=0; i<puntos.size(); i++){
        if (!puntos[i].empty()){
            puntosValidos.push_back(puntos[i]);
        }
    }

    vector<vector<Point3f> > mundos;
    mundos.push_back(vector<Point3f>());
    //genera la cuadricula
    for(int i = 0; i < tablero.height; i++){
        for(int j = 0; j < tablero.width; j++){
            mundos.back().push_back(Point3f(i * 30, j * 30, 0));
        }
    }

    //replica los datos de la cuadricula
    for (uint i=1; i<puntosValidos.size(); i++){
        mundos.push_back(mundos.front());
    }

    //estima el valor inicial de la matriz de camara
    Mat camaraEstimada = initCameraMatrix2D(mundos, puntosValidos, dimensionImagenes);

    vector<Mat> rotaciones, translaciones;
    Mat distCoeffs;
    for (uint i=0; i < puntosValidos.size(); i++){
        rotaciones.push_back(Mat());
        translaciones.push_back(Mat());
        solvePnP(mundos[i], puntosValidos[i], camaraEstimada, distCoeffs, rotaciones.back(), translaciones.back(), 0);
    }
    distCoeffs.zeros(distCoeffs.size(), distCoeffs.type());
    //flags comunes
    int flags = CV_CALIB_USE_INTRINSIC_GUESS;
    //calibratecamera modifica los vectores calculadospor solvePNP
    double error = calibrateCamera(mundos, puntosValidos, dimensionImagenes, camaraEstimada, distCoeffs, rotaciones, translaciones, flags);

    datosCalibracionCamara datos;
    datos.error = error;
    datos.k = camaraEstimada;
    datos.distCoeffs = distCoeffs;
    datos.rvects = rotaciones;
    datos.tvects = translaciones;
    return datos;
}

pair<vector<Mat>, vector<Mat> > rectificarImagenes(pair<vector<Mat>, vector<Mat> > imagenes, Mat r, Mat t, Mat k1, Mat k2, Mat distCoeffs1, Mat distCoeffs2, Mat &q){
    Mat rI, rD, pI, pD;
    stereoRectify(k1, distCoeffs1, k2, distCoeffs2, imagenes.first.front().size(), r, t, rI, rD, pI, pD, q, CALIB_ZERO_DISPARITY, -1);

    Mat  map1I, map2I;
    vector<Mat> imagenesIzquierdaRectificadas;
    for (uint i=0; i<imagenes.first.size(); i++){
        imagenesIzquierdaRectificadas.push_back(Mat());
        initUndistortRectifyMap(k1, distCoeffs1, rI, pI, imagenes.first[i].size(), CV_32FC1, map1I, map2I);
        remap(imagenes.first[i], imagenesIzquierdaRectificadas.back(), map1I, map2I, INTER_AREA);
    }

    Mat  map1D, map2D;
    vector<Mat> imagenesDerechaRectificadas;
    for (uint i=0; i<imagenes.second.size(); i++){
        imagenesDerechaRectificadas.push_back(Mat());
        initUndistortRectifyMap(k2, distCoeffs2, rD, pD, imagenes.second[i].size(), CV_32FC1, map1D, map2D);
        remap(imagenes.second[i], imagenesDerechaRectificadas.back(), map1D, map2D, INTER_AREA);
    }

    return pair<vector<Mat>, vector<Mat> >(imagenesIzquierdaRectificadas, imagenesDerechaRectificadas);
}

void mostrarParesImagenes(pair<vector<Mat>, vector<Mat> > imagenes){
    for (uint i=0; i<imagenes.first.size(); i++){
        Mat imCompuesta(max(imagenes.first[i].rows, imagenes.second[i].rows),
                        imagenes.first[i].cols + imagenes.second[i].cols, imagenes.first[i].type());

        Mat fragmentoCompuesta1(imCompuesta, Range(0, imagenes.first[i].rows), Range(0, imagenes.first[i].cols));

        Mat fragmentoCompuesta2(imCompuesta, Range(0, imagenes.second[i].rows),
                                Range(imagenes.first[i].cols, imagenes.first[i].cols + imagenes.second[i].cols));

        imagenes.first[i].copyTo(fragmentoCompuesta1);
        imagenes.second[i].copyTo(fragmentoCompuesta2);
        line(imCompuesta, Point2i(0, 100), Point2i(imCompuesta.cols, 100), Scalar(1), 1);
        imshow("Par", imCompuesta);
        waitKey();
        destroyWindow("Par");
    }
}


vector<Mat> mapaDisparidad(pair<vector<Mat>, vector<Mat> > imagenes){
     StereoSGBM sgbm;
    sgbm.SADWindowSize = 31;
    sgbm.P1 = 8 * sgbm.SADWindowSize * sgbm.SADWindowSize;
    sgbm.P2 = 32 * sgbm.SADWindowSize * sgbm.SADWindowSize;
    sgbm.minDisparity = 0;
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = 20;
    sgbm.speckleRange = 32;
    sgbm.disp12MaxDiff = 1;
    sgbm.numberOfDisparities=64;
    sgbm.fullDP=1;

    vector<Mat> disparidades;
    for (uint i=0; i < imagenes.first.size(); i++){
        disparidades.push_back(Mat());
        sgbm(imagenes.first[i], imagenes.second[i], disparidades.back());
        Mat disp8;

        normalize(disparidades.back(), disp8, 0, 255, NORM_MINMAX);
        disp8.convertTo(disp8, CV_8U);
        imshow("Izquierda", imagenes.first[i]);
        imshow("Derecha",  imagenes.second[i]);
        imshow("disp8", disp8);
        waitKey();
    }
    return disparidades;
}

void reconstruccion3D(vector<Mat> disparidades, Mat q){
    vector<Mat> reconstrucciones;
    for (uint i=0; i < disparidades.size(); i++){
        reconstrucciones.push_back(Mat());
        reprojectImageTo3D(disparidades[i], reconstrucciones.back(), q);
    }
}

int main() {
    cout << fixed;
    pair<vector<Mat>, vector<Mat> > imagenes = leerImagenes();
    pair<pair<Mat, Mat>, pair<Mat, Mat> > datosCalibracion = calibrarCamaras(imagenes);

    Mat k1 = (Mat_<double>(3, 3) << 533.52331, 0, 341.60376, 0, 533.52699, 235.19287, 0, 0, 1);
    Mat k2 = (Mat_<double>(3, 3) << 536.81377, 0, 326.28657, 0, 536.47649, 250.10121, 0, 0, 1);
    Mat distCoeffs1 = (Mat_<double>(1, 4) << -0.28838, 0.09714, 0.00109, -0.00030);
    Mat distCoeffs2 = (Mat_<double>(1, 4) << -0.28943, 0.10690, -0.00059, 0.00014);
    Mat r = (Mat_<double>(3, 1) << 0.00669, 0.00452, -0.00350);
    Mat t = (Mat_<double>(3, 1) << -99.80198,  1.12443, 0.05041);
    Mat q;
    pair<vector<Mat>, vector<Mat> > imagenesRectificadas = rectificarImagenes(imagenes, r, t, k1, k2, distCoeffs1, distCoeffs2, q);
    mostrarParesImagenes(imagenesRectificadas);
    vector<Mat>  disparidades = mapaDisparidad(imagenesRectificadas);
    reconstruccion3D(disparidades, q);
    return 0;
}

/*

1.Calibrar un par estéreo a partir de los datos de calibración de las cámaras individuales.
    - Estimar los parámetros intrínsecos y las lentes  de cada cámara individual.
    Calcular los valores para (R,t) fijando los valores de cada cámara en la calibración del par estéreo(stereoCalibrate()).
    Ver que el valor obtenido para  (R,T) del par es igual al que obtenemos llamando a solvePnP() usando
    los puntos en correspondencias  de cada par. Mostrar valores . 1 punto

2.Estimar las imágenes rectificadas de cada par estéreo. Mostrar las imágenes(1punto)
(stereoRectify,initUndistortRectifyMap())

3.Calcular el mapa de disparidad de cada par estéreo y visualizarlo.(0,5puntos)
(stereoSGBM()usar la parametrizacio dada en el fichero sgbm.txt)

4.Calcular la reconstrucción 3D  a partir de la disparidad. (0,5 puntos) (reprojectImageTo3D)

*/
// kate: indent-mode cstyle; space-indent on; indent-width 0;
